#!/usr/bin/env python
"""Standalone pipeline for Amazon query_data1.csv + metadata.csv.

Goals:
1) Retrieve from full metadata pool using query + interaction history.
2) (Optional) Use Transformers-based Qwen model to rerank top candidates.
3) Evaluate HR/NDCG at K based on target rank.

No dependency on MACRec internals or langchain.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def safe_text(v) -> str:
    if v is None:
        return ''
    if isinstance(v, float) and math.isnan(v):
        return ''
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


def item_to_text(row: pd.Series) -> str:
    return ' '.join([
        safe_text(row.get('title', '')),
        safe_text(row.get('description', '')),
        safe_text(row.get('category', '')),
        f"price {safe_text(row.get('price', ''))}",
        f"ranking {safe_text(row.get('ranking', ''))}",
    ])


def ndcg_at(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def build_retriever(metadata_df: pd.DataFrame):
    metadata_df = metadata_df.copy()
    metadata_df['id'] = metadata_df['id'].astype(str)
    metadata_df['doc_text'] = metadata_df.apply(item_to_text, axis=1)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=120000)
    doc_matrix = vectorizer.fit_transform(metadata_df['doc_text'])

    id_to_index = {iid: i for i, iid in enumerate(metadata_df['id'].tolist())}
    id_to_doc = dict(zip(metadata_df['id'].tolist(), metadata_df['doc_text'].tolist()))
    return metadata_df, vectorizer, doc_matrix, id_to_index, id_to_doc


class QwenReranker:
    def __init__(self, model_name: str = 'Qwen/Qwen3-8B', max_new_tokens: int = 256, enable_thinking: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto',
        )
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

    def rerank(self, query: str, history: str, candidates: List[dict]) -> List[str]:
        cand_lines = '\n'.join([f"{c['id']}: {c['text'][:220]}" for c in candidates])
        prompt = (
            "You are a recommendation ranker. "
            "Given user requirement and interaction history, reorder candidate item ids by relevance. "
            "Return only a JSON array of item ids.\n\n"
            f"[Requirement]\n{query}\n\n"
            f"[History]\n{history if history else 'None'}\n\n"
            f"[Candidates]\n{cand_lines}\n"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if '</think>' in output:
            output = output.split('</think>')[-1].strip()

        # try parse json array
        left = output.find('[')
        right = output.rfind(']')
        if left == -1 or right == -1 or right <= left:
            return [c['id'] for c in candidates]
        part = output[left:right + 1]
        try:
            parsed = json.loads(part)
            parsed = [str(x) for x in parsed]
            valid = [x for x in parsed if x in {c['id'] for c in candidates}]
            rem = [c['id'] for c in candidates if c['id'] not in valid]
            return valid + rem
        except Exception:
            return [c['id'] for c in candidates]


def run_pipeline(args):
    query_df = pd.read_csv(args.query_file)
    metadata_df = pd.read_csv(args.metadata_file)
    query_df['id'] = query_df['id'].astype(str)

    metadata_df, vec, X, id_to_index, id_to_doc = build_retriever(metadata_df)

    reranker = None
    if args.use_qwen_rerank:
        reranker = QwenReranker(
            model_name=args.model_name,
            max_new_tokens=args.rerank_max_new_tokens,
            enable_thinking=args.enable_thinking,
        )

    topks = sorted(set(args.topks))
    max_k = max(topks)
    n = len(query_df) if args.max_samples <= 0 else min(len(query_df), args.max_samples)

    hr_sum = {k: 0.0 for k in topks}
    ndcg_sum = {k: 0.0 for k in topks}

    result_rows = []
    for i in range(n):
        row = query_df.iloc[i]
        target_id = row['id']
        query = safe_text(row.get('new_query', '')) or safe_text(row.get('query', ''))

        hist_ids = [x.strip() for x in safe_text(row.get('remaining_interaction_string', '')).split('|') if x.strip()]
        hist_docs = [id_to_doc[x] for x in hist_ids if x in id_to_doc]
        hist_text = ' '.join(hist_docs)

        qv = vec.transform([query])
        score = args.weight_query * linear_kernel(qv, X).ravel()
        if hist_text:
            hv = vec.transform([hist_text])
            score += args.weight_history * linear_kernel(hv, X).ravel()

        for hid in hist_ids:
            j = id_to_index.get(hid)
            if j is not None:
                score[j] = -1e12

        pre_topn = min(args.preselect_k, len(score))
        idx = np.argpartition(-score, pre_topn - 1)[:pre_topn]
        idx = idx[np.argsort(-score[idx])]
        pre_ids = metadata_df.iloc[idx]['id'].tolist()

        if reranker is None:
            ranked_ids = pre_ids[:max_k]
        else:
            cand = [{'id': iid, 'text': id_to_doc[iid]} for iid in pre_ids]
            reranked = reranker.rerank(query=query, history=hist_text, candidates=cand)
            ranked_ids = reranked[:max_k]

        rank = ranked_ids.index(target_id) + 1 if target_id in ranked_ids else 0

        for k in topks:
            hr_sum[k] += 1.0 if 0 < rank <= k else 0.0
            ndcg_sum[k] += ndcg_at(rank, k)

        result_rows.append({
            'row_index': i,
            'user_id': safe_text(row.get('user_id', '')),
            'target_id': target_id,
            'target_rank': rank,
            'recommendation_list': ranked_ids,
        })

    metrics = {
        'samples': n,
        'topks': topks,
        'mode': 'qwen_rerank' if reranker is not None else 'tfidf_only',
        'HR': {f'HR@{k}': hr_sum[k] / max(n, 1) for k in topks},
        'NDCG': {f'NDCG@{k}': ndcg_sum[k] / max(n, 1) for k in topks},
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    mpath = Path(args.metrics_file)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with mpath.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--query_file', required=True)
    p.add_argument('--metadata_file', required=True)
    p.add_argument('--output_file', default='run/amazon_beauty/standalone_retrieval.jsonl')
    p.add_argument('--metrics_file', default='run/amazon_beauty/standalone_metrics.json')
    p.add_argument('--topks', type=int, nargs='+', default=[10, 20, 40])
    p.add_argument('--max_samples', type=int, default=0)

    p.add_argument('--weight_query', type=float, default=0.75)
    p.add_argument('--weight_history', type=float, default=0.25)
    p.add_argument('--preselect_k', type=int, default=80)

    p.add_argument('--use_qwen_rerank', action='store_true')
    p.add_argument('--model_name', default='Qwen/Qwen3-8B')
    p.add_argument('--rerank_max_new_tokens', type=int, default=256)
    p.add_argument('--enable_thinking', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    run_pipeline(parse_args())
