#!/usr/bin/env python
"""Standalone MACRec-like strict pipeline (independent implementation).

This script reproduces MACRec-style multi-step logic with an independent code path:
- Manager loop: Thought -> Action -> Observation -> ... -> Finish
- Action space: Search / Analyse / Finish
- Data: query_data1.csv + metadata.csv
- Eval: HR/NDCG@K

No MACRec imports, no langchain.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
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


def ndcg_at(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def item_text(row: pd.Series) -> str:
    return ' | '.join([
        f"title={safe_text(row.get('title', ''))}",
        f"desc={safe_text(row.get('description', ''))}",
        f"cate={safe_text(row.get('category', ''))}",
        f"price={safe_text(row.get('price', ''))}",
        f"ranking={safe_text(row.get('ranking', ''))}",
    ])


class Corpus:
    def __init__(self, metadata_file: str):
        df = pd.read_csv(metadata_file)
        df['id'] = df['id'].astype(str)
        df['doc_text'] = df.apply(item_text, axis=1)
        self.df = df
        self.id_to_doc = dict(zip(df['id'].tolist(), df['doc_text'].tolist()))
        self.id_to_idx = {iid: i for i, iid in enumerate(df['id'].tolist())}
        self.vec = TfidfVectorizer(lowercase=True, stop_words='english', max_features=120000)
        self.X = self.vec.fit_transform(df['doc_text'])

    def retrieve(self, query: str, history_text: str, blocked_ids: set[str], topn: int) -> list[str]:
        q = self.vec.transform([query])
        s = 0.75 * linear_kernel(q, self.X).ravel()
        if history_text:
            h = self.vec.transform([history_text])
            s += 0.25 * linear_kernel(h, self.X).ravel()
        for iid in blocked_ids:
            j = self.id_to_idx.get(iid)
            if j is not None:
                s[j] = -1e12
        n = min(topn, len(s))
        idx = np.argpartition(-s, n - 1)[:n]
        idx = idx[np.argsort(-s[idx])]
        return self.df.iloc[idx]['id'].tolist()

    def score_candidates(self, query: str, history_text: str, candidates: list[str]) -> list[tuple[str, float]]:
        docs = [self.id_to_doc[iid] for iid in candidates]
        Xc = self.vec.transform(docs)
        q = self.vec.transform([query])
        s = 0.75 * linear_kernel(q, Xc).ravel()
        if history_text:
            h = self.vec.transform([history_text])
            s += 0.25 * linear_kernel(h, Xc).ravel()
        pairs = list(zip(candidates, s.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


class QwenPolicy:
    def __init__(self, model_name: str, max_new_tokens: int = 256, enable_thinking: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto',
        )
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

    def ask(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if '</think>' in output:
            output = output.split('</think>')[-1].strip()
        return output


@dataclass
class ManagerState:
    scratchpad: str = ''
    finished: bool = False
    final_rank: list[str] | None = None


class StrictStandaloneAgent:
    def __init__(self, corpus: Corpus, policy: str = 'heuristic', model_name: str = 'Qwen/Qwen3-8B', max_step: int = 4, enable_thinking: bool = True):
        self.corpus = corpus
        self.policy = policy
        self.max_step = max_step
        self.qwen = None
        if policy == 'qwen':
            self.qwen = QwenPolicy(model_name=model_name, enable_thinking=enable_thinking)

    @staticmethod
    def _parse_action(text: str) -> tuple[str, str]:
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            return 'Analyse', ''
        try:
            obj = json.loads(m.group(0))
            a = str(obj.get('action', 'Analyse')).strip()
            arg = str(obj.get('argument', '')).strip()
            return a, arg
        except Exception:
            return 'Analyse', ''

    def _manager_action(self, query: str, history: str, step: int, scratchpad: str) -> tuple[str, str]:
        if self.policy == 'heuristic':
            if step == 1:
                return 'Search', query
            if step == 2:
                return 'Analyse', ''
            return 'Finish', ''

        prompt = (
            "You are a recommender manager. Choose ONE action in JSON only.\n"
            "Valid actions: Search, Analyse, Finish.\n"
            "Format: {\"action\": \"Search|Analyse|Finish\", \"argument\": \"...\"}.\n"
            f"User requirement: {query}\n"
            f"History: {history[:800]}\n"
            f"Scratchpad: {scratchpad[-1500:]}\n"
        )
        txt = self.qwen.ask(prompt)
        return self._parse_action(txt)

    def run_one(self, query: str, history_ids: list[str], preselect_k: int = 80, out_k: int = 40) -> list[str]:
        history_docs = [self.corpus.id_to_doc[x] for x in history_ids if x in self.corpus.id_to_doc]
        history_text = ' '.join(history_docs)
        blocked = set(history_ids)

        state = ManagerState()
        cand_ids: list[str] = []

        for step in range(1, self.max_step + 1):
            action, arg = self._manager_action(query, history_text, step, state.scratchpad)
            if action.lower() == 'search':
                cand_ids = self.corpus.retrieve(query=arg or query, history_text=history_text, blocked_ids=blocked, topn=preselect_k)
                obs = f"searched {len(cand_ids)} candidates"
            elif action.lower() == 'analyse':
                if not cand_ids:
                    cand_ids = self.corpus.retrieve(query=query, history_text=history_text, blocked_ids=blocked, topn=preselect_k)
                scored = self.corpus.score_candidates(query=query, history_text=history_text, candidates=cand_ids)
                cand_ids = [iid for iid, _ in scored]
                obs = 'analysed and reranked candidates'
            elif action.lower() == 'finish':
                state.finished = True
                if not cand_ids:
                    cand_ids = self.corpus.retrieve(query=query, history_text=history_text, blocked_ids=blocked, topn=preselect_k)
                state.final_rank = cand_ids[:out_k]
                break
            else:
                obs = f'invalid action {action}, fallback analyse'

            state.scratchpad += f"\nThought {step}: ...\nAction {step}: {action}({arg})\nObservation: {obs}\n"

        if state.final_rank is None:
            if not cand_ids:
                cand_ids = self.corpus.retrieve(query=query, history_text=history_text, blocked_ids=blocked, topn=preselect_k)
            state.final_rank = cand_ids[:out_k]

        return state.final_rank


def run(args):
    qdf = pd.read_csv(args.query_file)
    qdf['id'] = qdf['id'].astype(str)
    corpus = Corpus(args.metadata_file)
    agent = StrictStandaloneAgent(
        corpus=corpus,
        policy=args.policy,
        model_name=args.model_name,
        max_step=args.max_step,
        enable_thinking=args.enable_thinking,
    )

    topks = sorted(set(args.topks))
    maxk = max(topks)
    n = len(qdf) if args.max_samples <= 0 else min(len(qdf), args.max_samples)

    hr = {k: 0.0 for k in topks}
    ndcg = {k: 0.0 for k in topks}
    rows = []

    for i in range(n):
        row = qdf.iloc[i]
        target = row['id']
        query = safe_text(row.get('new_query', '')) or safe_text(row.get('query', ''))
        history_ids = [x.strip() for x in safe_text(row.get('remaining_interaction_string', '')).split('|') if x.strip()]

        rank_list = agent.run_one(query=query, history_ids=history_ids, preselect_k=args.preselect_k, out_k=maxk)
        rank = rank_list.index(target) + 1 if target in rank_list else 0

        for k in topks:
            hr[k] += 1.0 if 0 < rank <= k else 0.0
            ndcg[k] += ndcg_at(rank, k)

        rows.append({
            'row_index': i,
            'user_id': safe_text(row.get('user_id', '')),
            'target_id': target,
            'target_rank': rank,
            'recommendation_list': rank_list,
        })

    metrics = {
        'samples': n,
        'policy': args.policy,
        'topks': topks,
        'HR': {f'HR@{k}': hr[k] / max(n, 1) for k in topks},
        'NDCG': {f'NDCG@{k}': ndcg[k] / max(n, 1) for k in topks},
    }

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--query_file', required=True)
    p.add_argument('--metadata_file', required=True)
    p.add_argument('--output_file', default='run/amazon_beauty/strict_standalone.jsonl')
    p.add_argument('--metrics_file', default='run/amazon_beauty/strict_standalone_metrics.json')
    p.add_argument('--topks', type=int, nargs='+', default=[10, 20, 40])
    p.add_argument('--max_samples', type=int, default=0)

    p.add_argument('--policy', choices=['heuristic', 'qwen'], default='heuristic')
    p.add_argument('--model_name', default='Qwen/Qwen3-8B')
    p.add_argument('--enable_thinking', action='store_true')
    p.add_argument('--max_step', type=int, default=4)
    p.add_argument('--preselect_k', type=int, default=80)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
