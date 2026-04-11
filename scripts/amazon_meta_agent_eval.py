import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    text = str(value).strip()
    return '' if text.lower() == 'nan' else text


def build_metadata_index(metadata_df: pd.DataFrame):
    metadata_df = metadata_df.copy()
    for col in ['title', 'description', 'category', 'price', 'ranking']:
        if col not in metadata_df.columns:
            metadata_df[col] = ''
        metadata_df[col] = metadata_df[col].apply(safe_text)

    metadata_df['doc_text'] = metadata_df.apply(
        lambda row: ' '.join([
            row['title'],
            row['description'],
            row['category'],
            f"price {row['price']}",
            f"ranking {row['ranking']}",
        ]).strip(),
        axis=1,
    )
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=100000)
    doc_matrix = vectorizer.fit_transform(metadata_df['doc_text'])
    id_to_index = {item_id: idx for idx, item_id in enumerate(metadata_df['id'].astype(str).tolist())}
    id_to_doc = dict(zip(metadata_df['id'].astype(str).tolist(), metadata_df['doc_text'].tolist()))
    return metadata_df, vectorizer, doc_matrix, id_to_index, id_to_doc


def history_profile(row: pd.Series, id_to_doc: dict[str, str]) -> str:
    his_raw = safe_text(row.get('remaining_interaction_string', ''))
    if not his_raw:
        return ''
    item_ids = [x.strip() for x in his_raw.split('|') if x.strip()]
    docs = [id_to_doc[item_id] for item_id in item_ids if item_id in id_to_doc]
    return ' '.join(docs)


def ndcg(rank: int, topk: int) -> float:
    if rank <= 0 or rank > topk:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--metadata_file', type=str, required=True)
    parser.add_argument('--topks', type=int, nargs='+', default=[10, 20, 40])
    parser.add_argument('--output_file', type=str, default='run/amazon_beauty/retrieval_top40.jsonl')
    parser.add_argument('--metrics_file', type=str, default='run/amazon_beauty/retrieval_metrics.json')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--weight_query', type=float, default=0.60)
    parser.add_argument('--weight_history', type=float, default=0.25)
    args = parser.parse_args()

    query_df = pd.read_csv(args.query_file)
    metadata_df = pd.read_csv(args.metadata_file)
    query_df['id'] = query_df['id'].astype(str)
    metadata_df['id'] = metadata_df['id'].astype(str)

    metadata_df, vectorizer, doc_matrix, id_to_index, id_to_doc = build_metadata_index(metadata_df)

    topks = sorted(set(args.topks))
    max_topk = max(topks)
    n = len(query_df) if args.max_samples <= 0 else min(args.max_samples, len(query_df))

    hr_sum = {k: 0.0 for k in topks}
    ndcg_sum = {k: 0.0 for k in topks}
    ranks = []

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as out:
        for i in range(n):
            row = query_df.iloc[i]
            query_text = safe_text(row.get('new_query', '')) or safe_text(row.get('query', ''))
            his_text = history_profile(row, id_to_doc)

            q_vec = vectorizer.transform([query_text])
            score = args.weight_query * linear_kernel(q_vec, doc_matrix).ravel()

            if his_text:
                h_vec = vectorizer.transform([his_text])
                score += args.weight_history * linear_kernel(h_vec, doc_matrix).ravel()

            history_ids = {x.strip() for x in safe_text(row.get('remaining_interaction_string', '')).split('|') if x.strip()}
            for item_id in history_ids:
                idx = id_to_index.get(item_id)
                if idx is not None:
                    score[idx] = -1e12

            topn = min(max_topk, len(score))
            idx = np.argpartition(-score, topn - 1)[:topn]
            idx = idx[np.argsort(-score[idx])]
            ranked_ids = metadata_df.iloc[idx]['id'].tolist()

            target_id = row['id']
            rank = ranked_ids.index(target_id) + 1 if target_id in ranked_ids else 0
            ranks.append(rank)

            for k in topks:
                hr_sum[k] += 1.0 if 0 < rank <= k else 0.0
                ndcg_sum[k] += ndcg(rank, k)

            out.write(json.dumps({
                'row_index': int(i),
                'user_id': safe_text(row.get('user_id', '')),
                'query': query_text,
                'target_id': target_id,
                'target_rank': rank,
                'recommendation_list': ranked_ids,
            }, ensure_ascii=False) + '\n')

    metrics = {
        'samples': n,
        'HR': {f'HR@{k}': hr_sum[k] / max(n, 1) for k in topks},
        'NDCG': {f'NDCG@{k}': ndcg_sum[k] / max(n, 1) for k in topks},
        'mean_rank_or_0': float(np.mean(ranks)) if ranks else 0.0,
        'found_ratio_top_maxk': float(np.mean([1 if r > 0 else 0 for r in ranks])) if ranks else 0.0,
    }

    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
    with open(args.metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
