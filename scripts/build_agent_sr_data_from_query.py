"""Build sr-style evaluation csv from query_data1.csv + metadata.csv for native MACRec agent policy.

This script intentionally does NOT use the `preferences` column. Instead, it writes
`user_profile` as the current query requirement so the agent infers preference by itself
from query + historical interactions.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def safe_text(v: object) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


def item_attr(row: pd.Series) -> str:
    return f"Title: {safe_text(row.get('title',''))}, Category: {safe_text(row.get('category',''))}, Description: {safe_text(row.get('description',''))}, Price: {safe_text(row.get('price',''))}, Ranking: {safe_text(row.get('ranking',''))}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--metadata_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--n_candidate', type=int, default=40)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--weight_query', type=float, default=0.75)
    parser.add_argument('--weight_history', type=float, default=0.25)
    args = parser.parse_args()

    qdf = pd.read_csv(args.query_file)
    mdf = pd.read_csv(args.metadata_file)
    qdf['id'] = qdf['id'].astype(str)
    mdf['id'] = mdf['id'].astype(str)

    for col in ['title', 'description', 'category', 'price', 'ranking']:
        if col not in mdf.columns:
            mdf[col] = ''

    mdf['item_attributes'] = mdf.apply(item_attr, axis=1)
    mdf['doc'] = mdf['item_attributes']

    id_to_attr = dict(zip(mdf['id'].tolist(), mdf['item_attributes'].tolist()))
    id_to_idx = {iid: i for i, iid in enumerate(mdf['id'].tolist())}

    vec = TfidfVectorizer(lowercase=True, stop_words='english', max_features=100000)
    X = vec.fit_transform(mdf['doc'])

    n = len(qdf) if args.max_samples <= 0 else min(len(qdf), args.max_samples)
    out_rows = []

    for i in range(n):
        row = qdf.iloc[i]
        target_id = safe_text(row['id'])
        query = safe_text(row.get('new_query', '')) or safe_text(row.get('query', ''))

        his_ids = [x.strip() for x in safe_text(row.get('remaining_interaction_string', '')).split('|') if x.strip()]
        his_attrs = [id_to_attr[x] for x in his_ids if x in id_to_attr]
        history_text = '\n'.join(his_attrs) if his_attrs else 'None'

        qv = vec.transform([query])
        score = args.weight_query * linear_kernel(qv, X).ravel()

        if his_attrs:
            hv = vec.transform([' '.join(his_attrs)])
            score += args.weight_history * linear_kernel(hv, X).ravel()

        for hid in his_ids:
            j = id_to_idx.get(hid)
            if j is not None:
                score[j] = -1e12

        topn = min(args.n_candidate, len(score))
        cand_idx = np.argpartition(-score, topn - 1)[:topn]
        cand_idx = cand_idx[np.argsort(-score[cand_idx])]
        cand_ids = mdf.iloc[cand_idx]['id'].tolist()

        # ensure target exists in candidate list for sr evaluation label parsing
        if target_id in id_to_idx and target_id not in cand_ids:
            cand_ids[-1] = target_id

        cand_text = '\n'.join([f"{cid}: {id_to_attr[cid]}" for cid in cand_ids])

        # Put current query into user_profile so native prompts can consume the requirement.
        user_profile = f"Current requirement query: {query}"

        out_rows.append({
            'user_id': safe_text(row.get('user_id', f'u{i}')),
            'user_profile': user_profile,
            'history': history_text,
            'candidate_item_attributes': cand_text,
            'item_id': target_id,
            'rating': 5,
        })

    out_df = pd.DataFrame(out_rows)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_file, index=False)
    print(f'saved {len(out_df)} rows to {args.output_file}')


if __name__ == '__main__':
    main()
