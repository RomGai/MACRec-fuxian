import os
import json
import math
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from macrec.tasks.base import Task


class AmazonMetaRetrievalTask(Task):
    """Evaluate retrieval ranking on amazon query_data1.csv + metadata.csv."""

    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--query_file', type=str, required=True, help='Path to query_data1.csv')
        parser.add_argument('--metadata_file', type=str, required=True, help='Path to metadata.csv')
        parser.add_argument('--topks', type=int, nargs='+', default=[10, 20, 40], help='Top-k for HR/NDCG')
        parser.add_argument('--output_file', type=str, default=None, help='Optional output path for ranked results jsonl')
        parser.add_argument('--max_samples', type=int, default=0, help='Evaluate first N samples only (0 means all)')
        parser.add_argument('--weight_query', type=float, default=0.60, help='Weight of query text in retrieval')
        parser.add_argument('--weight_history', type=float, default=0.25, help='Weight of interaction-history profile in retrieval')
        return parser

    @staticmethod
    def _safe_text(value: object) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ''
        text = str(value).strip()
        return '' if text.lower() == 'nan' else text

    def _build_metadata_index(self, metadata_df: pd.DataFrame):
        metadata_df = metadata_df.copy()
        for col in ['title', 'description', 'category', 'price', 'ranking']:
            if col not in metadata_df.columns:
                metadata_df[col] = ''
            metadata_df[col] = metadata_df[col].apply(self._safe_text)

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
        return metadata_df, vectorizer, doc_matrix, id_to_index

    def _history_profile(self, row: pd.Series, id_to_doc: dict[str, str]) -> str:
        his_raw = self._safe_text(row.get('remaining_interaction_string', ''))
        if not his_raw:
            return ''
        item_ids = [x.strip() for x in his_raw.split('|') if x.strip()]
        docs = [id_to_doc[item_id] for item_id in item_ids if item_id in id_to_doc]
        return ' '.join(docs)

    @staticmethod
    def _ndcg(rank: int, topk: int) -> float:
        if rank <= 0 or rank > topk:
            return 0.0
        return 1.0 / math.log2(rank + 1)

    def run(
        self,
        query_file: str,
        metadata_file: str,
        topks: list[int],
        output_file: str | None,
        max_samples: int,
        weight_query: float,
        weight_history: float,
    ):
        query_df = pd.read_csv(query_file)
        metadata_df = pd.read_csv(metadata_file)

        if 'id' not in query_df.columns or 'id' not in metadata_df.columns:
            raise ValueError('Both query and metadata csv files must have an `id` column.')

        metadata_df['id'] = metadata_df['id'].astype(str)
        query_df['id'] = query_df['id'].astype(str)

        metadata_df, vectorizer, doc_matrix, id_to_index = self._build_metadata_index(metadata_df)
        id_to_doc = dict(zip(metadata_df['id'].tolist(), metadata_df['doc_text'].tolist()))

        topks = sorted(set(topks))
        max_topk = max(topks)
        n = len(query_df) if max_samples <= 0 else min(max_samples, len(query_df))

        hr_sum = {k: 0.0 for k in topks}
        ndcg_sum = {k: 0.0 for k in topks}
        ranks: list[int] = []
        per_case = []

        for i in range(n):
            row = query_df.iloc[i]
            query_text = self._safe_text(row.get('new_query', '')) or self._safe_text(row.get('query', ''))
            his_text = self._history_profile(row, id_to_doc)

            q_vec = vectorizer.transform([query_text])
            h_vec = vectorizer.transform([his_text]) if his_text else None

            score = weight_query * linear_kernel(q_vec, doc_matrix).ravel()
            if h_vec is not None:
                score += weight_history * linear_kernel(h_vec, doc_matrix).ravel()

            history_ids = {
                x.strip() for x in self._safe_text(row.get('remaining_interaction_string', '')).split('|') if x.strip()
            }
            for item_id in history_ids:
                idx = id_to_index.get(item_id)
                if idx is not None:
                    score[idx] = -1e12

            ranked_idx = np.argpartition(-score, min(max_topk, len(score) - 1))[:max_topk]
            ranked_idx = ranked_idx[np.argsort(-score[ranked_idx])]
            ranked_ids = metadata_df.iloc[ranked_idx]['id'].tolist()

            target_id = row['id']
            rank = ranked_ids.index(target_id) + 1 if target_id in ranked_ids else 0
            ranks.append(rank)

            for k in topks:
                hit = 1.0 if (rank > 0 and rank <= k) else 0.0
                hr_sum[k] += hit
                ndcg_sum[k] += self._ndcg(rank, k)

            per_case.append({
                'row_index': int(i),
                'user_id': self._safe_text(row.get('user_id', '')),
                'target_id': target_id,
                'target_rank': rank,
                'topk_list': ranked_ids,
            })

        metrics = {
            'samples': n,
            'topks': topks,
            'HR': {f'HR@{k}': hr_sum[k] / max(n, 1) for k in topks},
            'NDCG': {f'NDCG@{k}': ndcg_sum[k] / max(n, 1) for k in topks},
            'mean_target_rank_in_topk_or_0': float(np.mean(ranks)) if ranks else 0.0,
            'target_found_ratio': float(np.mean([1 if r > 0 else 0 for r in ranks])) if ranks else 0.0,
        }

        logger.success(json.dumps(metrics, ensure_ascii=False, indent=2))

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
            with open(output_file, 'w', encoding='utf-8') as f:
                for row in per_case:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            logger.info(f'Saved ranked lists to: {output_file}')

        return metrics


if __name__ == '__main__':
    AmazonMetaRetrievalTask().launch()
