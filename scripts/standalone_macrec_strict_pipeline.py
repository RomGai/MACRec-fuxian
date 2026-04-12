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


class WikipediaTool:
    """A lightweight Search/Lookup wrapper similar to MACRec Searcher tool usage."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self._enabled = True
        try:
            import wikipedia  # type: ignore

            self.wikipedia = wikipedia
            self.wikipedia.set_lang('en')
        except Exception:
            self.wikipedia = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self.wikipedia is not None

    def search(self, query: str) -> list[str]:
        if not self.enabled:
            return []
        try:
            return self.wikipedia.search(query, results=self.top_k)
        except Exception:
            return []

    def lookup(self, title: str, term: str | None = None) -> str:
        if not self.enabled:
            return ''
        try:
            page = self.wikipedia.page(title, auto_suggest=False)
            text = page.summary or ''
        except Exception:
            return ''
        if term:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            snippets = []
            for sent in re.split(r'(?<=[.!?])\s+', text):
                if pattern.search(sent):
                    snippets.append(sent.strip())
                if len(snippets) >= 2:
                    break
            if snippets:
                return ' '.join(snippets)
        return text[:600]


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
        self._token_pattern = re.compile(r'[a-z0-9]+')
        self.doc_terms = [set(self._token_pattern.findall(t.lower())) for t in df['doc_text'].tolist()]

    def _overlap_score(self, query_terms: set[str], doc_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        return len(query_terms & doc_terms) / len(query_terms)

    def _terms(self, text: str) -> set[str]:
        return set(self._token_pattern.findall(text.lower()))

    def retrieve(self, query: str, history_text: str, blocked_ids: set[str], topn: int) -> list[str]:
        q_terms = self._terms(query)
        h_terms = self._terms(history_text) if history_text else set()
        s = np.array([0.75 * self._overlap_score(q_terms, terms) for terms in self.doc_terms], dtype=float)
        if h_terms:
            s += np.array([0.25 * self._overlap_score(h_terms, terms) for terms in self.doc_terms], dtype=float)
        for iid in blocked_ids:
            j = self.id_to_idx.get(iid)
            if j is not None:
                s[j] = -1e12
        n = min(topn, len(s))
        idx = np.argpartition(-s, n - 1)[:n]
        idx = idx[np.argsort(-s[idx])]
        return self.df.iloc[idx]['id'].tolist()

    def score_candidates(self, query: str, history_text: str, candidates: list[str]) -> list[tuple[str, float]]:
        q_terms = self._terms(query)
        h_terms = self._terms(history_text) if history_text else set()
        pairs = []
        for iid in candidates:
            terms = self.doc_terms[self.id_to_idx[iid]]
            score = 0.75 * self._overlap_score(q_terms, terms)
            if h_terms:
                score += 0.25 * self._overlap_score(h_terms, terms)
            pairs.append((iid, float(score)))
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
    def __init__(
        self,
        corpus: Corpus,
        policy: str = 'heuristic',
        model_name: str = 'Qwen/Qwen3-8B',
        max_step: int = 4,
        enable_thinking: bool = True,
        wiki_top_k: int = 3,
        search_max_turns: int = 3,
    ):
        self.corpus = corpus
        self.policy = policy
        self.max_step = max_step
        self.wiki_tool = WikipediaTool(top_k=wiki_top_k)
        self.search_max_turns = search_max_turns
        self.qwen = None
        if policy == 'qwen':
            self.qwen = QwenPolicy(model_name=model_name, enable_thinking=enable_thinking)

    @staticmethod
    def _parse_search_action(text: str) -> tuple[str, str]:
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            return 'finish', ''
        try:
            obj = json.loads(m.group(0))
            action = str(obj.get('action', 'finish')).strip().lower()
            argument = obj.get('argument', '')
            if isinstance(argument, (dict, list)):
                argument = json.dumps(argument, ensure_ascii=False)
            return action, str(argument).strip()
        except Exception:
            return 'finish', ''

    def _searcher_react(self, requirements: str, max_turns: int = 3) -> tuple[str, list[dict]]:
        """Searcher-style ReAct loop: Search/Lookup/Finish with observations."""
        trace: list[dict] = []
        context_chunks: list[str] = []
        last_titles: list[str] = []

        for turn in range(1, max_turns + 1):
            if self.policy == 'qwen' and self.qwen is not None:
                history_text = '\n'.join(
                    [f"Turn {i + 1} command={h['command']} observation={h['observation'][:300]}" for i, h in enumerate(trace)]
                )
                prompt = (
                    "You are a Searcher agent. Choose one JSON action only.\n"
                    "Valid actions: Search, Lookup, Finish.\n"
                    "Formats:\n"
                    "- {\"action\":\"Search\",\"argument\":\"query\"}\n"
                    "- {\"action\":\"Lookup\",\"argument\":\"title||term\"}\n"
                    "- {\"action\":\"Finish\",\"argument\":\"final notes\"}\n"
                    f"Requirements: {requirements}\n"
                    f"History:\n{history_text}\n"
                )
                raw = self.qwen.ask(prompt)
                action, argument = self._parse_search_action(raw)
            else:
                if turn == 1:
                    action, argument = 'search', requirements
                elif turn == 2 and last_titles:
                    action, argument = 'lookup', f"{last_titles[0]}||{requirements}"
                else:
                    action, argument = 'finish', ''

            if action == 'search':
                titles = self.wiki_tool.search(argument or requirements) if self.wiki_tool.enabled else []
                last_titles = titles
                observation = json.dumps(titles[:5], ensure_ascii=False)
            elif action == 'lookup':
                title = ''
                term = ''
                if '||' in argument:
                    title, term = argument.split('||', 1)
                elif last_titles:
                    title = last_titles[0]
                    term = requirements
                snippet = self.wiki_tool.lookup(title=title.strip(), term=term.strip()) if self.wiki_tool.enabled and title.strip() else ''
                if snippet:
                    context_chunks.append(snippet)
                observation = snippet[:600] if snippet else 'lookup empty'
            elif action == 'finish':
                observation = 'finish search'
                trace.append({'command': f'{action}({argument})', 'observation': observation})
                break
            else:
                observation = f'unknown action {action}'

            trace.append({'command': f'{action}({argument})', 'observation': observation})

        return ' '.join(context_chunks).strip()[:2000], trace

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

    def run_one(self, query: str, history_ids: list[str], preselect_k: int = 500, out_k: int = 40) -> list[str]:
        history_docs = [self.corpus.id_to_doc[x] for x in history_ids if x in self.corpus.id_to_doc]
        history_text = ' '.join(history_docs)
        blocked = set(history_ids)

        state = ManagerState()
        cand_ids: list[str] = []
        wiki_context = ''

        for step in range(1, self.max_step + 1):
            action, arg = self._manager_action(query, history_text, step, state.scratchpad)
            print(f"[Agent] Step {step}/{self.max_step} -> action={action}, argument={arg[:80] if arg else ''}")
            if action.lower() == 'search':
                if self.wiki_tool.enabled:
                    search_query = arg or query
                    wiki_context, search_trace = self._searcher_react(requirements=search_query, max_turns=self.search_max_turns)
                    obs = f"searcher_react turns={len(search_trace)}, context_len={len(wiki_context)}"
                else:
                    obs = 'wiki tool unavailable; skipped external search'
            elif action.lower() == 'analyse':
                if not cand_ids:
                    retrieve_query = (query + ' ' + wiki_context).strip()
                    cand_ids = self.corpus.retrieve(query=retrieve_query, history_text=history_text, blocked_ids=blocked, topn=preselect_k)
                rerank_query = (query + ' ' + wiki_context).strip()
                scored = self.corpus.score_candidates(query=rerank_query, history_text=history_text, candidates=cand_ids)
                cand_ids = [iid for iid, _ in scored]
                obs = 'analysed and reranked candidates'
            elif action.lower() == 'finish':
                state.finished = True
                if not cand_ids:
                    retrieve_query = (query + ' ' + wiki_context).strip()
                    cand_ids = self.corpus.retrieve(query=retrieve_query, history_text=history_text, blocked_ids=blocked, topn=preselect_k)
                state.final_rank = cand_ids[:out_k]
                break
            else:
                obs = f'invalid action {action}, fallback analyse'

            state.scratchpad += f"\nThought {step}: ...\nAction {step}: {action}({arg})\nObservation: {obs}\n"

        if state.final_rank is None:
            if not cand_ids:
                retrieve_query = (query + ' ' + wiki_context).strip()
                cand_ids = self.corpus.retrieve(query=retrieve_query, history_text=history_text, blocked_ids=blocked, topn=preselect_k)
            state.final_rank = cand_ids[:out_k]

        return state.final_rank


def run(args):
    print('[Progress] Stage 1/5: loading query data...')
    qdf = pd.read_csv(args.query_file)
    qdf['id'] = qdf['id'].astype(str)
    print('[Progress] Stage 2/5: building metadata corpus (non-TF-IDF lexical index)...')
    corpus = Corpus(args.metadata_file)
    print('[Progress] Stage 3/5: initializing strict standalone agent...')
    agent = StrictStandaloneAgent(
        corpus=corpus,
        policy=args.policy,
        model_name=args.model_name,
        max_step=args.max_step,
        enable_thinking=args.enable_thinking,
        wiki_top_k=args.wiki_top_k,
        search_max_turns=args.search_max_turns,
    )

    topks = sorted(set(args.topks))
    maxk = max(topks)
    n = len(qdf) if args.max_samples <= 0 else min(len(qdf), args.max_samples)
    print(f'[Progress] Stage 4/5: start inference for {n} users...')

    hr = {k: 0.0 for k in topks}
    ndcg = {k: 0.0 for k in topks}
    rows = []

    for i in range(n):
        row = qdf.iloc[i]
        target = row['id']
        query = safe_text(row.get('new_query', '')) or safe_text(row.get('query', ''))
        history_ids = [x.strip() for x in safe_text(row.get('remaining_interaction_string', '')).split('|') if x.strip()]
        print(f"\n[User {i + 1}/{n}] user_id={safe_text(row.get('user_id', ''))}, target_id={target}")
        print(f"[User {i + 1}/{n}] Running retrieval/ranking...")

        rank_list = agent.run_one(query=query, history_ids=history_ids, preselect_k=args.preselect_k, out_k=maxk)
        rank = rank_list.index(target) + 1 if target in rank_list else 0
        print(f"[User {i + 1}/{n}] target_rank={rank if rank > 0 else 'not_in_topk'}")

        for k in topks:
            hr[k] += 1.0 if 0 < rank <= k else 0.0
            ndcg[k] += ndcg_at(rank, k)

        seen = i + 1
        avg_hr = {f'HR@{k}': hr[k] / seen for k in topks}
        avg_ndcg = {f'NDCG@{k}': ndcg[k] / seen for k in topks}
        print(
            f"[User {seen}/{n}] Processed-average metrics: "
            f"HR={json.dumps(avg_hr, ensure_ascii=False)} | "
            f"NDCG={json.dumps(avg_ndcg, ensure_ascii=False)}"
        )

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

    print('[Progress] Stage 5/5: writing outputs completed.')
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
    p.add_argument('--preselect_k', type=int, default=500)
    p.add_argument('--wiki_top_k', type=int, default=3)
    p.add_argument('--search_max_turns', type=int, default=3)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
