#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
STOPWORDS = set(ENGLISH_STOP_WORDS)
GENERIC_BLOCKLIST = {
    "committee", "state", "party", "recommends", "recommendation", "efforts", "including",
    "concerned", "also", "continue", "information", "implementation", "measures", "national",
    "international", "provided", "report", "periodic report", "concluding observations",
}
WEAK_QUERY_TERMS = {"condition", "conditions", "situation", "issue", "issues", "status", "state"}
QUERY_ANCHOR_EXPANSIONS = {
    "school": {"education", "student", "students", "teacher", "teachers", "curriculum", "curricula", "learning"},
}
DEFAULT_REGION_TERMS = {
    "africa", "asia", "europe", "oceania", "americas", "middle east",
    "asia-pacific", "latin america", "caribbean", "north africa", "sub-saharan africa",
}
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


@dataclass
class Candidate:
    text: str
    count: int
    source: str


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def simple_stem(token: str) -> str:
    for suffix in ("ing", "edly", "ed", "ly", "ies", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def informative_stems(text: str) -> set[str]:
    return {simple_stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) >= 3}


def build_query_anchors(query: str) -> set[str]:
    anchors = {s for s in informative_stems(query) if s not in WEAK_QUERY_TERMS}
    expanded = set(anchors)
    for a in list(anchors):
        for src, targets in QUERY_ANCHOR_EXPANSIONS.items():
            if a == simple_stem(src):
                expanded.update(simple_stem(t) for t in targets)
    return expanded


def load_geo_terms(geo_json_path: Path | None) -> set[str]:
    terms = set(DEFAULT_REGION_TERMS)
    if geo_json_path is None or not geo_json_path.exists():
        return terms
    try:
        with geo_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                countries = item.get("Countries", [])
                regions = item.get("Regions", [])
                if isinstance(countries, list):
                    for c in countries:
                        if isinstance(c, str) and c.strip():
                            terms.add(c.strip().lower())
                if isinstance(regions, list):
                    for r in regions:
                        if isinstance(r, str) and r.strip():
                            terms.add(r.strip().lower())
    except Exception:
        return terms
    return terms


def geo_stems_set(geo_terms: set[str]) -> set[str]:
    out: set[str] = set()
    for t in geo_terms:
        for tok in tokenize(t):
            out.add(simple_stem(tok))
    return out


def detect_query_locations(query: str, geo_terms: set[str]) -> list[str]:
    q = query.lower()
    found: list[str] = []
    for term in sorted(geo_terms, key=len, reverse=True):
        t = term.strip().lower()
        if not t:
            continue
        pattern = r"(?<![a-z])" + re.escape(t) + r"(?![a-z])"
        if re.search(pattern, q):
            found.append(term)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in found:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(t)
    return deduped


def read_tsv(path: Path, text_col: str, source: str) -> list[Candidate]:
    out: list[Candidate] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            txt = (row.get(text_col) or "").strip()
            if not txt:
                continue
            try:
                cnt = int((row.get("count") or "0").strip())
            except ValueError:
                cnt = 0
            out.append(Candidate(text=txt, count=cnt, source=source))
    return out


def weighted_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if math.isclose(lo, hi):
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def rank_candidates(
    query: str,
    candidates: list[Candidate],
    top_k: int,
    geo_terms: set[str],
    block_geo_candidates: bool,
) -> list[tuple[Candidate, float, dict[str, float]]]:
    texts = [c.text for c in candidates]
    docs = texts + [query]
    q_idx = len(docs) - 1

    word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b")
    Xw = word_vec.fit_transform(docs)
    word_scores = cosine_similarity(Xw[:-1], Xw[q_idx]).ravel()

    char_vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5))
    Xc = char_vec.fit_transform(docs)
    char_scores = cosine_similarity(Xc[:-1], Xc[q_idx]).ravel()

    n_docs, n_features = Xw.shape
    max_k = min(64, n_docs - 1, n_features - 1)
    if max_k >= 2:
        svd = TruncatedSVD(n_components=max_k, random_state=42)
        Xl = svd.fit_transform(Xw)
        lsi_scores = cosine_similarity(Xl[:-1], Xl[q_idx].reshape(1, -1)).ravel()
    else:
        lsi_scores = np.zeros(len(candidates), dtype=float)

    geo_stems = geo_stems_set(geo_terms)
    q_stems = {s for s in informative_stems(query) if s not in geo_stems}
    q_anchors = build_query_anchors(query)
    q_anchors = {s for s in q_anchors if s not in geo_stems}
    c_stems = [informative_stems(c.text) for c in candidates]
    stem_scores = np.array([weighted_jaccard(q_stems, s) for s in c_stems], dtype=float)

    counts = np.array([c.count for c in candidates], dtype=float)
    freq_scores = normalize_scores(np.log1p(counts))

    word_n = normalize_scores(word_scores)
    char_n = normalize_scores(char_scores)
    lsi_n = normalize_scores(lsi_scores)
    stem_n = normalize_scores(stem_scores)

    final = (
        0.45 * word_n
        + 0.24 * lsi_n
        + 0.21 * stem_n
        + 0.07 * char_n
        + 0.03 * freq_scores
    )

    top_raw = float(np.max(final)) if len(final) else 0.0
    min_score = max(0.18, top_raw * 0.30)

    anchor_exists = False
    if q_anchors:
        for s in c_stems:
            if s & q_anchors:
                anchor_exists = True
                break

    keep_mask = np.zeros(len(candidates), dtype=bool)
    for i, cand in enumerate(candidates):
        cand_lower = cand.text.lower().strip()
        if cand_lower in GENERIC_BLOCKLIST:
            continue
        if block_geo_candidates:
            cand_tokens = [simple_stem(t) for t in tokenize(cand_lower)]
            if cand_lower in geo_terms or (cand_tokens and all(t in geo_stems for t in cand_tokens)):
                continue

        overlap = len(c_stems[i] & q_stems)
        anchor_overlap = len(c_stems[i] & q_anchors) if q_anchors else 0

        if anchor_overlap > 0 and final[i] >= max(0.10, min_score * 0.45):
            keep_mask[i] = True
            continue

        if not anchor_exists and overlap > 0 and final[i] >= min_score:
            keep_mask[i] = True
            continue

        # Only allow non-overlap semantic-only matches when anchors are absent.
        if (not anchor_exists) and final[i] >= max(0.58, top_raw * 0.75):
            keep_mask[i] = True

    # Relax progressively until we have enough useful candidates.
    if np.count_nonzero(keep_mask) < min(8, top_k):
        if anchor_exists:
            # When anchors exist (e.g., "school"), do not leak non-anchor items.
            relaxed = np.array([len(c_stems[i] & q_anchors) > 0 for i in range(len(candidates))], dtype=bool)
        else:
            relaxed = (final >= max(0.12, min_score * 0.45))
            if q_anchors:
                for i in range(len(candidates)):
                    if len(c_stems[i] & q_anchors) > 0:
                        relaxed[i] = True
        for i, cand in enumerate(candidates):
            if cand.text.lower().strip() in GENERIC_BLOCKLIST:
                relaxed[i] = False
        keep_mask = relaxed

    if not np.any(keep_mask):
        keep_mask = final >= min_score

    order = np.argsort(-final)
    results: list[tuple[Candidate, float, dict[str, float]]] = []
    for idx in order:
        i = int(idx)
        if not keep_mask[i]:
            continue
        detail = {
            "word_tfidf": float(word_n[i]),
            "lsi": float(lsi_n[i]),
            "stem": float(stem_n[i]),
            "char": float(char_n[i]),
            "freq": float(freq_scores[i]),
        }
        results.append((candidates[i], float(final[i]), detail))
        if len(results) >= top_k:
            break
    return results


def build_candidates(words_path: Path, phrases_path: Path) -> list[Candidate]:
    items = read_tsv(words_path, "word", "word") + read_tsv(phrases_path, "phrase", "phrase")
    seen: set[str] = set()
    unique: list[Candidate] = []
    for c in items:
        k = c.text.lower()
        if k in seen:
            continue
        seen.add(k)
        unique.append(c)
    return unique


def build_candidates_from_topics(topics_path: Path) -> list[Candidate]:
    items = read_tsv(topics_path, "term", "topic")
    seen: set[str] = set()
    unique: list[Candidate] = []
    for c in items:
        k = c.text.lower()
        if k in seen:
            continue
        seen.add(k)
        unique.append(c)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Find semantically similar terms from local word/phrase lists without an LLM.")
    parser.add_argument("query", help="Natural language sentence/query")
    parser.add_argument("--words", type=Path, default=DATA_DIR / "short_words_gpt.tsv")
    parser.add_argument("--phrases", type=Path, default=DATA_DIR / "short_phrases_gpt.tsv")
    parser.add_argument(
        "--topics",
        type=Path,
        default=DATA_DIR / "gpt_topics.tsv",
        help="Optional TSV with columns rank,term,count (for example top_1000_terms_from_gpt_oss_topics.tsv). If set, --words/--phrases are ignored.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--geo-json", type=Path, default=DATA_DIR / "export_full_en.json")
    parser.add_argument("--allow-geo-candidates", action="store_true")
    parser.add_argument("--show-components", action="store_true", help="Print component scores for debugging")
    args = parser.parse_args()

    if args.topics is not None:
        candidates = build_candidates_from_topics(args.topics)
    else:
        candidates = build_candidates(args.words, args.phrases)
    if not candidates:
        raise ValueError("No candidates loaded. Check --topics or --words/--phrases files.")

    geo_terms = load_geo_terms(args.geo_json)
    detected_locations = detect_query_locations(args.query, geo_terms)
    ranked = rank_candidates(
        args.query,
        candidates,
        args.top_k,
        geo_terms=geo_terms,
        block_geo_candidates=not args.allow_geo_candidates,
    )
    print(f"Query: {args.query}")
    print(f"Detected locations: {detected_locations if detected_locations else '[]'}")
    print(f"Candidates: {len(candidates)}")
    print(f"Top {args.top_k} matches:")
    for i, (cand, score, detail) in enumerate(ranked, start=1):
        if args.show_components:
            print(
                f"{i:>2}. {cand.text} | score={score:.4f} | source={cand.source} | "
                f"w={detail['word_tfidf']:.3f}, lsi={detail['lsi']:.3f}, stem={detail['stem']:.3f}, char={detail['char']:.3f}, f={detail['freq']:.3f}"
            )
        else:
            print(f"{i:>2}. {cand.text} | score={score:.4f} | source={cand.source}")


if __name__ == "__main__":
    main()
