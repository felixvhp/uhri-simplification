#!/usr/bin/env python3
import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
SEMANTIC_TERM_MATCH_PATH = ROOT_DIR / "code" / "src" / "search.py"

_semantic_spec = importlib.util.spec_from_file_location(
    "semantic_term_match", SEMANTIC_TERM_MATCH_PATH
)
if _semantic_spec is None or _semantic_spec.loader is None:
    raise ImportError(f"Cannot load semantic matcher module from {SEMANTIC_TERM_MATCH_PATH}")
_semantic_term_match = importlib.util.module_from_spec(_semantic_spec)
_semantic_spec.loader.exec_module(_semantic_term_match)

build_candidates_from_topics = _semantic_term_match.build_candidates_from_topics
detect_query_locations = _semantic_term_match.detect_query_locations
load_geo_terms = _semantic_term_match.load_geo_terms
rank_candidates = _semantic_term_match.rank_candidates


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())




BLOCKED_META_TOPICS = {
    "recommendation",
    "recommendations",
    "recommend",
    "observation",
    "observations",
    "concern",
    "concerns",
}

def load_topics_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            idx = obj.get("index")
            if isinstance(idx, int):
                rows[idx] = obj
    return rows


def print_cli_summary(
    detected_locations: list[str],
    ranked_topics,
    results: list[dict[str, Any]],
) -> None:
    print("Detected locations:")
    for i, loc in enumerate(detected_locations, start=1):
        print(f"{i:>2}. {loc}")

    print("\nTop relevant topics:")
    for i, (cand, score, _detail) in enumerate(ranked_topics, start=1):
        print(f"{i:>2}. {cand.text} (score={score:.4f})")

    print("\nTop related items (1-20):")
    for i, row in enumerate(results[:20], start=1):
        text = row.get("Text", "")
        content = " ".join(text.split()) if isinstance(text, str) else ""
        print(f"{i:>2}. {content}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast exact filter: keep items whose Countries contain an exact location "
            "and whose top_10_topics contain at least one of semantic_term_match top-k topics."
        )
    )
    parser.add_argument("query", help="Query sentence for semantic_term_match top topics.")
    parser.add_argument("--export", type=Path, default=ROOT_DIR / "data" / "export_full_en.json")
    parser.add_argument(
        "--topics-jsonl",
        type=Path,
        default=ROOT_DIR / "data" / "gpt_processed.json",
    )
    parser.add_argument(
        "--topics-tsv",
        type=Path,
        default=ROOT_DIR / "data" / "gpt_topics.tsv",
        help="TSV with rank,term,count used by semantic_term_match.",
    )
    parser.add_argument("--top-k-topics", type=int, default=25)
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_results",
        help="Print ranked results to stdout.",
    )
    parser.add_argument("--text-max-chars", type=int, default=1200)
    parser.add_argument(
        "--top-k-results",
        type=int,
        default=0,
        help="If > 0, keep only the top K ranked results in output/printing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ranked_items_exact_location_topk.json"),
    )
    args = parser.parse_args()

    candidates = build_candidates_from_topics(args.topics_tsv)
    if not candidates:
        raise ValueError("No topic candidates loaded from --topics-tsv.")

    geo_terms = load_geo_terms(args.export)
    detected_locations_raw = detect_query_locations(args.query, geo_terms)
    detected_locations = [norm(x) for x in detected_locations_raw]

    ranked_topics = rank_candidates(
        query=args.query,
        candidates=candidates,
        top_k=args.top_k_topics,
        geo_terms=geo_terms,
        block_geo_candidates=True,
    )
    topic_scores = {
        norm(c.text): float(score)
        for c, score, _ in ranked_topics
        if norm(c.text) not in BLOCKED_META_TOPICS
    }
    topic_set = set(topic_scores.keys())

    row_by_index = load_topics_rows(args.topics_jsonl)
    location_set = set(detected_locations)

    with args.export.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected --export to be a top-level JSON list.")

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if location_set:
            countries = item.get("Countries", [])
            if not isinstance(countries, list):
                continue
            country_norms = [norm(c) for c in countries if isinstance(c, str) and c.strip()]
            if not (set(country_norms) & location_set):
                continue

        topic_row = row_by_index.get(idx)
        if not topic_row:
            continue
        item_topics = topic_row.get("top_10_topics", [])
        if not isinstance(item_topics, list):
            continue
        item_topic_norms = [norm(t) for t in item_topics if isinstance(t, str) and t.strip()]
        matched_topics = [t for t in item_topic_norms if t in topic_set]
        if not matched_topics:
            continue

        score = sum(topic_scores[t] for t in matched_topics)
        text = item.get("Text", "")
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if len(text) > args.text_max_chars:
            text = text[: args.text_max_chars]

        results.append(
            {
                "index": idx,
                "AnnotationId": item.get("AnnotationId"),
                "Symbol": item.get("Symbol"),
                "score": round(score, 6),
                "matched_topics": matched_topics,
                "Countries": item.get("Countries", []),
                "top_10_topics": item_topics,
                "Text": text,
            }
        )

    results.sort(key=lambda x: (x["score"], len(x["matched_topics"])), reverse=True)
    if args.top_k_results > 0:
        results = results[: args.top_k_results]

    output = {
        "query": args.query,
        "detected_locations": detected_locations_raw,
        "top_k_topics": args.top_k_topics,
        "top_topics_used": [t for t, _ in sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)],
        "matched_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Query: {args.query}")
    print(f"Detected locations: {detected_locations_raw}")
    print(f"Top topics used: {len(topic_set)}")
    print(f"Matched items: {len(results)}")
    print(f"Saved: {args.output}")

    if args.print_results:
        print_cli_summary(
            detected_locations=detected_locations_raw,
            ranked_topics=ranked_topics,
            results=results,
        )


if __name__ == "__main__":
    main()
