"""
Pkg 3: Evaluation of simplified UHRI texts [last updated 2026-05-27]
 
Evaluates a simplified text against its original using LLM-as-judge (Anthropic Claude), Flesch-Kincaid
and semantic readability
 
Usage:
    from src.evaluation import readability, score_simplification
    orig_read = readability(record.text)
    metrics   = score_simplification(record.text, simplified, orig_read)
 
Requirements:
    anthropic, textstat, sentence-transformers
 
Open Issues:
    - maybe remove semantic readability metric since not that informative and we have to load
    - smaller models instead one big model
 
Resumption:
    EVAL_DIMENSIONS must stay in sync with the JSON keys in EVAL_PROMPT.
    If you add a dimension, update both.
"""
 
import json
from dotenv import load_dotenv 
import anthropic
import textstat
from sentence_transformers import SentenceTransformer, util
import re

load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present
 
EVAL_MODEL = "claude-sonnet-4-5"
eval_client = anthropic.Anthropic()
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # for semantic similarity measurement
 
EVAL_PROMPT = """\
You are evaluating whether a simplified version of a UN human rights text is accurate and accessible.
 
Carefully compare the original and simplified texts. Be strict: if anything important is missing, changed, or added incorrectly, mark as false.
 
Original data:
{original}
 
Simplified text:
{simplified}
 
Evaluate on these dimensions:
{explanations}
 
Respond with ONLY a JSON object in exactly this format, with no preamble, no markdown fences, no explanation outside the JSON:
 
{{
    {eval_dimensions}
}}
 
The "score" field must be a JSON boolean (true or false), never a string. The "reason" field is one short sentence explaining the score.\
"""
 
EVAL_DIMENSIONS = {
    "content_preserved": "Does the simplified text preserve ALL important obligations, actions, and decisions from the original?",
    "actors_preserved": "Are ALL responsible actors and affected parties correctly identified (no actors removed, generalized, or changed)?",
    "no_meaning_change":  "Does the simplified text avoid changing the meaning, strength, or conditions of the original (e.g. 'must' vs 'should', removing exceptions)?",
    "no_hallucination": "Is the simplified text free of any new information, claims, or interpretations not present in the original?",
    "accessible_and_simple": "Is the simplified text clearly easier to read and understandable to a non-expert, while still preserving the important meaning?",
}
 
 
def evaluate(
    original,
    simplified: str,
    model: str = EVAL_MODEL,
) -> dict:
    """
    Evaluate a simplified text against its original on 5 dimensions.
 
    Args:
        original: The original source text, and associated meta data.
        simplified: The simplified output to evaluate.
        model: Anthropic model name for the judge.
 
    Returns:
        Dict keyed by dimension name. 
        Score conventions: 1 = yes/pass, 0 = no/fail, -1 = parse error.
    """

    prompt = build_eval_prompt(original, simplified)
 
    response = eval_client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    
    raw = response.content[0].text.strip()
 
    # in case llm dumb and adds stuff around the JSON brackets
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]
 
    # Parse JSON
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  Failed to parse judge response as JSON. Raw output:\n{raw}")
        parsed = {}
 
    # ensure every expected dimension is present, give score 1 or 0, -1 if error
    result = {}
    for dim in EVAL_DIMENSIONS:
        entry = parsed.get(dim)
        if not isinstance(entry, dict) or "score" not in entry:
            result[dim] = {"score": -1, "reason": "missing"}
            continue
 
        score_val = entry["score"]
        if score_val is True:
            score_int = 1
        elif score_val is False:
            score_int = 0
        else:
            score_int = -1
 
        result[dim] = {"score": score_int, "reason": entry.get("reason", "reason missing")}
 
    return result

def build_eval_prompt(
    original: str,
    simplified: str,
):
    """
    Build the prompt for evaluating the query.
    """
    numerator = 1
    formated_dim_explanations = ""
    dim_explanations_template = "{num}. {dim}: {desc} \n"

    
    formated_eval_dimensions = ""
    for dim, description in EVAL_DIMENSIONS.items():
        # The explanation of each item.
        explanation = dim_explanations_template.format(
            num = numerator,
            dim = dim,
            desc = description
        )
        numerator += 1
        formated_dim_explanations += explanation

        # The json format template for the evaluation.
        formated_eval = '"%s": : {"reason": "<one short sentence>", "score": true}, \n' % dim
        formated_eval_dimensions += formated_eval


    prompt = EVAL_PROMPT.format(original=original, 
                                simplified=simplified, 
                                explanations = formated_dim_explanations,
                                eval_dimensions = formated_eval_dimensions
                               )

    return prompt

 

def readability(text: str) -> dict[str, float]:
    """Flesch-Kincaid grade and ease, after normalizing for legalese formatting."""
    text = text.replace(";", ".")
    text = re.sub(r"\([a-z]\)\s*", "", text) 
    text = re.sub(r"^\d+\.\s*", "", text)  
    return {
        "fk_grade": textstat.flesch_kincaid_grade(text),
        "fk_ease": textstat.flesch_reading_ease(text),
    }
 
def semantic_similarity(original: str, simplified: str) -> float:
    """Cosine similarity between sentence-transformer embeddings"""
    embeddings = embed_model.encode([original, simplified])
    return float(util.cos_sim(embeddings[0], embeddings[1]))
 
 
def score_simplification(
    original: str,
    simplified: str,
    orig_readability: dict[str, float],
) -> dict:
    """
    Compute all evaluation metrics for one simplified output.
 
    Bundles the judge scores, readability deltas, and semantic similarity into
    a single flat dict so the caller doesn't have to assemble them by hand.
 
    Args:
        original: The original source text.
        simplified: The simplified output to score.
        orig_readability: Readability dict for the original text 
 
    Returns:
        Flat dict of metrics
    """
    simp_read = readability(simplified)
    scores = evaluate(original, simplified)
 
    out = {
        "fk_grade": simp_read["fk_grade"],
        # "fk_ease": simp_read["fk_ease"],
        "fk_grade_delta": orig_readability["fk_grade"] - simp_read["fk_grade"],
        # "semantic_sim": semantic_similarity(original, simplified),
    }
    for dim, entry in scores.items():
        out[f"{dim}__score"] = entry["score"]
        out[f"{dim}__reason"] = entry["reason"]
 
    return out
 
