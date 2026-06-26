"""
Pkg 3: Natural language translation [last updated 2026-05-14]

The goal of this document is to test out using LLMs for simplifying the legalese texts.
Motivation:
    Using the OpenAI API allows access to frontier models without local hardware.
    Since presumably original text will always be provided also hallucinations should be less of an issue.
    Hallucinations also get reduced by the small context window, so no need for relying on background information.
    Enables easy tweaking of the "translation" through tweaking eg. the model or the system prompt.

Requirements:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in a .env file at the repo root (see .env.example) or as a shell env var.

Input:
    A UHRI export file at DATA_PATH, loaded via src.Dataloader.load_records.

Output:
    Simplified text written to OUTPUT_PATH as CSV.

Open issues:
    The code needs to be modularized and organized.
    Work more on pre-processing (adding context)
    Spending some time on choosing the best model might be wise.
"""

from dotenv import load_dotenv
from src.uhri_record import UHRIRecord
from src.Dataloader import load_records
import csv
import random
from openai import OpenAI
from src.evaluation import readability, score_simplification, add_fk_normalized_score
from src.prompts import PROMPTS, DEFAULT_PROMPT, build_prompt

load_dotenv()  # loads OPENAI_API_KEY from .env if present

DATA_PATH = "../data/email_recs.xlsx"
MODEL = "gpt-5.5" # Find others at: https://platform.openai.com/docs/models
OUTPUT_PATH = "../data/cleaned.csv"  # tidy/long format, one row per (text, prompt)
FULL_TESTS = True

# Pre-processing knobs (previously done by hand in scripts/analyse_simplification.ipynb)
SAMPLE_SIZE = None   # None = evaluate on all records; set an int to draw a random subset
RANDOM_SEED = 42     # fixes which records get sampled, so runs are reproducible

openai_client = OpenAI()

# ================ End of test-prompt setup ================================
def simplify(record: UHRIRecord, model: str = MODEL, template: str = DEFAULT_PROMPT) -> str:
    """
    Simplify a single UHRI record.

    Builds the prompt with build_prompt() and sends it to the given model via
    the OpenAI responses API. No post-processing.

    Args:
        record: The UHRIRecord to simplify.
        model: OpenAI model ID (e.g. "gpt-5.5", "gpt-4o")
        template: Prompt template passed through to build_prompt().
                  Defaults to the module-level DEFAULT_PROMPT.

    Returns:
        The model's generated simplified text.
    """
    prompt = build_prompt(record, template)
    response = openai_client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "medium"},  # "minimal" | "low" | "medium" | "high"
    )
    return response.output_text


def select_records(records, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Returns all records when sample_size is None or larger than the dataset,
    otherwise a random subset of the given size.
    """
    if sample_size is None or sample_size >= len(records):
        return list(records)
    return random.Random(seed).sample(records, sample_size)


def run_experiment(records, prompts=PROMPTS, model=MODEL):
    """
    Run every prompt on every record and return long rows.

    Generates one row per (record, prompt) pair, with each metric as its own column (pivoted columns)
    """
    rows = []
    for record in records:
        print(f"\n{'=' * 60}")
        print(f"{record.recommending_body} to {record.country} [{record.type}]")
        orig_read = readability(record.text)

        for name, template in prompts.items():
            conversion = simplify(record, model=model, template=template)
            print(f"  Finished simplifying with {name}")

            metrics = score_simplification(record, conversion, orig_read)
            print(f"  -> eval: {metrics}")

            row = {
                "original_text": record.text,
                "prompt": name,
                "transcribed_text": conversion,
            }
            for k, v in metrics.items():
                row[k.replace("__", "_")] = v
            rows.append(row)

    return rows


if __name__ == "__main__":
    records = load_records(DATA_PATH)

    if not FULL_TESTS:
        record = records[0]
        print("\n Original:")
        print(record.text)
        print("\n Prompt template:")
        print(DEFAULT_PROMPT)
        print("\n Conversion:")
        print(simplify(record))

    else:
        sample = select_records(records)
        print(f"Evaluating {len(sample)} of {len(records)} records "
              f"across {len(PROMPTS)} prompts.")

        rows = run_experiment(sample)
        rows = add_fk_normalized_score(rows)

        # one row per (text, prompt), each metric its own column.
        col_names = list(rows[0].keys())
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=col_names)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Results saved to {OUTPUT_PATH}")

