"""
Pkg 3: Natural language translation [last updated 2026-05-14]

The goal of this document is to test out using LLMs for simplifying the legalese texts.
Motivation:
    Using the Claude API allows access to frontier models without local hardware.
    Since presumably original text will always be provided also hallucinations should be less of an issue.
    Hallucinations also get reduced by the small context window, so no need for relying on background information.
    Enables easy tweaking of the "translation" through tweaking eg. the model or the system prompt.

Requirements:
    pip install anthropic python-dotenv
    Set ANTHROPIC_API_KEY in a .env file at the repo root (see .env.example) or as a shell env var.

Input:
    A UHRI export file at DATA_PATH, loaded via src.Dataloader.load_records.

Output:
    Simplified text written to OUTPUT_PATH as CSV.

Open issues:
    The code needs to be modularized and organized.
    Work more on pre-processing (adding context)
    Spending some time on choosing the best model might be wise.
"""

import anthropic
from dotenv import load_dotenv
from src.uhri_record import UHRIRecord
from src.Dataloader import load_records
import csv
import re
from openai import OpenAI
from src.evaluation import readability, score_simplification
from src.prompts import PROMPTS, DEFAULT_PROMPT, build_prompt

load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present

DATA_PATH = "../data/email_recs.xlsx"
MODEL = "gpt-5.5" # Find others at: https://ollama.com/search
OUTPUT_PATH = "../data/simplification_results3.csv"
FULL_TESTS = True 
openai_client = OpenAI() 

# ================ End of test-prompt setup ================================
def simplify(record: UHRIRecord, model: str = MODEL, template: str = DEFAULT_PROMPT) -> str:
    """
    Simplify a single UHRI record.

    Builds the prompt with build_prompt() and sends it to the given model via
    the Anthropic messages API. No post-processing.

    Args:
        record: The UHRIRecord to simplify.
        model: Claude model ID (e.g. "claude-haiku-4-5-20251001", "claude-sonnet-4-6")
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
        results_full = []
        for record in records:
            print(f"\n{'=' * 60}")
            print(f"{record.recommending_body} to {record.country} [{record.type}]")
            # print(f"ORIGINAL: {record.text}")


            results = {"original_text": record.text}
            orig_read = readability(record.text) 

            for name, text in PROMPTS.items():
                conversion = simplify(record, template = text)
                print(f"\nFinished Simplifying with {name}")
                # print(conversion)
                results[name] = conversion

                metrics = score_simplification(record, conversion, orig_read)
                for k, v in metrics.items():
                    results[f"{name}__{k}"] = v
                
                print(f"  -> eval: {metrics}")

            results_full.append(results)

        # Write to CSV
        col_names = results_full[0].keys()
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=col_names)
            writer.writeheader()

            for row in results_full:
                writer.writerow(row)

        print(f"Results saved to {OUTPUT_PATH}")

