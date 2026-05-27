
"""
Pkg 3: Natural language translation [last updated 2026-03-11]

The goal of this document is to test out using (local-)LLM's for simplifying the legalese texts.
Motivation:
    Using Ollama allows different models to be used, and more up to date ones.
    Since presumably original text will always be provided also hallucinations should be less of an issue.
    Hallucinations also get reduced by the small context window, so no need for relying on background information.
    Enables easy tweaking of the "translation" through tweaking eg. the model or the system prompt.

Requirements:
    Ollama https://ollama.com/
        Be sure to install the model under the MODEL variable also.
    Ollama python library https://github.com/ollama/ollama-python

Input:
    A UHRI export file at DATA_PATH, loaded via src.Dataloader.load_records.

Output:
    Simplified text printed to stdout. Not persisted anywhere yet.

Open issues:
    The code needs to be modularized and organized.
    Work more on pre-processing (adding context)
    Spending some time on choosing the best model might be wise.
    Output persistence, right now we're just printing
"""

import ollama
from src.uhri_record import UHRIRecord
from src.Dataloader import load_records
import csv
import re
import anthropic
from openai import OpenAI
from src.evaluation import readability, score_simplification
from src.prompts import PROMPTS, DEFAULT_PROMPT, build_prompt

# Note: I simply picked a model I allready had installed for demonstration purposes.
DATA_PATH = "../data/email_recs.xlsx"
MODEL = "gpt-5.5" # Find others at: https://ollama.com/search
OUTPUT_PATH = "../data/simplification_results3.csv"
openai_client = OpenAI() 

# ================ End of test-prompt setup ================================
def simplify(record: UHRIRecord, model: str = MODEL, template: str = DEFAULT_PROMPT) -> str:
    """
    Simplify a single UHRI record.

    Builds the prompt with build_prompt() and sends it to the given model via
    ollama.generate. No post-processing.

    Args:
        record: The UHRIRecord to simplify.
        model: Ollama model name (must already be pulled locally)
        template: Prompt template passed through to build_prompt().
                  Defaults to the module-level DEFAULT_PROMPT.

    Returns:
        The model's generated simplifiedtext.
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
    
    results_full = []
    for record in records:
        print(f"\n{'=' * 60}")
        print(f"{record.recommending_body} to {record.country} [{record.type}]")
        print(f"ORIGINAL: {record.text}")

        results = {"original_text": record.text}
        orig_read = readability(record.text) 

        for name, text in PROMPTS.items():
            print(f"\nSimplified with {name}:")
            conversion = simplify(record, template = text)
            print(conversion)
            results[name] = conversion

            metrics = score_simplification(record.text, conversion, orig_read)
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


    


