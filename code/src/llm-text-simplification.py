
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

load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present

DATA_PATH = "../data/email_recs.xlsx"
MODEL = "claude-haiku-4-5-20251001"  # fast + cheap; swap for claude-sonnet-4-6 for higher quality
OUTPUT_PATH = "../data/simplification_results.csv"
FULL_TESTS = False

# =========== Prompts go here ============
prompt_felix = """\
Simplify the following {type} from {body} regarding {country} into plain English that a \
non-expert can understand. Preserve the meaning and any specific obligations or actions mentioned.
Output ONLY the rewritten text. Do not include any preamble such as "Here is the rewritten/simplified text.
 
Original text:
{text}
 
Simplified version:"""

too_simple_summary = """\
Please summarise the following {type} from {body} into plain simple English.
Be sure to use only information from the text.
The text is about {country}

Original text:
{text}

Simplified version:
"""

transcribe_litteral = """
You are a text transcriber. Rewrite the following legal text \
by only simplifying the vocabulary - keep the original sentence structure and meaning as close as possible.
Replace complex words with simpler synonyms but maintain the formal tone. 
Do not summarize, explain, or add anything. Keep it literal.

For additional context, the text is type: {type}, written by {body}, and is concearning {country}.

Original text:
{text}

Simplified version:
"""

transcribe_accessible = """
You are a accessibility converter. Your task is to make the following legal text 
accessible to someone with limited English. Preserve the original meaning exactly. 
Use very simple words and short sentences. Keep all the detail. 
Do not add any information not in the original text.

For additional context, the text is type: {type}, written by {body}, and is concearning {country}.

Original text:
{text}

Simplified version:
"""

summarise = """
You are a legal text simplifier. Rewrite the following legal text \
into plain, simple English that a general audience can understand. 
Use common words and short sentences. Keep all key facts and events. 
Do not add any information not in the original text.

For additional context, the text is type: {type}, written by {body}, and is concearning {country}.

Original text:
{text}

Simplified version:
"""

prompt_dict = {
    "prompt_felix": prompt_felix,
    "simple_summary": too_simple_summary,
    "literal_transcribe": transcribe_litteral,
    "accessible_transcribe": transcribe_accessible,
    "summarise": summarise,
}

# ================ End of test-prompt setup ================================

# Backup system prompt so that code can remain unchanged.
SYSTEM_PROMPT = prompt_felix

def build_prompt(record: UHRIRecord, template: str = SYSTEM_PROMPT) -> str:
    """
    Build a prompt from a record and a template.

    Args:
        record: The UHRIRecord to simplify.
        template: Prompt template with {type}, {body}, {country}, {text} slots.
                  Defaults to the module-level SYSTEM_PROMPT.

    Returns:
        The fully-formatted prompt string, ready to send to the model.
    """
    return template.format(
        type="recommendation" if record.is_recommendation else "observation",
        body=record.recommending_body,
        country=record.country,
        text=record.text,
    )

def simplify(record: UHRIRecord, model: str = MODEL, template: str = SYSTEM_PROMPT) -> str:
    """
    Simplify a single UHRI record.

    Builds the prompt with build_prompt() and sends it to the given model via
    the Anthropic messages API. No post-processing.

    Args:
        record: The UHRIRecord to simplify.
        model: Claude model ID (e.g. "claude-haiku-4-5-20251001", "claude-sonnet-4-6")
        template: Prompt template passed through to build_prompt().
                  Defaults to the module-level SYSTEM_PROMPT.

    Returns:
        The model's generated simplified text.
    """
    prompt = build_prompt(record, template)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

if __name__ == "__main__":
    records = load_records(DATA_PATH)

    if not FULL_TESTS:
        record = records[0]
        print("\n Original:")
        print(record.text)
        print("\n Prompt:")
        print(SYSTEM_PROMPT)
        print("\n Conversion:")
        print(simplify(record))
    
     
    # Runs all the prompts on all the saved example items.
    if FULL_TESTS:
        results_full = []
        # Test with first 3 records
        for record in records[:3]:
            print(f"\n{'=' * 60}")
            print(f"{record.recommending_body} to {record.country} [{record.type}]")
            print(f"ORIGINAL: {record.text}")

            results = {"original_text": record.text}

            for name, text in prompt_dict.items():
                print(f"\nSimplified with {name}:")
                conversion = simplify(record, template = text)
                print(conversion)
                results[name] = conversion

            results_full.append(results)

        # Write to CSV
        col_names = results_full[0].keys()
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=col_names)
            writer.writeheader()

            for row in results_full:
                writer.writerow(row)

        print(f"Results saved to {OUTPUT_PATH}")


    


