
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

# Note: I simply picked a model I allready had installed for demonstration purposes.
DATA_PATH = "../data/email_recs.xlsx"
MODEL = 'llama3:latest' # Find others at: https://ollama.com/search
SYSTEM_PROMPT = """\
Simplify the following {type} from {body} regarding {country} into plain English that a \
non-expert can understand. Preserve the meaning and any specific obligations or actions mentioned.
Output ONLY the rewritten text. Do not include any preamble such as "Here is the rewritten/simplified text.
 
Original text:
{text}
 
Simplified version:"""

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
    ollama.generate. No post-processing.

    Args:
        record: The UHRIRecord to simplify.
        model: Ollama model name (must already be pulled locally)
        template: Prompt template passed through to build_prompt().
                  Defaults to the module-level SYSTEM_PROMPT.

    Returns:
        The model's generated simplified text.
    """
    prompt = build_prompt(record, template)
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]

records = load_records(DATA_PATH)
 
# Test with first 3 records
for record in records[:3]:
    print(f"\n{'=' * 60}")
    print(f"{record.recommending_body} to {record.country} [{record.type}]")
    print(f"ORIGINAL: {record.text}")
    print(f"\nSIMPLIFIED:")
    print(simplify(record))

