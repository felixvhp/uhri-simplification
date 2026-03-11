
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
    Original "legaleese" text, plus potentially some context.
Output:
    Text translated to plain English.

Open issues:
    The code needs to be modularized and organized.
    The model will need more context to perform better, eg what countries it is about, but this depends on the data format.
        Pre-process the input data
    Spending some time on choosing the best model might be wise.
"""

import ollama

# Note: I simply picked a model I allready had installed for demonstration purposes.
MODEL = 'phi4' # Find others at: https://ollama.com/search
SYSTEM_PROMPT = \
"""
Please summarise the following legal text into plain simple English.
Be sure to use only information from the text.
"""

# Hardcoding this for now untill there is more clarity on the data format.
test_text = """
4.The Committee acknowledges the measures taken by the State Party in areas relevant to the Convention following the adoption of the Committee’s previous concluding observations, including:

(a)The adoption of the Guidelines for the Search for Missing Persons, in July 2025;

(b)The adoption of the Judicial Reform Strategy 2024–2027;

(c)The adoption, by the Supreme State Prosecutor’s Office, of the 2024–2027 strategy for the investigation of war crimes, and the corresponding initial action plan for 2024–2025;

(d)The amendment of the Criminal Code in 2023, ensuring the non-applicability of the statute of limitations to torture;

(e)The amendments to the Law on the Judicial Council and Judges and the law on the State Prosecutor’s office to, inter alia, enhance the independence and accountability of judges and prosecutors.
"""

# Defining the input data, for now just using the test_text
input = test_text
prompt = SYSTEM_PROMPT + input
output = ollama.generate(model = MODEL, prompt= prompt)
print(output['response'])

