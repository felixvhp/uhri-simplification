from src.uhri_record import UHRIRecord
"""
Prompt templates for simplifying UHRI texts.
"""

prompt_felix =  """\
You are simplifying a {type} from {body} concerning {country} for a non-expert reader.

GOALS
- Make the text easier to read using everyday vocabulary and shorter sentences.
- Preserve every named actor, body, obligation, modal verb, and specific reference exactly.

GUIDELINES

1. Word choice: replace formal/legal vocabulary with everyday equivalents.
   "is mandated to" -> "is responsible for"
   "conduct criminal investigations" -> "carry out criminal investigations"
   "characterised by undue delays" -> "marked by delays that are too long"

2. Technical UN/human-rights terms: keep the term, add a brief definition in parentheses on first use.
   "reasonable accommodation" -> "reasonable accommodation (necessary adjustments so persons with disabilities can exercise their rights equally)"
   "non-refoulement" -> "non-refoulement (the prohibition on returning people to countries where they face serious harm)"

3. Legal and country-specific references: keep the proper name, add a brief explanation in parentheses on first use.
   "Istanbul Convention" -> "Istanbul Convention (2011 Council of Europe treaty requiring states to prevent violence against women)"
   "Commonwealth Rent Assistance" -> "Commonwealth Rent Assistance (Australian federal payment to low-income renters)"

4. Sentences: split overly long sentences, but do not fragment into choppy short ones. "John wrote a book. I read that book." is too far.

STRICT PRESERVATION RULES
- Every named body or actor stays named. Do NOT generalize "the Criminal Investigation Department and the Attorney General's Department" to "the relevant authorities".
- Modal verbs keep their force: "must" stays "must", not "should". "shall" stays "shall".
- All conditions, exceptions, and qualifications survive.
- No new information, interpretations, or context that is not in the original.

Output ONLY the rewritten text. No preamble.

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

too_simple_summary = """\
Please summarise the following {type} from {body} into plain simple English.
Be sure to use only information from the text.
The text is about {country}

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

PROMPTS = {
    "prompt_felix": prompt_felix,
    "simple_summary": too_simple_summary,
    "literal_transcribe": transcribe_litteral,
    "accessible_transcribe": transcribe_accessible,
    "summarise": summarise,
}

DEFAULT_PROMPT = prompt_felix

def build_prompt(record: UHRIRecord, template: str = DEFAULT_PROMPT) -> str:
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