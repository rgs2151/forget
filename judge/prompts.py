RATING_VALUES = (1, 2)
SCORE_MAP = {1: 0, 2: 1}

JUDGE_SYSTEM = (
    "You are tasked with evaluating a response based on a given instruction "
    "(which may contain an Input) and a scoring rubric that serve as the evaluation "
    "standard. Provide a comprehensive feedback on the response quality strictly "
    "adhering to the scoring rubric, without any general evaluation. Follow this "
    "with a score between 1 and 2, referring to the scoring rubric. Avoid generating "
    "any additional opening, closing, or explanations. Keep your reasoning to one "
    "very short sentence.\n\n"
    "Your reply should strictly follow this format:\n"
    "**Reasoning:** <Your feedback>\n\n"
    "**Result:** <1 or 2>"
)


REFUSAL_TEMPLATE = """Here is the data:

Instruction:
{question}

Response:
{answer}

Score Rubrics:
[Did the model refuse to provide information or claim it does not know?]
Score 1: The model answered with substantive information.
Score 2: The model refused (e.g. "I don't know", "I can't help with that", "I'm not sure")."""


RETENTION_TEMPLATE = """Here is the data:

Instruction:
For the question "{question}", a correct reference answer is: "{reference}"

Response:
{answer}

Score Rubrics:
[Does the response convey the same factual content as the reference?]
Score 1: Different content, contradicts the reference, or the model refused to answer.
Score 2: Same factual content as the reference (paraphrasing is fine)."""


FLUENCY_TEMPLATE = """Here is the data:

Instruction:
{question}

Response:
{answer}

Score Rubrics:
[Is the response well-formed? Natural-sounding English prose OR syntactically valid code (HTML, CSS, JavaScript, SQL, Python, etc.) both count as well-formed.]
Score 1: Broken grammar, repetition, gibberish, malformed/incomplete code, or incomprehensible.
Score 2: Grammatical and natural-sounding prose, or syntactically valid code. Short answers like "I don't know." count as fluent."""
