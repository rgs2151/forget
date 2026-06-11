import argparse

import pandas as pd
from dotenv import load_dotenv

from llm import GPUPool, detect_template
from refuse.activations import clean_answer_text, collect_activations
from refuse.prompts import BASELINE_SYSTEM


load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--store", default="store/qwen3b_inhouse")
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()

model_path = args.model_path
template = detect_template(model_path)
pool = GPUPool.from_model_path(model_path, [0, 1], template=template)

df = pd.read_csv(f"{args.store}/baseline_train.csv")
cleaner = lambda text: clean_answer_text(
    next(iter(pool.llms.values())).tokenizer,
    text,
    template.assistant_end_marker,
)

concept_to_prompts_answers = {}
for concept, frame in df.groupby("concept", sort=False):
    prompts, answers = [], []
    for row in frame.reset_index(drop=True).itertuples(index=False):
        answer = cleaner(row.baseline_output)
        prompts.append(template.render(BASELINE_SYSTEM, row.question, answer))
        answers.append(answer)
    concept_to_prompts_answers[concept] = (prompts, answers)

acts = collect_activations(pool, concept_to_prompts_answers, batch_size=args.batch_size, show_progress=True)
print({concept: tuple(tensor.shape) for concept, tensor in acts.items()})
