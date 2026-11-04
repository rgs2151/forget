"""AxBench LMJudgeEvaluator prompts and scoring, with a resumable API cache.

Prompts are copied verbatim from AxBench commit
41c8332543e5a631f9a8c0a9df38799893ace758. See test_core.py for source parity.
"""

import asyncio
import hashlib
import json
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .core import CACHE, ROOT


TEMPLATES = {
    'concept_relevance': """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
""",
    'instruction_relevance': """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate the instruction \
relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the \
instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, \
and 2 indicates it is more clearly and directly related to the instruction. Provide your rating \
using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
""",
    'fluency': """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""}


def parse_rating(completion):
    if 'Rating:' not in completion:
        return 0., False
    text = completion.rsplit('Rating:', 1)[-1].strip().split('\n')[0].strip()
    text = text.replace('[', '').replace(']', '').rstrip('.').strip('"').strip("'").strip('*').strip()
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text) is None:
        return 0., False
    rating = float(text)
    return (rating, True) if 0 <= rating <= 2 else (0., False)


def aggregate(scores):
    return 0. if 0 in scores else len(scores) / sum(1. / value for value in scores)


async def score_frame(frame, cfg):
    load_dotenv(ROOT / '.env')
    directory = CACHE / 'judge_responses'
    directory.mkdir(parents=True, exist_ok=True)
    prompts = {}
    row_keys = []
    for row in frame.itertuples():
        keys = []
        for axis, template in TEMPLATES.items():
            prompt = template.format(concept=cfg['judge_concept'], instruction=row.question, sentence=row.model_output)
            key = hashlib.sha256(json.dumps([cfg['judge_model'], cfg['judge_temperature'], prompt]).encode()).hexdigest()
            prompts[key] = prompt
            keys.append((axis, key))
        row_keys.append(keys)
    responses = {}
    pending = []
    for key, prompt in prompts.items():
        path = directory / f'{key}.json'
        if path.exists():
            responses[key] = json.loads(path.read_text())
        else:
            pending.append((key, prompt))
    print(f'AxBench judge: {len(frame)} rows; {len(pending)} new rubric requests after exact-prompt deduplication', flush=True)
    async with AsyncOpenAI(timeout=60., max_retries=3) as client:
        async def request(key, prompt):
            response = await client.chat.completions.create(model=cfg['judge_model'],
                messages=[{'role': 'user', 'content': prompt}], temperature=cfg['judge_temperature'])
            text = response.choices[0].message.content.strip()
            rating, valid = parse_rating(text)
            result = dict(completion=text, rating=rating, parsed=valid,
                          model=response.model, usage=response.usage.model_dump())
            path = directory / f'{key}.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(path)
            responses[key] = result

        for start in range(0, len(pending), cfg['judge_batch_size']):
            await asyncio.gather(*(request(k, p) for k, p in pending[start:start + cfg['judge_batch_size']]))
            print(f'AxBench judge: {min(start + cfg["judge_batch_size"], len(pending))}/{len(pending)} requests complete', flush=True)
    result = frame.copy()
    for axis in TEMPLATES:
        selected = [responses[dict(keys)[axis]] for keys in row_keys]
        result[f'axbench_{axis}'] = [item['rating'] for item in selected]
        result[f'axbench_{axis}_completion'] = [item['completion'] for item in selected]
        result[f'axbench_{axis}_parsed'] = [item['parsed'] for item in selected]
    result['axbench_aggregate'] = [aggregate(list(row)) for row in result[
        [f'axbench_{axis}' for axis in TEMPLATES]].itertuples(index=False, name=None)]
    return result


def judge_frame(frame, cfg):
    return asyncio.run(score_frame(frame, cfg))
