from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch as t
from tqdm.auto import tqdm

from .activations import collect_answer_activations_batched
from .chat_templates import detect_template
from .model import load_llm


class GPUPool:
    def __init__(self, llm_factory, gpu_ids, template):
        self.template = template
        self.gpu_ids = list(gpu_ids)
        self.llms = {gpu_id: llm_factory(gpu_id) for gpu_id in self.gpu_ids}

    @classmethod
    def from_model_path(cls, model_path, gpu_ids, template=None, hf_token=None):
        if template is None:
            template = detect_template(model_path)
        return cls(
            lambda gid: load_llm(model_path, gid, template, hf_token),
            gpu_ids,
            template,
        )

    def __len__(self):
        return len(self.gpu_ids)

    def map(self, fn, shards):
        if len(shards) > len(self.gpu_ids):
            raise ValueError("shards cannot exceed gpu count.")
        if len(self.gpu_ids) == 1:
            llm = self.llms[self.gpu_ids[0]]
            return [fn(llm, shard) for shard in shards]
        with ThreadPoolExecutor(len(shards)) as executor:
            futures = [
                executor.submit(fn, self.llms[gpu_id], shard)
                for gpu_id, shard in zip(self.gpu_ids, shards)
            ]
            return [future.result() for future in futures]

    def generate(self, prompts, generation_kwargs=None, batch_size=64, trim_fn=None, show_progress=True):
        if not prompts:
            return []
        gen_kwargs = dict(generation_kwargs or {})
        gen_kwargs.setdefault("max_new_tokens", 64)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("temperature", 1.0)

        shards = _chunk_split(prompts, len(self.gpu_ids))

        def run(llm, shard):
            outputs = []
            iterator = range(0, len(shard), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="generate")
            for start in iterator:
                batch = shard[start:start + batch_size]
                llm.reset_all()
                batch_outputs = llm.batch_generate(batch, **gen_kwargs)
                llm.reset_all()
                if trim_fn is not None:
                    batch_outputs = [trim_fn(o) for o in batch_outputs]
                outputs.extend(batch_outputs)
            return outputs

        results = self.map(run, shards)
        flat = []
        for shard_result in results:
            flat.extend(shard_result)
        return flat

    def collect_activations(self, concept_to_prompts_answers, batch_size=128, show_progress=True):
        concepts = list(concept_to_prompts_answers.keys())
        if not concepts:
            return {}, {}

        n_shards = min(len(self.gpu_ids), len(concepts))
        shards = [concepts[i::n_shards] for i in range(n_shards)]

        def run(llm, concept_shard):
            acts, masks = {}, {}
            iterator = tqdm(concept_shard, desc="activations") if show_progress else concept_shard
            for concept in iterator:
                prompts, answers = concept_to_prompts_answers[concept]
                acts[concept], masks[concept] = collect_answer_activations_batched(
                    llm,
                    prompts,
                    answers,
                    batch_size=batch_size,
                    assistant_end_marker=self.template.assistant_end_marker,
                    return_token_mask=True,
                    show_progress=False,
                )
            return acts, masks

        results = self.map(run, shards)
        merged_acts, merged_masks = {}, {}
        for shard_acts, shard_masks in results:
            merged_acts.update(shard_acts)
            merged_masks.update(shard_masks)
        return merged_acts, merged_masks

    def run_jobs(self, jobs, steering, generation_kwargs=None, batch_size=64,
                 trim_fn=None, result_metadata=None):
        from .intervention import run_generation_jobs

        shards = _split_jobs_for_gpus(jobs, len(self.gpu_ids))

        def run(llm, jobs_shard):
            return run_generation_jobs(
                llm,
                jobs_shard,
                steering,
                generation_kwargs=generation_kwargs,
                trim_output_fn=trim_fn or (lambda x: x),
                batch_size=batch_size,
                result_metadata=result_metadata,
            )

        results = self.map(run, shards)
        merged = pd.concat(results, ignore_index=True)
        merged = merged.sort_values(["prompt_index", "target", "scale"]).reset_index(drop=True)
        return merged


def _chunk_split(items, n):
    if n <= 1 or len(items) <= 1:
        return [items]
    size = (len(items) + n - 1) // n
    return [items[i:i + size] for i in range(0, len(items), size)]


def _split_jobs_for_gpus(jobs, n_gpus):
    if n_gpus <= 1:
        return [jobs.reset_index(drop=True)]
    prompt_indices = pd.Series(jobs["prompt_index"].drop_duplicates().tolist())
    shards = []
    for i in range(n_gpus):
        gpu_indices = prompt_indices.iloc[i::n_gpus]
        gpu_jobs = jobs[jobs["prompt_index"].isin(gpu_indices)].reset_index(drop=True)
        shards.append(gpu_jobs)
    return shards
