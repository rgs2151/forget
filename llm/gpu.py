from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

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

        shards = chunk_split(prompts, len(self.gpu_ids))

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


def chunk_split(items, n):
    if n <= 1 or len(items) <= 1:
        return [items]
    size = (len(items) + n - 1) // n
    return [items[i:i + size] for i in range(0, len(items), size)]
