"""Optional vLLM serving backend.

The module deliberately imports vLLM inside ``__init__`` so the rest of the
package remains importable on CPU-only machines.
"""

from __future__ import annotations

import time
from typing import Any

from scaletimellm.backends.base import Generation, LMBackend


class VLLMBackend(LMBackend):
    """Fast serving path with the same interface as :class:`HFBackend`."""

    name = "vllm"

    def __init__(
        self,
        model_name: str,
        n_params: int | None = None,
        seed: int = 0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 2048,
        **_: Any,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise ImportError(
                "vLLM is not installed. Install with `pip install -e '.[vllm]'` "
                "or use backend=hf/mock."
            ) from error

        started = time.perf_counter()
        self.model_name = model_name
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if n_params is None:
            config = self.llm.llm_engine.model_config.hf_config
            n_params = int(getattr(config, "num_parameters", 0)) or 494_032_768
        super().__init__(n_params=n_params, seed=seed)
        self.load_time_s = time.perf_counter() - started

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 128,
        seed: int | None = None,
    ) -> list[list[Generation]]:
        if n <= 0:
            raise ValueError("n must be positive")
        seed = self.seed if seed is None else int(seed)
        params = self._SamplingParams(
            n=n,
            temperature=max(float(temperature), 1e-5) if temperature > 0 else 0.0,
            max_tokens=max_new_tokens,
            seed=seed,
        )
        results = self.llm.generate(prompts, params)
        output: list[list[Generation]] = []
        for request in results:
            prompt_tokens = len(request.prompt_token_ids)
            generations = []
            for index, candidate in enumerate(request.outputs):
                completion_tokens = len(candidate.token_ids)
                self._record(prompt_tokens, completion_tokens)
                generations.append(
                    Generation(
                        text=candidate.text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        sample_index=index,
                        seed=seed,
                    )
                )
            output.append(generations)
        return output
