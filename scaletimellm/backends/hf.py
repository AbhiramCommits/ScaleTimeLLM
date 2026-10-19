"""Hugging Face Transformers backend with batched seeded sampling."""

from __future__ import annotations

import time
from typing import Any

from scaletimellm.backends.base import Generation, LMBackend
from scaletimellm.utils.seeding import set_all_seeds


class HFBackend(LMBackend):
    """Causal-LM backend built on ``transformers``.

    Weights are downloaded lazily on construction.  The backend pads on the left
    so decoder-only generation is correct.  A fresh generator is used for each
    call so results are reproducible for a fixed seed.
    """

    name = "hf"

    def __init__(
        self,
        model_name: str,
        n_params: int | None = None,
        device: str = "auto",
        dtype: str = "float32",
        seed: int = 0,
        trust_remote_code: bool = False,
        **_: Any,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.model_name = model_name
        self.device = self._resolve_device(device, torch)
        self.dtype = dtype
        set_all_seeds(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        measured = sum(p.numel() for p in self.model.parameters())
        super().__init__(n_params=measured if n_params is None else n_params, seed=seed)
        self.measured_n_params = measured
        self.load_time_s = time.perf_counter() - started

    @staticmethod
    def _resolve_device(device: str, torch: Any) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 128,
        seed: int | None = None,
    ) -> list[list[Generation]]:
        import torch

        if n <= 0:
            raise ValueError("n must be positive")
        if not prompts:
            return []
        seed = self.seed if seed is None else int(seed)
        set_all_seeds(seed)
        output: list[list[Generation]] = []
        for prompt in prompts:
            encoded = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            prompt_tokens = int(encoded["input_ids"].shape[1])
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            do_sample = temperature > 0
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": n,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                kwargs["temperature"] = max(float(temperature), 1e-5)
                kwargs["top_p"] = 0.95
            with torch.inference_mode():
                sequences = self.model.generate(**encoded, **kwargs)
            generations: list[Generation] = []
            for index, sequence in enumerate(sequences):
                new_tokens = sequence[prompt_tokens:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                completion_tokens = int(new_tokens.shape[0])
                self._record(prompt_tokens, completion_tokens)
                generations.append(
                    Generation(
                        text=text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        sample_index=index,
                        seed=seed,
                    )
                )
            output.append(generations)
        return output

    def info(self) -> dict[str, Any]:
        info = super().info()
        info.update(
            {
                "model_name": self.model_name,
                "device": self.device,
                "dtype": self.dtype,
                "measured_n_params": self.measured_n_params,
                "load_time_s": self.load_time_s,
            }
        )
        return info
