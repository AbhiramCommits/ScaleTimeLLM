"""Verifier scoring: trained tiny transformers plus oracle/length baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scaletimellm.strategies.verifiers import LengthPriorVerifier, OracleVerifier
from scaletimellm.verifier.data import verifier_input_text

PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
BYTE_OFFSET = 3
BYTE_VOCAB = 256 + BYTE_OFFSET


class ByteTokenizer:
    """Deterministic byte tokenizer with a head+tail view of the text."""

    def __init__(self, max_len: int = 192) -> None:
        self.max_len = int(max_len)

    def encode(self, text: str) -> list[int]:
        """Encode text to a fixed-length byte-token sequence."""
        raw = list(text.encode("utf-8", errors="ignore"))
        budget = self.max_len - 3
        if len(raw) > budget:
            head = budget // 2
            tail = budget - head
            raw = raw[:head] + raw[-tail:]
        tokens = [CLS_ID] + [byte + BYTE_OFFSET for byte in raw] + [SEP_ID]
        if len(tokens) < self.max_len:
            tokens += [PAD_ID] * (self.max_len - len(tokens))
        return tokens[: self.max_len]


class TinyVerifier(nn.Module):
    """A two-layer transformer encoder with a scalar value head."""

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        max_len: int = 192,
    ) -> None:
        super().__init__()
        self.config = {
            "d_model": d_model,
            "nhead": nhead,
            "layers": layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "max_len": max_len,
        }
        self.embedding = nn.Embedding(BYTE_VOCAB, d_model, padding_idx=PAD_ID)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return one scalar verifier logit per sequence."""
        padding_mask = input_ids.eq(PAD_ID)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.embedding(input_ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden[:, 0])
        return self.head(hidden).squeeze(-1)


def save_custom_checkpoint(
    model: TinyVerifier,
    directory: str | Path,
    *,
    objective: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a tiny verifier and its configuration."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    config = {
        "kind": "custom_byte_transformer",
        "objective": objective,
        "threshold": 0.0,
        **model.config,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (path / "verifier_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (path / "metadata.json").write_text(
        json.dumps(metadata or {}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def load_custom_checkpoint(directory: str | Path) -> tuple[TinyVerifier, dict[str, Any]]:
    """Load a tiny verifier checkpoint and verify its architecture."""
    path = Path(directory)
    config_path = path / "verifier_config.json"
    if not config_path.exists():
        config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = TinyVerifier(
        d_model=int(config["d_model"]),
        nhead=int(config["nhead"]),
        layers=int(config["layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_len=int(config["max_len"]),
    )
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


class Verifier:
    """A batched scalar verifier with a higher-is-better score."""

    name = "learned"

    def __init__(
        self,
        model: Any,
        *,
        kind: str,
        max_len: int = 192,
        tokenizer: Any | None = None,
        checkpoint: str | None = None,
    ) -> None:
        self.model = model
        self.kind = kind
        self.max_len = int(max_len)
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.byte_tokenizer = ByteTokenizer(max_len=max_len)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, objective: str = "outcome") -> Verifier:
        """Load a custom or HF sequence-classification checkpoint.

        ``path`` may point directly at a checkpoint or at a parent containing an
        ``outcome``/``pairwise`` subdirectory.
        """
        path = Path(path)
        candidate = path / objective
        if candidate.exists():
            path = candidate
        config_path = path / "verifier_config.json"
        if not config_path.exists():
            config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"verifier checkpoint not found: {path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        kind = config.get("kind", "custom_byte_transformer")
        if kind == "custom_byte_transformer":
            model, config = load_custom_checkpoint(path)
            return cls(
                model,
                kind="custom",
                max_len=int(config["max_len"]),
                checkpoint=str(path),
            )
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        return cls(
            model,
            kind="hf_seqcls",
            max_len=int(config.get("max_len", 192)),
            tokenizer=tokenizer,
            checkpoint=str(path),
        )

    def score(self, prompt: str, completion: str) -> float:
        """Score one completion."""
        return self.score_many(prompt, [completion])[0]

    def score_many(self, prompt: str, completions: list[str]) -> list[float]:
        """Score a batch of completions for one prompt."""
        if not completions:
            return []
        texts = [verifier_input_text(prompt, completion) for completion in completions]
        with torch.inference_mode():
            if self.kind == "custom":
                ids = torch.tensor(
                    [self.byte_tokenizer.encode(text) for text in texts], dtype=torch.long
                )
                logits = self.model(ids)
            else:
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                logits = self.model(**encoded).logits.reshape(-1)
        return [float(value) for value in logits.detach().cpu().tolist()]

    def probability(self, score: float) -> float:
        """Map a logit to a probability using a stable sigmoid."""
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-score))
        exponent = math.exp(score)
        return exponent / (1.0 + exponent)

    def info(self) -> dict[str, Any]:
        """Checkpoint metadata for result records."""
        return {
            "kind": self.kind,
            "checkpoint": self.checkpoint,
            "max_len": self.max_len,
        }


__all__ = [
    "ByteTokenizer",
    "LengthPriorVerifier",
    "OracleVerifier",
    "TinyVerifier",
    "Verifier",
    "load_custom_checkpoint",
    "save_custom_checkpoint",
]
