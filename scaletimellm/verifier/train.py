"""Train tiny outcome and pairwise verifiers, optionally with TRL's RewardTrainer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from scaletimellm.config import ExperimentConfig
from scaletimellm.utils.seeding import set_all_seeds
from scaletimellm.verifier.data import (
    generate_verifier_data,
    load_rows,
    preference_pairs,
    verifier_input_text,
)
from scaletimellm.verifier.eval import evaluate_scores, evaluate_verifier
from scaletimellm.verifier.score import (
    ByteTokenizer,
    TinyVerifier,
    Verifier,
    save_custom_checkpoint,
)


def _row_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [verifier_input_text(str(row["prompt"]), str(row["completion"])) for row in rows]


def _batch_logits(
    model: TinyVerifier, texts: list[str], tokenizer: ByteTokenizer, batch_size: int = 64
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            ids = torch.tensor([tokenizer.encode(text) for text in batch], dtype=torch.long)
            scores.extend(float(value) for value in model(ids).cpu().tolist())
    return scores


def _eval_model(model: TinyVerifier, tokenizer: ByteTokenizer, rows: list[dict[str, Any]]) -> dict:
    scores = _batch_logits(model, _row_texts(rows), tokenizer)
    return evaluate_scores(rows, scores)


def _custom_train_loop(
    *,
    objective: str,
    pairs: list[dict[str, str]],
    rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    epochs: int,
    seed: int,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
) -> tuple[TinyVerifier, ByteTokenizer, dict[str, Any]]:
    set_all_seeds(seed)
    tokenizer = ByteTokenizer(max_len=192)
    model = TinyVerifier(max_len=192)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    if objective == "outcome":
        texts = _row_texts(rows)
        labels = [float(row["label"]) for row in rows]
    else:
        if not pairs:
            raise ValueError("no preference pairs available for pairwise training")
        texts = []
        labels = []
    if not texts and objective == "outcome":
        raise ValueError("no training examples for outcome objective")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_auroc = -math.inf
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        if objective == "outcome":
            order = torch.randperm(len(texts), generator=generator).tolist()
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch_texts = [texts[index] for index in indices]
                ids = torch.tensor(
                    [tokenizer.encode(text) for text in batch_texts], dtype=torch.long
                )
                logits = model(ids)
                target = torch.tensor([labels[index] for index in indices], dtype=torch.float32)
                loss = loss_fn(logits, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1
        else:
            order = torch.randperm(len(pairs), generator=generator).tolist()
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                chosen_texts = [pairs[index]["chosen"] for index in indices]
                rejected_texts = [pairs[index]["rejected"] for index in indices]
                chosen_ids = torch.tensor(
                    [tokenizer.encode(text) for text in chosen_texts], dtype=torch.long
                )
                rejected_ids = torch.tensor(
                    [tokenizer.encode(text) for text in rejected_texts], dtype=torch.long
                )
                chosen = model(chosen_ids)
                rejected = model(rejected_ids)
                loss = torch.nn.functional.softplus(-(chosen - rejected)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1
        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            metrics = _eval_model(model, tokenizer, eval_rows)
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / max(1, batches),
                    "eval_auroc": metrics["auroc"],
                }
            )
            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    model.eval()
    return model, tokenizer, {"history": history, "best_eval_auroc": best_auroc}


def _build_bpe_tokenizer(texts: list[str]) -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        eos_token="[SEP]",
    )


def train_trl_pairwise(
    pairs: list[dict[str, str]],
    *,
    output_dir: str | Path,
    eval_rows: list[dict[str, Any]],
    seed: int = 0,
    max_steps: int = 40,
) -> dict[str, Any]:
    """Train a small sequence-classification reward model with TRL."""
    from datasets import Dataset
    from transformers import BertConfig, BertForSequenceClassification
    from trl import RewardConfig, RewardTrainer

    set_all_seeds(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [pair["chosen"] for pair in pairs] + [pair["rejected"] for pair in pairs]
    tokenizer = _build_bpe_tokenizer(texts)
    dataset = Dataset.from_dict(
        {
            "chosen": [pair["chosen"] for pair in pairs],
            "rejected": [pair["rejected"] for pair in pairs],
        }
    )
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=192,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForSequenceClassification(config)
    arguments = RewardConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=16,
        max_steps=max_steps,
        learning_rate=1e-3,
        warmup_steps=5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
        remove_unused_columns=True,
        max_length=192,
        optim="adamw_torch",
        seed=seed,
        data_seed=seed,
    )
    trainer = RewardTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    train_output = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    verifier_config = {
        "kind": "hf_seqcls",
        "objective": "pairwise_trl",
        "max_len": 192,
        "threshold": 0.0,
        "trainer": "trl.RewardTrainer",
        "max_steps": max_steps,
    }
    (output_dir / "verifier_config.json").write_text(
        json.dumps(verifier_config, indent=2), encoding="utf-8"
    )
    verifier = Verifier.from_checkpoint(output_dir, objective="pairwise_trl")
    metrics = evaluate_verifier(verifier, eval_rows)
    metrics.update(
        {
            "training_loss": float(train_output.training_loss),
            "wall_time_s": time.perf_counter() - started,
            "trainer": "trl.RewardTrainer",
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def train_verifiers(
    config: ExperimentConfig,
    *,
    n_train_problems: int = 120,
    n_eval_problems: int = 40,
    k: int = 4,
    epochs: int = 35,
    seed: int = 0,
    artifact_dir: str | Path = "artifacts/verifier",
    results_path: str | Path = "results/verifier_metrics.json",
    train_trl: bool = True,
    reuse_data: bool = False,
) -> dict[str, Any]:
    """Generate data, train outcome/pairwise verifiers, and write metrics."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_dir = artifact_dir / "data"
    metadata_path = data_dir / "metadata.json"
    if reuse_data and metadata_path.exists():
        data_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        data_metadata = generate_verifier_data(
            config,
            n_train_problems=n_train_problems,
            n_eval_problems=n_eval_problems,
            k=k,
            seed=seed,
            output_dir=data_dir,
        )
    train_rows = load_rows(data_dir / "train.jsonl")
    eval_rows = load_rows(data_dir / "eval.jsonl")
    pairs = preference_pairs(train_rows)
    result: dict[str, Any] = {
        "data": data_metadata,
        "seed": seed,
        "epochs": epochs,
        "pairs": len(pairs),
        "training_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "objectives": {},
        "backend": data_metadata.get("backend"),
        "synthetic": data_metadata.get("synthetic"),
    }
    started = time.perf_counter()
    for objective in ("outcome", "pairwise"):
        directory = artifact_dir / objective
        model, tokenizer, training_info = _custom_train_loop(
            objective=objective,
            pairs=pairs,
            rows=train_rows,
            eval_rows=eval_rows,
            epochs=epochs,
            seed=seed,
        )
        metrics = _eval_model(model, tokenizer, eval_rows)
        metadata = {
            "objective": objective,
            "training_info": training_info,
            "data": data_metadata,
            "epochs": epochs,
            "seed": seed,
        }
        save_custom_checkpoint(
            model, directory, objective=objective, metadata=metadata
        )
        (directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        result["objectives"][objective] = metrics
    if train_trl:
        try:
            trl_metrics = train_trl_pairwise(
                pairs,
                output_dir=artifact_dir / "trl_pairwise",
                eval_rows=eval_rows,
                seed=seed,
            )
            result["objectives"]["pairwise_trl"] = trl_metrics
            result["trl_status"] = "trained"
        except Exception as error:  # pragma: no cover - optional dependency path
            result["trl_status"] = f"failed: {type(error).__name__}: {error}"
    else:
        result["trl_status"] = "disabled"
    result["wall_time_s"] = time.perf_counter() - started
    result["artifact_dir"] = str(artifact_dir)
    result["best_checkpoint"] = str(artifact_dir / "outcome")
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
