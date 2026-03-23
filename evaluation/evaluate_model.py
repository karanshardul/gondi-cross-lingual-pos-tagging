import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from training.common_training import (
    load_config,
    prepare_data_and_tokenizer,
    build_full_model,
    get_device,
)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def evaluate_checkpoint(config_path: str, experiment: str, checkpoint_path: str, split: str = "test") -> str:
    cfg = load_config(config_path)
    tokenizer, train_ds, dev_ds, test_ds, label2id, id2label = prepare_data_and_tokenizer(cfg)
    device = get_device(cfg)

    if experiment == "baseline":
        flags = dict(use_adapters=False, use_router=False, use_similarity_bias=False, use_lexicon_loss=False)
    elif experiment == "static_adapters":
        flags = dict(use_adapters=True, use_router=False, use_similarity_bias=False, use_lexicon_loss=False)
    elif experiment == "router":
        flags = dict(use_adapters=True, use_router=True, use_similarity_bias=False, use_lexicon_loss=False)
    elif experiment == "full_model":
        flags = dict(use_adapters=True, use_router=True, use_similarity_bias=True, use_lexicon_loss=True)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    model = build_full_model(cfg, label2id=label2id, **flags)
    model.to(device)
    ckpt = _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    ds = {"train": train_ds, "dev": dev_ds, "test": test_ds}[split]
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch], 0),
            "labels": torch.stack([b["labels"] for b in batch], 0),
            "word": [b["word"] for b in batch],
        },
    )

    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=None,
                words=batch["word"],
                lambda_lex=0.0,
                label2id=label2id,
                lexicon=None,
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(id2label)))).astype(int)

    metrics: Dict[str, Any] = {
        "experiment": experiment,
        "split": split,
        "checkpoint_path": checkpoint_path,
        "epoch": int(ckpt.get("epoch", -1)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "id2label": {str(k): v for k, v in id2label.items()},
    }

    out_dir = os.path.join(cfg["experiment"]["output_dir"], experiment)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{split}_epoch{metrics['epoch']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, required=True, choices=["baseline", "static_adapters", "router", "full_model"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    args = parser.parse_args()

    out = evaluate_checkpoint(args.config, args.experiment, args.checkpoint, split=args.split)
    print(f"Saved metrics to {out}")

