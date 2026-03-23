import argparse
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from training.common_training import (
    load_config,
    prepare_data_and_tokenizer,
    build_full_model,
    create_dataloaders,
    create_optimizer_and_scheduler,
    set_seed,
    get_device,
    save_checkpoint,
    setup_logging,
    save_metrics_json,
)

from utils.dataset_loader import WORD_COLUMN
from utils.lexicon_utils import build_lexicon_lookup


def move_batch_to_device(batch, device):
    """
    Move only tensor values to device.
    Keep metadata like words untouched.
    """
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def train_full_model(config_path: str):

    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])

    tokenizer, train_ds, dev_ds, test_ds, label2id, id2label = prepare_data_and_tokenizer(cfg)

    df = pd.read_csv(cfg["data"]["lexicon_path"])
    lexicon = build_lexicon_lookup(df, word_column=WORD_COLUMN, pos_column="pos")

    device = get_device(cfg)
    logger = setup_logging(cfg["experiment"]["output_dir"], "full_model")

    logger.info(f"Using device: {device}")
    logger.info(f"Train size: {len(train_ds)} | Dev size: {len(dev_ds)} | Test size: {len(test_ds)}")

    model = build_full_model(
        cfg,
        use_adapters=True,
        use_router=True,
        use_similarity_bias=True,
        use_lexicon_loss=True,
        label2id=label2id,
    )

    model.to(device)

    train_loader, dev_loader = create_dataloaders(
        train_ds,
        dev_ds,
        cfg["training"]["batch_size"],
    )

    num_training_steps = len(train_loader) * cfg["training"]["num_epochs"]

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        cfg,
        num_training_steps,
    )

    lambda_lex = cfg["lexicon_loss"]["lambda_weight"]

    # =============================
    # Resume from checkpoint
    # =============================

    start_epoch = 1
    resume_ckpt = cfg["training"].get("resume_from_checkpoint")

    if resume_ckpt and os.path.exists(resume_ckpt):
        logger.info(f"Resuming training from checkpoint: {resume_ckpt}")

        ckpt = torch.load(resume_ckpt, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1

    # =============================
    # Training loop
    # =============================

    for epoch in range(start_epoch, cfg["training"]["num_epochs"] + 1):

        model.train()
        total_loss = 0.0

        for batch in train_loader:

            words = batch["word"]

            batch = move_batch_to_device(batch, device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                words=words,
                lambda_lex=lambda_lex,
                label2id=label2id,
                lexicon=lexicon,
            )

            loss = outputs["loss"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["gradient_clip"],
            )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        logger.info(f"[FullModel] Epoch {epoch} - train loss: {avg_loss:.4f}")

        # =============================
        # Dev evaluation
        # =============================

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for batch in dev_loader:

                words = batch["word"]

                labels = batch["labels"].to(device)

                inputs = move_batch_to_device(batch, device)

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=None,
                    words=words,
                    lambda_lex=lambda_lex,
                    label2id=label2id,
                    lexicon=lexicon,
                )

                logits = outputs["logits"]

                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)

        macro_f1 = f1_score(
            all_labels,
            all_preds,
            average="macro",
        )

        logger.info(
            f"[FullModel] Epoch {epoch} - dev accuracy: {acc:.4f} | macro F1: {macro_f1:.4f}"
        )

        save_metrics_json(
            cfg,
            "full_model",
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "dev_accuracy": acc,
                "dev_macro_f1": macro_f1,
            },
            filename=f"metrics_epoch{epoch}.json",
        )

        if cfg["training"]["save_every_epoch"]:

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                cfg,
                experiment_name="full_model",
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    train_full_model(args.config)