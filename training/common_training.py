import os
import json
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

from utils.dataset_loader import (
    set_seed,
    load_lexicon,
    create_tokenizer,
    GondiPosDataset,
)
from utils.lexicon_utils import lexicon_guided_loss
from models.adapter import MultiLanguageAdapters
from models.router import AdapterRouter
from models.pos_head import POSClassificationHead


def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{experiment_name}_{ts}.log")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data_and_tokenizer(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    tokenizer = create_tokenizer(cfg["model"]["pretrained_name"])
    train_ex, dev_ex, test_ex, label2id, id2label = load_lexicon(
        data_cfg["lexicon_path"],
        data_cfg["train_ratio"],
        data_cfg["dev_ratio"],
        data_cfg["test_ratio"],
    )

    if cfg["model"].get("num_labels", 0) in (0, None):
        cfg["model"]["num_labels"] = len(label2id)

    train_ds = GondiPosDataset(train_ex, tokenizer, data_cfg["max_length"])
    dev_ds = GondiPosDataset(dev_ex, tokenizer, data_cfg["max_length"])
    test_ds = GondiPosDataset(test_ex, tokenizer, data_cfg["max_length"])

    return tokenizer, train_ds, dev_ds, test_ds, label2id, id2label


def build_xlmr_encoder(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_name = cfg["model"]["pretrained_name"]
    num_labels = cfg["model"]["num_labels"]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    encoder = AutoModel.from_pretrained(model_name, config=hf_config)
    return encoder


def build_full_model(
    cfg: Dict[str, Any],
    use_adapters: bool,
    use_router: bool,
    use_similarity_bias: bool,
    use_lexicon_loss: bool,
    label2id: Dict[str, int],
) -> torch.nn.Module:
    """
    Construct the combined model based on experiment flags.
    """
    class GondiPOSModel(torch.nn.Module):
        def __init__(
            self,
            cfg,
            use_adapters,
            use_router,
            use_similarity_bias,
            use_lexicon_loss,
        ):
            super().__init__()
            self.cfg = cfg
            self.use_adapters = use_adapters
            self.use_router = use_router
            self.use_similarity_bias = use_similarity_bias
            self.use_lexicon_loss = use_lexicon_loss
            self.encoder = build_xlmr_encoder(cfg)
            hidden_size = self.encoder.config.hidden_size
            num_labels = cfg["model"]["num_labels"]

            # Language adapters
            self.adapters = None
            self.router = None
            if use_adapters:
                self.adapters = MultiLanguageAdapters(
                    hidden_size=hidden_size,
                    bottleneck_size=cfg["model"]["adapter_hidden_size"],
                    languages=["hi", "mr", "te"],
                )
                if use_router:
                    bias_vec = torch.tensor(
                        cfg["router"]["similarity_bias"],
                        dtype=torch.float,
                    )
                    self.router = AdapterRouter(
                        hidden_size=hidden_size,
                        languages=["hi", "mr", "te"],
                        similarity_bias=bias_vec,
                        use_similarity_bias=use_similarity_bias,
                    )
            self.classifier = POSClassificationHead(
                hidden_size=hidden_size,
                num_labels=num_labels,
                dropout=cfg["model"]["dropout"],
            )

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            words=None,
            lambda_lex: float = 0.0,
            label2id: Optional[Dict[str, int]] = None,
            lexicon: Optional[Dict[str, str]] = None,
            return_router_weights: bool = False,
        ):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

            if self.use_adapters and self.adapters is not None:
                adapter_outputs = self.adapters(hidden)
                if self.use_router and self.router is not None:
                    # Pool using first token for routing decision
                    pooled = hidden[:, 0, :]
                    weights = self.router(pooled)
                    hidden = self.router.combine_adapters(adapter_outputs, weights)
                else:
                    # Static equal weights across adapters
                    num_langs = len(adapter_outputs)
                    stacked = torch.stack(list(adapter_outputs.values()), dim=0)
                    hidden = stacked.mean(dim=0)
                    weights = None
            else:
                weights = None

            logits = self.classifier(hidden)  # [batch, num_labels]

            loss = None
            ce_loss = None
            lex_loss = None
            if labels is not None:
                ce_loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = ce_loss
                if self.use_lexicon_loss and words is not None and lexicon is not None and label2id is not None:
                    lex_loss = lexicon_guided_loss(
                        words=words,
                        logits=logits,
                        label2id=label2id,
                        lexicon=lexicon,
                        lambda_weight=lambda_lex,
                    )
                    loss = loss + lex_loss

            return {
                "loss": loss,
                "logits": logits,
                "ce_loss": ce_loss,
                "lex_loss": lex_loss,
                "router_weights": weights if return_router_weights else None,
            }

    return GondiPOSModel(
        cfg=cfg,
        use_adapters=use_adapters,
        use_router=use_router,
        use_similarity_bias=use_similarity_bias,
        use_lexicon_loss=use_lexicon_loss,
    )


def create_dataloaders(
    train_ds,
    dev_ds,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        keys = batch[0].keys()
        for k in keys:
            if k == "word":
                out[k] = [b[k] for b in batch]
            else:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, dev_loader


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    num_training_steps: int,
):
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(num_training_steps * cfg["training"]["warmup_ratio"]),
        num_training_steps=num_training_steps,
    )
    return optim, scheduler


def get_device(cfg: Dict[str, Any]) -> torch.device:
    if cfg["device"]["use_gpu"] and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    cfg: Dict[str, Any],
    experiment_name: str,
):
    out_dir = os.path.join(cfg["experiment"]["output_dir"], experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    return ckpt_path


def save_metrics_json(cfg: Dict[str, Any], experiment_name: str, metrics: Dict[str, Any], filename: str) -> str:
    out_dir = os.path.join(cfg["experiment"]["output_dir"], experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path

