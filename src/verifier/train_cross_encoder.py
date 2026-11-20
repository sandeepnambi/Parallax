"""
Parallax Cross-Encoder Fine-Tuning Module
=========================================
Fine-tunes the natural language inference (NLI) cross-encoder on domain-specific
premise-hypothesis pairs from the Parallax training split (data/benchmark/train.parquet).

Maps the 4-class benchmark taxonomy to 3-class semantic inference:
- SUPPORTED_CURRENT -> ENTAILMENT (id: 1)
- OUTDATED          -> ENTAILMENT (id: 1, entailed by context; temporal arbiter handles freshness)
- CONTRADICTED      -> CONTRADICTION (id: 0)
- UNSUPPORTED       -> NEUTRAL (id: 2)

Evaluates on validation data (data/benchmark/dev.parquet) at every epoch, saves the best
checkpoint, and benchmarks on test data (data/benchmark/test.parquet).
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# Ensure project root is in path
_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_cur, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.evaluation.metrics import compute_parallax_metrics  # noqa: E402


# 3-Class NLI Label Mappings matching cross-encoder/nli-distilroberta-base
LABEL2ID = {"CONTRADICTION": 0, "ENTAILMENT": 1, "NEUTRAL": 2}
ID2LABEL = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}

TAXONOMY_TO_NLI = {
    "SUPPORTED_CURRENT": "ENTAILMENT",
    "OUTDATED": "ENTAILMENT",
    "CONTRADICTED": "CONTRADICTION",
    "UNSUPPORTED": "NEUTRAL",
}


class ParallaxNLIDataset(Dataset):
    """PyTorch Dataset for Premise-Hypothesis NLI fine-tuning."""
    def __init__(self, df: pd.DataFrame):
        self.premises = [str(t) for t in df["evidence_text"]]
        self.hypotheses = [str(t) for t in df["answer_claim"]]
        self.gold_labels = df["gold_label"].tolist()
        self.nli_labels = [
            LABEL2ID[TAXONOMY_TO_NLI.get(gl, "NEUTRAL")] for gl in self.gold_labels
        ]

    def __len__(self) -> int:
        return len(self.premises)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "premise": self.premises[idx],
            "hypothesis": self.hypotheses[idx],
            "label": self.nli_labels[idx],
            "gold_taxonomy": self.gold_labels[idx],
        }


class DynamicPaddingCollator:
    """Collates and dynamically pads batches to the maximum sequence length in the batch."""
    def __init__(self, tokenizer: Any, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        premises = [b["premise"] for b in batch]
        hypotheses = [b["hypothesis"] for b in batch]
        labels = [b["label"] for b in batch]

        encoded = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


def evaluate_model(
    model: Any,
    dev_loader: DataLoader,
    dev_df: pd.DataFrame,
    device: str,
) -> Tuple[float, float, Dict[str, Any]]:
    """Evaluates the model on validation data with both 3-class NLI and 4-class Parallax metrics."""
    model.eval()
    all_nli_preds: List[int] = []
    
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_nli_preds.extend(preds)

    # 4-Class Parallax Two-Stage Temporal Arbitration
    parallax_preds: List[str] = []
    for i, pred_id in enumerate(all_nli_preds):
        nli_rel = ID2LABEL[pred_id]
        row = dev_df.iloc[i]
        ref_time = str(row.get("reference_time", "2024-12-01"))
        valid_to = str(row.get("valid_to", "present"))
        superseded_by = row.get("superseded_by")

        if nli_rel == "ENTAILMENT":
            is_superseded = False
            if valid_to != "present" and valid_to <= ref_time:
                is_superseded = True
            elif superseded_by is not None and str(superseded_by) <= ref_time:
                is_superseded = True

            if is_superseded:
                parallax_preds.append("OUTDATED")
            else:
                parallax_preds.append("SUPPORTED_CURRENT")
        elif nli_rel == "CONTRADICTION":
            parallax_preds.append("CONTRADICTED")
        else:
            parallax_preds.append("UNSUPPORTED")

    gold_taxonomy = dev_df["gold_label"].tolist()
    metrics = compute_parallax_metrics(gold_taxonomy, parallax_preds)
    return metrics["accuracy"], metrics["macro_f1"], metrics


def train_cross_encoder(
    train_path: str = "data/benchmark/train.parquet",
    dev_path: str = "data/benchmark/dev.parquet",
    test_path: str = "data/benchmark/test.parquet",
    base_model: str = "cross-encoder/nli-distilroberta-base",
    output_dir: str = "models/parallax_nli_finetuned",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 128,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Fine-tunes the NLI cross-encoder with dynamic batching, balanced class loss, and validation."""
    print("=" * 65)
    print("PARALLAX NEURAL CROSS-ENCODER FINE-TUNING")
    print("=" * 65)
    print(f"Base Model:     {base_model}")
    print(f"Output Dir:     {output_dir}")
    print(f"Training Split: {train_path}")
    print(f"Dev Split:      {dev_path}")
    print(f"Epochs:         {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"Compute Device: {device}")
    print("-" * 65)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    dev_df = pd.read_parquet(dev_path) if dev_path.endswith(".parquet") else pd.read_csv(dev_path)
    test_df = pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)

    print(f"Loaded {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test records.")

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model).to(device)

    # 3. DataLoaders
    collator = DynamicPaddingCollator(tokenizer, max_length=max_length)
    train_dataset = ParallaxNLIDataset(train_df)
    dev_dataset = ParallaxNLIDataset(dev_df)
    test_dataset = ParallaxNLIDataset(test_df)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    # 4. Balanced CrossEntropyLoss Weights (contradiction=400, entailment=800, neutral=400)
    # Effective weights inverse to frequencies: [1.0, 0.5, 1.0]
    class_weights = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 5. Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 6. Evaluate Baseline Before Fine-Tuning
    print("\nEvaluating Zero-Shot Pre-Trained Model on Dev Set...")
    init_acc, init_macro_f1, _ = evaluate_model(model, dev_loader, dev_df, device)
    print(f"Pre-Training Baseline -> Dev Accuracy: {init_acc*100:.2f}% | Dev Macro-F1: {init_macro_f1:.4f}")

    # 7. Training Loop
    history: List[Dict[str, Any]] = []
    best_macro_f1 = -1.0
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

            if step % 25 == 0 or step == len(train_loader):
                avg_l = total_loss / step_count
                lr_cur = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}/{epochs} | Step {step:3d}/{len(train_loader)} | Loss: {avg_l:.4f} | LR: {lr_cur:.2e}")

        # Evaluation on Dev
        dev_acc, dev_macro_f1, dev_metrics = evaluate_model(model, dev_loader, dev_df, device)
        epoch_time = time.time() - epoch_start
        print(f"--> Epoch {epoch} Complete ({epoch_time:.1f}s) | Dev Acc: {dev_acc*100:.2f}% | Dev Macro-F1: {dev_macro_f1:.4f} | FSR: {dev_metrics['fsr_outdated']*100:.2f}%")

        epoch_stats = {
            "epoch": epoch,
            "train_loss": total_loss / step_count,
            "dev_accuracy": dev_acc,
            "dev_macro_f1": dev_macro_f1,
            "dev_fsr_outdated": dev_metrics["fsr_outdated"],
            "epoch_duration_seconds": epoch_time,
        }
        history.append(epoch_stats)

        # Checkpoint Saving
        if dev_macro_f1 > best_macro_f1:
            best_macro_f1 = dev_macro_f1
            print(f"    [*] New best dev Macro-F1: {best_macro_f1:.4f}! Saving checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    train_duration = time.time() - total_start
    print("\n" + "=" * 65)
    print(f"Fine-Tuning Finished in {train_duration/60:.2f} minutes.")
    print(f"Best Validation Macro-F1: {best_macro_f1:.4f}")
    print("=" * 65)

    # 8. Load Best Checkpoint and Evaluate on Held-out Test Split
    print("\nEvaluating Best Fine-Tuned Checkpoint on Official Test Split (800 records)...")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_acc, test_macro_f1, test_metrics = evaluate_model(best_model, test_loader, test_df, device)

    print("\n" + "=" * 65)
    print("PARALLAX FINE-TUNED NEURAL VERIFIER TEST SPLIT RESULTS")
    print("=" * 65)
    print(f"Test Accuracy:           {test_acc*100:.2f}%")
    print(f"Test Macro-F1:          {test_macro_f1:.4f}")
    print(f"False-Support Rate:      {test_metrics['fsr_outdated']*100:.2f}%")
    print(f"SUPPORTED_CURRENT F1:   {test_metrics['per_class']['SUPPORTED_CURRENT']['f1']*100:.2f}%")
    print(f"OUTDATED F1:            {test_metrics['per_class']['OUTDATED']['f1']*100:.2f}%")
    print(f"CONTRADICTED F1:        {test_metrics['per_class']['CONTRADICTED']['f1']*100:.2f}%")
    print(f"UNSUPPORTED F1:         {test_metrics['per_class']['UNSUPPORTED']['f1']*100:.2f}%")
    print("Confusion Matrix (Rows=True [Supp, Outd, Contra, Unsupp], Cols=Pred):")
    for row in test_metrics["confusion_matrix"]:
        print(f"  {row}")
    print("=" * 65)

    results = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "total_training_duration_seconds": train_duration,
        "best_dev_macro_f1": best_macro_f1,
        "test_results": test_metrics,
        "training_history": history,
    }

    # Save training report
    os.makedirs("experiments", exist_ok=True)
    report_path = "experiments/cross_encoder_finetune_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full fine-tuning report to {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parallax Cross-Encoder Fine-Tuning")
    parser.add_argument("--train_path", type=str, default="data/benchmark/train.parquet")
    parser.add_argument("--dev_path", type=str, default="data/benchmark/dev.parquet")
    parser.add_argument("--test_path", type=str, default="data/benchmark/test.parquet")
    parser.add_argument("--base_model", type=str, default="cross-encoder/nli-distilroberta-base")
    parser.add_argument("--output_dir", type=str, default="models/parallax_nli_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_cross_encoder(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
