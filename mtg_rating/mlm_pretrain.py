"""MLM pretraining stage: adapt MiniLM to MTG vocabulary via a larger-rank LoRA
(rank 32, vs the rank-4 used for the actual rating fine-tune), trained on masked
language modeling over the full legacy/modern/standard-legal card pool (~30k
cards, see fetch_scryfall.fetch_bulk_oracle_cards) -- not the small ~5-6k labeled
rating dataset.

Rationale for the larger rank here specifically: this stage has ~5-6x more data
than the rating fine-tune, so more adapter capacity is affordable without the
overfitting risk that a bigger rank showed on the (much smaller) rating task
(rank 8 vs 4 there). Full fine-tuning of all 22.7M base params was considered and
rejected (see project discussion) -- the corpus, while much bigger than the
rating dataset, is still small relative to MiniLM's original pretraining corpus,
so even here a modest-rank LoRA is safer than unrestricted fine-tuning.

Unlike LoraTextEncoder (AutoModel, frozen everywhere except LoRA), this uses
AutoModelForMaskedLM -- the sentence-transformers checkpoint doesn't ship the
original MLM prediction head weights (stripped since it's not needed for sentence
embeddings), so `cls.predictions.*` loads randomly initialized and needs to be
fully trainable (via peft's `modules_to_save`), not just LoRA-adapted like the
attention layers.

After training, merge_and_unload() bakes the LoRA deltas into the base encoder
weights, producing a standalone "MTG-adapted MiniLM" saved to MLM_OUTPUT_DIR --
meant to replace the generic `sentence-transformers/all-MiniLM-L6-v2` as the
starting point for a fresh (small-rank) LoraTextEncoder in the rating fine-tune.

Run with: conda run -n env_coinche python -m mtg_rating.mlm_pretrain
"""

import copy
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from mtg_rating.fetch_scryfall import fetch_bulk_oracle_cards
from mtg_rating.text_embeddings import MODEL_NAME

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "minilm_mtg_pretrained"
LORA_RANK = 32
LORA_LR = 2e-4
HEAD_LR = 1e-3
VAL_FRACTION = 0.1
SPLIT_SEED = 42
# 64 OOM'd on the 6GB GTX 1660 Super (MLM computes vocab-sized logits, 30522-wide,
# for every token position -- much heavier per-example than the rating task's
# single pooled 384-dim output). 32 fit (~3GB peak) but wasn't actually faster
# per-epoch than 16 (~188s vs ~169s, measured) since this GPU/model is too small
# for bigger batches to pay off here -- 16 is both safer and slightly quicker.
BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 5
MLM_PROBABILITY = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_texts(texts: list, val_fraction: float, seed: int):
    shuffled = texts[:]
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def build_model(tokenizer, full_finetune: bool = False, lora_rank: int = LORA_RANK):
    base_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    if full_finetune:
        # No peft wrapping at all -- every one of the 34.9M params (base
        # encoder + the freshly-initialized MLM head) is trainable. Comparison
        # point only, per project discussion: full fine-tuning on this
        # (relatively small, ~1-1.5M token) corpus was rejected as the default
        # approach due to overfitting/catastrophic-forgetting risk, but a
        # capped (<=7 epoch) run costs about the same as the LoRA runs (same
        # forward/backward graph either way) and directly tests that risk
        # empirically instead of just arguing it.
        n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in base_model.parameters())
        print(f"trainable params: {n_trainable:,} || all params: {n_total:,} || trainable%: {100*n_trainable/n_total:.4f}")
        return base_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        target_modules=["query", "value"],
        modules_to_save=["cls"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def iter_batches(texts: list, batch_size: int):
    for start in range(0, len(texts), batch_size):
        yield texts[start : start + batch_size]


def run_epoch(model, tokenizer, collator, texts: list, batch_size: int, train: bool, optimizer=None):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch_texts in iter_batches(texts, batch_size):
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        batch = collator([{k: v[i] for k, v in encoded.items()} for i in range(len(batch_texts))])
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.set_grad_enabled(train):
            outputs = model(**batch)
            loss = outputs.loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        n_masked = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * n_masked
        total_tokens += n_masked
    return total_loss / total_tokens


def save_pretrained_checkpoint(model, tokenizer, output_dir: Path, full_finetune: bool = False):
    # merge_and_unload() mutates the underlying layers in place and strips the
    # peft wrapper -- done on a deepcopy so mid-training snapshots don't disturb
    # the live model still being trained. full_finetune has no peft wrapper to
    # merge/strip, so the deepcopy is saved directly.
    snapshot = copy.deepcopy(model)
    bert = snapshot.bert if full_finetune else snapshot.merge_and_unload().bert
    output_dir.mkdir(parents=True, exist_ok=True)
    bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[checkpoint] saved to {output_dir}")


def main(
    seed: int = 0,
    epochs: int = EPOCHS,
    checkpoint_every: int = 0,
    checkpoint_root: Path = None,
    full_finetune: bool = False,
    output_dir: Path = None,
    lora_rank: int = LORA_RANK,
):
    torch.manual_seed(seed)
    output_dir = Path(output_dir or OUTPUT_DIR)  # accept str or Path
    if checkpoint_every:
        checkpoint_root = Path(checkpoint_root or OUTPUT_DIR.parent / "minilm_mtg_pretrained_checkpoints")

    cards = fetch_bulk_oracle_cards()
    texts = [c["oracle_text"] for c in cards]
    train_texts, val_texts = split_texts(texts, VAL_FRACTION, SPLIT_SEED)
    print(f"[split] {len(train_texts)} train texts, {len(val_texts)} val texts")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY)

    model = build_model(tokenizer, full_finetune=full_finetune, lora_rank=lora_rank).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Same differential-LR reasoning as the rating fine-tune (lower for the
    # LoRA-adapted attention weights, higher for the freshly-initialized MLM
    # head) -- approximated here with a single group since peft's
    # modules_to_save/LoRA params aren't as easily split into two named groups
    # without walking named_parameters(); revisit if this turns out to matter.
    optimizer = torch.optim.Adam(trainable, lr=LORA_LR)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    rng = random.Random(seed)

    for epoch in range(epochs):
        shuffled_train = train_texts[:]
        rng.shuffle(shuffled_train)
        train_loss = run_epoch(model, tokenizer, collator, shuffled_train, BATCH_SIZE, train=True, optimizer=optimizer)
        val_loss = run_epoch(model, tokenizer, collator, val_texts, BATCH_SIZE, train=False)
        print(f"[epoch {epoch}] train loss={train_loss:.3f}  val loss={val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[early stop] no val improvement for {PATIENCE} epochs, stopping at epoch {epoch}")
                break

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_pretrained_checkpoint(model, tokenizer, checkpoint_root / f"epoch{epoch + 1}", full_finetune=full_finetune)

    model.load_state_dict(best_state)

    # bert is a BertForMaskedLM (children: "bert", "cls") either way -- only the
    # "bert" core encoder is saved, so this loads as a drop-in AutoModel
    # replacement in a fresh LoraTextEncoder later; the MLM head ("cls") isn't
    # needed downstream. full_finetune has no peft wrapper to merge/strip.
    bert = model.bert if full_finetune else model.merge_and_unload().bert
    output_dir.mkdir(parents=True, exist_ok=True)
    bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[saved] MTG-adapted MiniLM base saved to {output_dir}")
    return best_val_loss


if __name__ == "__main__":
    main()
