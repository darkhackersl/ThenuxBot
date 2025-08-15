import os, json, argparse
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

CHAT_FORMAT_SYS = "You are a helpful, concise assistant named ThenuxBot."
def format_turns(turns: List[Dict[str,str]]) -> str:
    s = [f"<|system|>\n{CHAT_FORMAT_SYS}"]
    for t in turns:
        role = t["role"]
        content = t["content"].strip()
        s.append(f"<|{role}|>\n{content}")
    return "\n".join(s) + "\n<|assistant|>\n"

def load_dataset(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                turns = obj["turns"]
                texts.append(format_turns(turns))
    return texts

class TxtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self): return len(self.encodings["input_ids"])
    def __getitem__(self, idx): 
        return {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample.jsonl")
    ap.add_argument("--out", default="model")
    ap.add_argument("--base", default="distilgpt2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_dataset(args.data)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512)
    ds = TxtDataset(enc)

    model = AutoModelForCausalLM.from_pretrained(args.base)
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        learning_rate=5e-5,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print("Saved fine-tuned model to", args.out)

if __name__ == "__main__":
