#!/usr/bin/env python3
"""远苍 SFT 训练脚本 (Transformers Trainer)
修复要点:
  1. labels 必须 深拷贝 (此前 .copy() 浅拷贝导致 input_ids 被 -100 污染 -> CUDA assert)
  2. 动态 padding collator (修复变长序列 stack 报错)
  3. loss masking: 仅 assistant 回复参与损失
  4. adamw_bnb_8bit 优化器 (fp32 AdamW 需 ~56GB, 8bit 仅 ~14GB, 适配当前 ~59GB 空闲显存)
用法:
  python3 train.py --data train_100.jsonl --max-steps 10 --output saves/test   # 冒烟测试 (正式训练不传--max-steps即按epoch长跑)
  python3 train.py --data train.jsonl   --epochs 1    --output saves/full      # 正式训练
"""
import os, argparse, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")   # Blackwell驱动595.84的PCIe P2P有bug,禁用后走host中转
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

MODEL = "/home/dja/桌面/远苍/Qwen2.5-Math-7B"

MAX_LEN = 2048

def encode_batch(batch):
    """模块级batched编码(供datasets.map多进程使用, Arrow磁盘缓存)"""
    tok = get_tok()
    input_ids_list, labels_list = [], []
    for inst, outp in zip(batch["instruction"], batch["output"]):
        inst = str(inst or "").strip(); outp = str(outp or "").strip()
        if not inst or not outp:
            input_ids_list.append([]); labels_list.append([]); continue
        prompt = f"user\n{inst}\nassistant\n"
        full_ids = tok(prompt + outp, truncation=True, max_length=MAX_LEN)["input_ids"]
        prompt_ids = tok(prompt, truncation=True, max_length=MAX_LEN)["input_ids"]
        labels = list(full_ids)
        cut = min(len(prompt_ids), len(labels))
        labels[:cut] = [-100] * cut
        input_ids_list.append(full_ids); labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}

_TOK = None
def get_tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL)
        if _TOK.pad_token is None: _TOK.pad_token = _TOK.eos_token
    return _TOK

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/dja/桌面/远苍/train_100.jsonl")
    ap.add_argument("--output", default="/home/dja/桌面/远苍/saves/test")
    ap.add_argument("--max-steps", type=int, default=-1, help="-1=按epochs训练")
    ap.add_argument("--epochs", type=float, default=-1.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval-data", default="/home/dja/桌面/远苍/eval.jsonl")
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="从output_dir最新checkpoint续训")
    ap.add_argument("--fp8", action="store_true", help="(已弃用,torchao FP8与Qwen2 view不兼容)")
    ap.add_argument("--precision", default="bf16", choices=["bf16","nvfp4"], help="NVFP4=85成层NVFP4训练+BF16尾层")
    ap.add_argument("--optim", default="adamw_torch", help="adamw_torch|adamw_bnb_8bit 等")
    ap.add_argument("--deepspeed", action="store_true", help="启用DeepSpeed ZeRO-2")
    return ap.parse_args()

ARGS = parse()

class Collator:
    """动态padding: input_ids补pad, labels补-100, 生成attention_mask"""
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        ids, labels, attn = [], [], []
        for f in features:
            x, y = f["input_ids"], f["labels"]
            pad = maxlen - len(x)
            ids.append(x + [self.pad_id] * pad)
            labels.append(y + [-100] * pad)
            attn.append([1] * len(x) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

class DS(torch.utils.data.Dataset):
    """包装Arrow内存映射dataset, 零拷贝"""
    def __init__(self, hf_ds): self.ds = hf_ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[i]
        return {"input_ids": r["input_ids"], "labels": r["labels"]}

def main():
    print(f"[1/5] tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[2/5] dataset: {ARGS.data}")
    ds = load_dataset("json", data_files=ARGS.data)["train"]
    print(f"      samples = {len(ds)}")

    def encode(inst, outp):
        prompt = f"user\n{inst}\nassistant\n"
        full = prompt + outp
        full_ids = tok(full, truncation=True, max_length=ARGS.max_len)["input_ids"]
        prompt_ids = tok(prompt, truncation=True, max_length=ARGS.max_len)["input_ids"]
        labels = list(full_ids)
        cut = min(len(prompt_ids), len(labels))
        labels[:cut] = [-100] * cut
        return full_ids, labels

    globals()["MAX_LEN"] = ARGS.max_len
    print(f"[3/5] tokenize (Arrow磁盘缓存, 多进程) ...")
    cols = [x for x in ds.column_names if x not in ("instruction","output")]
    ds = ds.map(encode_batch, batched=True, batch_size=256, num_proc=16,
                remove_columns=cols, desc="tokenizing")
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0, num_proc=8)
    print(f"      tokenized = {len(ds)}")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"[4/5] model (bf16, local_rank={local_rank}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if local_rank == -1:
        model = model.to("cuda:0")          # 单卡模式
    else:
        torch.cuda.set_device(local_rank)   # DDP: 每进程绑一张卡
        model = model.to(local_rank)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print(f"      params = {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if ARGS.precision == "nvfp4":
        import re
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        from torchao.quantization import quantize_
        n_l = model.config.num_hidden_layers
        cut = int(n_l * 0.85)          # 最后15%层(含末4层)保持BF16
        nq = 0
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name \
               and any(k in name for k in ("mlp.gate_proj","mlp.up_proj","mlp.down_proj")):
                m = re.search(r"layers\.(\d+)\.", name)
                if m and int(m.group(1)) < cut:
                    try:
                        quantize_(mod, NVFP4DynamicActivationNVFP4WeightConfig())
                        nq += 1
                    except Exception as e:
                        print(f"      NVFP4 skip {name}: {str(e)[:50]}")
        print(f"      NVFP4: {nq} linears quantized, layers>={cut} 保持BF16")

    eval_ds = None
    if os.path.exists(ARGS.eval_data):
        eds = load_dataset("json", data_files=ARGS.eval_data)["train"]
        globals()["MAX_LEN"] = ARGS.max_len
        ecols = [x for x in eds.column_names if x not in ("instruction","output")]
        eds = eds.map(encode_batch, batched=True, batch_size=256, num_proc=4,
                      remove_columns=ecols, desc="eval-tokenizing")
        eds = eds.filter(lambda ex: len(ex["input_ids"]) > 0)
        eval_ds = DS(eds)
        print(f"      eval(L1) = {len(eval_ds)} rows, every {ARGS.eval_steps} steps")

    targs = TrainingArguments(
        output_dir=ARGS.output,
        per_device_train_batch_size=ARGS.bsz,
        gradient_accumulation_steps=ARGS.accum,
        learning_rate=ARGS.lr,
        lr_scheduler_type="cosine",
        warmup_steps=ARGS.warmup,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=5,
        save_steps=200,
        save_total_limit=3,
        seed=42,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim=ARGS.optim,
        eval_strategy=("steps" if eval_ds else "no"),
        eval_steps=(ARGS.eval_steps if eval_ds else 500),
        per_device_eval_batch_size=ARGS.bsz,
        max_steps=(ARGS.max_steps if ARGS.max_steps > 0 else -1),
        num_train_epochs=(ARGS.epochs if ARGS.epochs > 0 else 1.0),
        dataloader_num_workers=0,
        report_to=[],
        deepspeed=("/home/dja/桌面/远苍/ds_z2.json" if ARGS.deepspeed else None),
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=DS(ds), eval_dataset=eval_ds, data_collator=Collator(tok.pad_token_id),
    )

    world = max(1, torch.cuda.device_count() if local_rank == -1 else int(os.environ.get("WORLD_SIZE", 1)))
    print(f"[5/5] training ... global_batch = {ARGS.bsz} x {ARGS.accum} x {world} = {ARGS.bsz*ARGS.accum*world}")
    trainer.train(resume_from_checkpoint=(True if ARGS.resume else None))
    trainer.save_model(ARGS.output)
    tok.save_pretrained(ARGS.output)
    print("DONE")

if __name__ == "__main__":
    main()
