#!/usr/bin/env python3
"""远苍 train_nvfp4.py (QAT NVFP4 正式训练, 含 BF16 保底回退)

用法:
  双卡正式(NVFP4 QAT): torchrun --nproc_per_node=2 train_nvfp4.py \
        --mode full --data "train_part*.jsonl" [--quant nvfp4|bf16]

说明:
- NVFP4: 用 torchao QAT fake-quant (NVFP4DynamicActivationNVFP4WeightConfig ->
  NVFP4FakeQuantizedLinear), Blackwell Tensor Core 低精度前向 + 高精度反向。
  需 real GPU(Blackwell sm_120)+Triton; 若环境/硬件不支持 -> 自动回退 BF16 FSDP。
- --quant bf16 : 强制走 BF16 保底主干 (等同 train.py)。
- 复用 train.py 的 tokenize/数据集/collator/TrainingArguments(FSDP)逻辑,保证两轨一致。

注意: prepare_data.py / train.py 均为 seed=42 可复现。
"""
import os, sys, time, argparse

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")

import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# ---- reuse core from train.py (identical data/tokenize/FSDP-config) ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    MODEL_PATH,
    resolve_data_files,
    build_dataset,
    build_eval_dataset,
    make_training_args,
)

try:
    # NVFP4 QAT imports (optional; absent -> bf16 fallback)
    from torchao.quantization.qat import QATConfig
    from torchao.prototype.mx_formats import (
        NVFP4DynamicActivationNVFP4WeightConfig,
    )
    from torchao import quantize_
    _QAT_AVAILABLE = True
except Exception as e:                      # pragma: no cover
    print(f"WARN QAT libs unavailable ({e}); will use BF16 fallback")
    _QAT_AVAILABLE = False


def apply_nvfp4_qat(model):
    """Swap Linear->NVFP4FakeQuantizedLinear. Returns model."""
    cfg = NVFP4DynamicActivationNVFP4WeightConfig()
    quantize_(model, QATConfig(cfg, step="prepare"))
    nq = sum(1 for m in model.modules() if type(m).__name__.startswith("NVFP4FakeQuantizedLinear"))
    print(f"[QAT] NVFP4 fake-quant applied: {nq} Linear modules swapped", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "full"], default="full")
    ap.add_argument("--data", required=True, help="jsonl path(s), 逗号分隔或glob")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--outdir", default="/home/dja/桌面/远苍/saves/nvfp4-sft")
    ap.add_argument("--quant", choices=["nvfp4", "bf16"], default="nvfp4",
                    help="nvfp4(默认,不可用则回退bf16) / bf16(强制保底)")
    ap.add_argument("--max_seq_len", type=int, default=None)
    args = ap.parse_args()

    full_mode = args.mode == "full"
    files = resolve_data_files(args.data)
    if not files:
        print("No data files:", args.data); sys.exit(1)

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    try:
        print("Loading base model (sdpa)...", flush=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
        print(f"Base loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    except Exception as e:
        print("FATAL model load:", e); sys.exit(2)

    quant_mode = args.quant
    if quant_mode == "nvfp4":
        if not _QAT_AVAILABLE or not torch.cuda.is_available():
            print("[WARN] NVFP4 QAT unavailable -> falling back to BF16", flush=True)
            quant_mode = "bf16"
        else:
            try:
                apply_nvfp4_qat(model)   # may raise on non-Blackwell/CPU
            except Exception as e:
                print(f"[WARN] NVFP4 QAT init failed ({type(e).__name__}: {str(e)[:120]}) "
                      f"-> BF16 fallback", flush=True)
                # reload clean model for bf16 path
                from transformers import AutoModelForCausalLM
                del model; torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
                quant_mode = "bf16"
    print(f"[RESOLVED] running in {quant_mode.upper()} mode", flush=True)

    max_seq_len = args.max_seq_len or (8192 if full_mode else 1024)
    train_ds = build_dataset(tokenizer, files, max_seq_len)
    eval_ds = build_eval_dataset(
        tokenizer,
        max_seq_len=(max_seq_len if full_mode else None))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                      pad_to_multiple_of=8)

    # FSDP FULL_SHARD + BF16; global batch 32 -> grad_accum auto-derived (4 on 2 GPU)
    targs = make_training_args(
        mode=args.mode, outdir=args.outdir, n_gpus=max(1, args.gpus),
        per_device_batch=(4 if full_mode else 4),
        global_batch=(32 if full_mode else None),
        max_seq_len=max_seq_len,
        eval_dataset_size=(len(eval_ds) if eval_ds is not None else None))
    targs.output_dir = args.outdir
    targs.bf16 = True      # NVFP4 QAT trains in bf16 autocast, weights fake-quantized forward
    targs.fp16 = False

    from transformers import Trainer, EarlyStoppingCallback
    callbacks = []
    if eval_ds is not None and len(eval_ds) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3,
                                               early_stopping_threshold=1e-4))
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds,
                      data_collator=collator, callbacks=callbacks)
    print(f"\n=== Starting training (quant={quant_mode}, mode={args.mode}) ===", flush=True)
    ckpt = _find_last_checkpoint(args.outdir) if full_mode else None
    trainer.train(resume_from_checkpoint=ckpt)

    final_dir = os.path.join(args.outdir, "final")
    trainer.save_model(final_dir); tokenizer.save_pretrained(final_dir)
    print("Training completed!")


def _find_last_checkpoint(outdir):
    import glob, re
    hits = sorted(glob.glob(os.path.join(outdir, "checkpoint-*")),
                  key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1) or 0))
    return hits[-1] if hits else None


if __name__ == "__main__":
    main()
