"""★ M8-15: LoRA SFT 最小训练循环（HF Trainer + peft）。

设计要点：
  - 限定 Qwen2.5 系基座（可用 OMNIMEM_ALLOW_ANY_BASE=1 放开）
  - 优先尝试 QLoRA 4bit（bitsandbytes 可用时）,否则回退 bf16/fp32 LoRA + 梯度检查点
  - 产物（adapter_config.json / adapter_model.safetensors）保存到 output_dir

本模块只在 _real_train 内部延迟导入,不影响无训练依赖环境。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = "{instruction}\n{input}\n"


def _resolve_base_model(base_model: str) -> str:
    """规范化基座模型 ID：裸名补 Qwen/ 前缀,并校验 Qwen2.5 系限定。"""
    name = base_model.strip()
    if "/" not in name and not Path(name).exists():
        name = f"Qwen/{name}"
    if "qwen2.5" not in name.lower().replace("-", ".").replace("_", "."):
        if os.environ.get("OMNIMEM_ALLOW_ANY_BASE", "") != "1":
            raise ValueError(
                f"base_model {base_model!r} not in Qwen2.5 family (set OMNIMEM_ALLOW_ANY_BASE=1 to override)"
            )
    return name


def _build_dataset(samples: list[dict[str, str]], tokenizer, max_len: int = 512):
    """把 alpaca 样本编码为带 labels 的 torch Dataset（prompt 部分掩掉损失）。"""
    import torch

    class _SFTDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self._items = []
            for s in samples:
                prompt = _PROMPT_TEMPLATE.format(
                    instruction=s.get("instruction", ""), input=s.get("input", "")
                )
                full = prompt + s.get("output", "") + (tokenizer.eos_token or "")
                enc = tokenizer(full, truncation=True, max_length=max_len)
                prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
                labels = list(enc["input_ids"])
                labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
                self._items.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": labels,
                })

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> dict:
            return self._items[idx]

    return _SFTDataset()


def _load_model(model_id: str):
    """加载基座:优先 QLoRA 4bit（bnb 可用）,否则 bf16/fp32 + 梯度检查点。"""
    import torch
    from transformers import AutoModelForCausalLM

    quantized = False
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto"
            )
            quantized = True
            logger.info("train_loop: QLoRA 4bit enabled (bitsandbytes)")
            return model, quantized
        except Exception as e:
            logger.info("train_loop: bitsandbytes unavailable (%s), fallback to LoRA", e)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.gradient_checkpointing_enable()
    return model, quantized


def run_sft(
    base_model: str,
    samples: list[dict[str, str]],
    output_dir: str | Path,
    *,
    epochs: int = 3,
    lr: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
    max_len: int = 512,
) -> dict[str, Any]:
    """执行最小 LoRA SFT 训练,返回训练摘要。产物保存在 output_dir。"""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model_id = _resolve_base_model(base_model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, quantized = _load_model(model_id)
    if quantized:
        model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False

    dataset = _build_dataset(samples, tokenizer, max_len=max_len)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    train_result = trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return {
        "base_model": model_id,
        "quantized_4bit": quantized,
        "samples": len(samples),
        "epochs": epochs,
        "train_loss": round(float(train_result.training_loss), 4),
        "peak_vram_gb": peak_mem_gb,
        "adapter_path": str(output_dir),
    }
