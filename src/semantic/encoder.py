"""Semantic Encoder — Qwen2.5-Coder + optional LoRA for 3B on Kaggle 2xT4.

Architecture:
    Code String -> Tokenizer -> Transformer Backbone (+LoRA) -> [CLS]/mean pooling -> Projection -> Embedding

LoRA recipe for Qwen2.5-Coder-3B on 2x T4 16GB (FP16, grad ckpt, bs1 len2048):
    r=32 alpha=64 dropout=0.05 target=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
    lr 2e-4 (10x full-finetune), eff batch 32, ~12GB/GPU -> FIT. Full-finetune 3B ~19GB -> OOM.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        output_dim: int = 256,
        freeze_layers: int = 28,
        dropout: float = 0.1,
        pooling: str = "mean",
        gradient_checkpointing: bool = False,
        use_fp16: bool = False,
        use_lora: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        use_rslora: bool = True,
        lora_target_modules: Optional[list[str]] = None,
        **_extra_kwargs,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.output_dim = output_dim
        self.use_lora = use_lora

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install `transformers`: pip install transformers") from exc

        logger.info("Loading semantic backbone: %s (lora=%s)", backbone, use_lora)
        # FP16 for 3B saves ~3GB vs bf16/fp32; LoRA FP16 is fastest on T4 (no 4bit dequant)
        dtype = torch.float16 if use_fp16 else None
        kwargs = {"trust_remote_code": True}
        if dtype is not None:
            # pyrefly: ignore [bad-assignment]
            kwargs["torch_dtype"] = dtype
        
        # Load directly to GPU 0 if available to avoid CPU RAM OOM on Kaggle
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["low_cpu_mem_usage"] = False

        self.backbone = AutoModel.from_pretrained(backbone, **kwargs)
        # Remove accelerate dispatch hooks so DataParallel can replicate backbone across multiple GPUs
        if torch.cuda.is_available():
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(self.backbone)
            except Exception as e:
                logger.debug("remove_hook_from_submodules skipped: %s", e)

        self.hidden_size = self.backbone.config.hidden_size
        logger.info("Backbone hidden size: %d", self.hidden_size)

        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable()
                # required when grad ckpt is on
                if hasattr(self.backbone, "config"):
                    self.backbone.config.use_cache = False
                logger.info("Gradient checkpointing ENABLED (use_cache=False)")
            except Exception as e:
                logger.warning("gradient_checkpointing_enable failed: %s", e)

        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, use_rslora, lora_target_modules)
        else:
            self._freeze_layers(freeze_layers)

        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def _freeze_layers(self, n: int) -> None:
        if hasattr(self.backbone, "embed_tokens"):
            for param in self.backbone.embed_tokens.parameters():
                param.requires_grad = False
        layers = None
        if hasattr(self.backbone, "layers"):
            layers = self.backbone.layers
        elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer
        if layers is not None:
            total = len(layers)
            frozen = min(n, total)
            for i, layer in enumerate(layers):
                if i < frozen:
                    for param in layer.parameters():
                        param.requires_grad = False
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            logger.info("Froze %d/%d layers. Trainable: %s / %s (%.1f%%)", frozen, total, f"{trainable:,}", f"{total_params:,}", 100.0 * trainable / total_params if total_params else 0)

    def _apply_lora(self, r: int, alpha: int, dropout: float, use_rslora: bool, target_modules: Optional[list[str]]) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA requires `peft`: pip install peft>=0.11.0") from exc
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Freeze base first (peft will re-enable LoRA adapters)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Some peft versions support use_rslora
        lora_kwargs: dict = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, bias="none", task_type="CAUSAL_LM")
        # try RsLoRA if available
        try:
            import inspect
            if "use_rslora" in inspect.signature(LoraConfig.__init__).parameters:
                lora_kwargs["use_rslora"] = use_rslora
        except Exception:
            pass
        peft_config = LoraConfig(**lora_kwargs)
        self.backbone = get_peft_model(self.backbone, peft_config)
        try:
            self.backbone.print_trainable_parameters()
        except Exception:
            pass
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        # count includes base frozen + lora + projection
        logger.info("LoRA applied r=%d alpha=%d dropout=%.2f target=%s RsLoRA=%s -> trainable %s / %s (%.2f%%)", r, alpha, dropout, target_modules, use_rslora, f"{trainable:,}", f"{total_params:,}", 100.0 * trainable / total_params if total_params else 0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_sequence: bool = False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        pooled_out = self.projection(pooled)
        if return_sequence:
            seq_out = self.projection(hidden_states)
            return pooled_out, seq_out
        return pooled_out

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
