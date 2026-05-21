from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel


class TransformerClassificationModel(nn.Module):
    """Generic HuggingFace transformer backbone with a classification head."""

    def __init__(
        self,
        base_transformer_model: str | PreTrainedModel,
        n_classes: int,
        dropout: float = 0.25,
        freeze_backbone: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__()

        if isinstance(base_transformer_model, str):
            self.model_name = base_transformer_model
            self.tokenizer = AutoTokenizer.from_pretrained(base_transformer_model)
            config_kwargs = {}
            if attn_implementation is not None:
                config_kwargs["attn_implementation"] = attn_implementation

            config = AutoConfig.from_pretrained(
                base_transformer_model,
                output_attentions=attn_implementation is not None,
                **config_kwargs,
            )
            self.backbone = AutoModel.from_pretrained(base_transformer_model, config=config)
        else:
            self.model_name = base_transformer_model.__class__.__name__
            self.backbone = base_transformer_model
            self.tokenizer = None

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

        if freeze_backbone:
            self.freeze_backbone(True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_attentions=False):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(self.dropout(pooled))

        if output_attentions:
            return logits, outputs.attentions
        return logits
