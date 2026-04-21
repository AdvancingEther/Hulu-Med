import os

import torch

from .HuluMed_Qwen2 import HuluMed_Qwen2


def _read_env_bool(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _read_env_float(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _normalize_cdpruner_config(config=None):
    default_config = {
        "enable": False,
        "kept_ratio": 0.25,
        "alpha": 1.0,
    }
    if config is None:
        return dict(default_config)

    merged = dict(default_config)
    merged.update(config)
    merged["enable"] = bool(merged["enable"])
    merged["kept_ratio"] = float(merged["kept_ratio"])
    merged["alpha"] = float(merged["alpha"])
    return merged


def _resolve_cdpruner_config(args):
    cdpruner_config = {}

    env_enable = _read_env_bool("CDPRUNER_ENABLE")
    env_kept_ratio = _read_env_float("CDPRUNER_KEPT_RATIO")
    env_alpha = _read_env_float("CDPRUNER_ALPHA")

    arg_enable = getattr(args, "cdpruner_enable", None)
    arg_kept_ratio = getattr(args, "cdpruner_kept_ratio", None)
    arg_alpha = getattr(args, "cdpruner_alpha", None)

    if env_enable is not None:
        cdpruner_config["enable"] = env_enable
    if env_kept_ratio is not None:
        cdpruner_config["kept_ratio"] = env_kept_ratio
    if env_alpha is not None:
        cdpruner_config["alpha"] = env_alpha

    if arg_enable is not None:
        cdpruner_config["enable"] = arg_enable
    if arg_kept_ratio is not None:
        cdpruner_config["kept_ratio"] = arg_kept_ratio
    if arg_alpha is not None:
        cdpruner_config["alpha"] = arg_alpha

    return _normalize_cdpruner_config(cdpruner_config)


def _apply_cdpruner_config(model, cdpruner_config):
    model.config.cdpruner_config = dict(cdpruner_config)

    if hasattr(model, "get_model"):
        base_model = model.get_model()
        if getattr(base_model, "config", None) is not None:
            base_model.config.cdpruner_config = dict(cdpruner_config)


class HuluMed_Qwen2_CDPruner(HuluMed_Qwen2):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        cdpruner_config = _resolve_cdpruner_config(args)
        _apply_cdpruner_config(self.model, cdpruner_config)
        self.cdpruner_config = cdpruner_config

    def _build_cd_text_embeds(self, prompt: str) -> torch.Tensor:
        prompt = (prompt or "").strip()
        if not prompt:
            prompt = self.tokenizer.eos_token or "."

        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokenized.items()
        }
        if tokenized["input_ids"].shape[1] == 0:
            tokenized = self.tokenizer(
                self.tokenizer.eos_token or ".",
                return_tensors="pt",
                add_special_tokens=False,
            )
            tokenized = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized.items()
            }

        with torch.no_grad():
            token_embeds = self.model.get_model().embed_tokens(tokenized["input_ids"])
            token_mask = tokenized["attention_mask"].unsqueeze(-1).to(token_embeds.dtype)
            text_embeds = (token_embeds * token_mask).sum(dim=1)
            text_embeds = text_embeds / token_mask.sum(dim=1).clamp(min=1.0)
        return text_embeds

    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        do_sample = True if self.temperature > 0 else False
        cd_text_embeds = self._build_cd_text_embeds(messages.get("prompt", ""))

        with torch.inference_mode():
            output_ids = self.model.generate(
                **llm_inputs,
                cd_text_embeds=cd_text_embeds,
                do_sample=False,
                temperature=self.temperature if do_sample else 0,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
