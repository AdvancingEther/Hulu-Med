import os

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


def _normalize_vision_zip_config(config=None):
    default_config = {
        "enable": False,
        "domain_kept_ratio": 0.65,
        "contextual_kept_ratio": 0.05,
    }
    if config is None:
        return dict(default_config)

    merged = dict(default_config)
    merged.update(config)
    merged["enable"] = bool(merged["enable"])
    merged["domain_kept_ratio"] = float(merged["domain_kept_ratio"])
    merged["contextual_kept_ratio"] = float(merged["contextual_kept_ratio"])
    return merged


def _resolve_vision_zip_config(args):
    vision_zip_config = {}

    env_enable = _read_env_bool("VISION_ZIP_ENABLE")
    env_domain_ratio = _read_env_float("VISION_ZIP_DOMAIN_KEPT_RATIO")
    env_contextual_ratio = _read_env_float("VISION_ZIP_CONTEXTUAL_KEPT_RATIO")

    arg_enable = getattr(args, "vision_zip_enable", None)
    arg_domain_ratio = getattr(args, "vision_zip_domain_kept_ratio", None)
    arg_contextual_ratio = getattr(args, "vision_zip_contextual_kept_ratio", None)

    if env_enable is not None:
        vision_zip_config["enable"] = env_enable
    if env_domain_ratio is not None:
        vision_zip_config["domain_kept_ratio"] = env_domain_ratio
    if env_contextual_ratio is not None:
        vision_zip_config["contextual_kept_ratio"] = env_contextual_ratio

    if arg_enable is not None:
        vision_zip_config["enable"] = arg_enable
    if arg_domain_ratio is not None:
        vision_zip_config["domain_kept_ratio"] = arg_domain_ratio
    if arg_contextual_ratio is not None:
        vision_zip_config["contextual_kept_ratio"] = arg_contextual_ratio

    return _normalize_vision_zip_config(vision_zip_config)


def _apply_vision_zip_config(model, vision_zip_config):
    model.config.vision_zip_config = dict(vision_zip_config)
    if getattr(model.config, "vision_encoder_config", None) is not None:
        model.config.vision_encoder_config.vision_zip_config = dict(vision_zip_config)

    if hasattr(model, "get_model"):
        base_model = model.get_model()
        if getattr(base_model, "config", None) is not None:
            base_model.config.vision_zip_config = dict(vision_zip_config)
            if getattr(base_model.config, "vision_encoder_config", None) is not None:
                base_model.config.vision_encoder_config.vision_zip_config = dict(vision_zip_config)
        if hasattr(base_model, "vision_encoder") and getattr(base_model, "vision_encoder", None) is not None:
            base_model.vision_encoder.config.vision_zip_config = dict(vision_zip_config)


class HuluMed_Qwen2_VisionZip(HuluMed_Qwen2):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        vision_zip_config = _resolve_vision_zip_config(args)
        _apply_vision_zip_config(self.model, vision_zip_config)
        self.vision_zip_config = vision_zip_config
