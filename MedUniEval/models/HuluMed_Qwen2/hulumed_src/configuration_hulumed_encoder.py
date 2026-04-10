# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/configuration_siglip.py.
# Below is the original copyright:
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HuluMed vision encoder model configuration."""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


def normalize_vision_zip_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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


class HulumedVisionEncoderConfig(PretrainedConfig):

    model_type = "hulumed_vision_encoder"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        vision_zip_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.vision_zip_config = normalize_vision_zip_config(vision_zip_config)
