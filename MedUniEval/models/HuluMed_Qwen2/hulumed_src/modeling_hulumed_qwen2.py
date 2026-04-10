# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""PyTorch HuluMed model."""

import importlib.util
import os.path as osp
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoModel, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 128
MAX_FRAMES = 768
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"

# Stream arguments
STREAM_START_TOKEN = "<|stream_start|>"
STREAM_END_TOKEN = "<|stream_end|>"
STREAM_MAX_FRAMES = 400

MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
    "<audio>": -202,
}

subimage_token_num=196
try:
    from .configuration_hulumed_qwen2 import HulumedQwen2Config
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "configuration_hulumed_qwen2",
        osp.join(osp.dirname(__file__), "configuration_hulumed_qwen2.py"),
    )
    configuration_hulumed_qwen2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration_hulumed_qwen2)
    HulumedQwen2Config = getattr(
        configuration_hulumed_qwen2,
        "HulumedQwen2Config",
    )


def build_mlp(depth, hidden_size, output_hidden_size):
    """Build MLP layers for projection."""
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


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


def build_vision_projector(config, delay_load=False, **kwargs):
    """Build vision projector based on config."""
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if projector_type == "linear":
        return nn.Linear(config.vision_encoder_config.hidden_size, config.hidden_size)
    elif projector_type.startswith("mlp"):
        return MlpGeluProjector(config, projector_type)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


class MlpGeluProjector(nn.Module):
    """MLP projector with GELU activation."""

    def __init__(self, config, projector_type):
        super().__init__()
        
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match is None:
            raise ValueError(f"Invalid projector type format: {projector_type}")
        mlp_depth = int(mlp_gelu_match.group(1))

        self.readout = build_mlp(
            mlp_depth, 
            config.vision_encoder_config.hidden_size, 
            config.hidden_size
        )

    def forward(self, x):
        return self.readout(x)


class HulumedMetaModel:
    """Meta model for HuluMed that handles vision encoder initialization."""

    def __init__(self, config):
        super(HulumedMetaModel, self).__init__(config)
        self.config.vision_zip_config = normalize_vision_zip_config(
            getattr(self.config, "vision_zip_config", None)
        )
        print('config.vision_encoder',config.vision_encoder)
        if config.vision_encoder is not None:
            # Load from pretrained path
            print('Load from pretrained path')
            self.vision_encoder = AutoModel.from_pretrained(
                config.vision_encoder,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
            self.config.vision_encoder_config = self.vision_encoder.config
            self.config.vision_encoder = None
        elif config.vision_encoder_config is not None:
            # Build from config
            print('Build from config')
            self.vision_encoder = AutoModel.from_config(
                self.config.vision_encoder_config,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
        else:
            raise ValueError("Vision encoder is not provided in config")

        self.config.vision_encoder_config = self.vision_encoder.config
        self.config.vision_encoder_config.vision_zip_config = normalize_vision_zip_config(
            getattr(self.config.vision_encoder_config, "vision_zip_config", self.config.vision_zip_config)
        )
        self.vision_encoder.config.vision_zip_config = dict(self.config.vision_encoder_config.vision_zip_config)
        self.config.vision_zip_config = dict(self.vision_encoder.config.vision_zip_config)
        self.mm_projector = build_vision_projector(config)

    def get_vision_encoder(self):
        return self.vision_encoder

    def get_mm_projector(self):
        return self.mm_projector


class HulumedQwen2Model(HulumedMetaModel, Qwen2Model):
    """HuluMed Qwen2 Model."""
    
    config_class = HulumedQwen2Config

    def __init__(self, config: HulumedQwen2Config):
        super(HulumedQwen2Model, self).__init__(config)


class HulumedMetaForCausalLM(ABC):
    """Meta class for HuluMed Causal LM with multimodal support."""

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode images using vision encoder and projector."""
        vision_outputs = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        if isinstance(vision_outputs, tuple):
            mm_features, attn_scores, attn_keys = vision_outputs
        else:
            mm_features, attn_scores, attn_keys = vision_outputs, None, None
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features, attn_scores, attn_keys

    def _get_vision_zip_config(self) -> Dict[str, Any]:
        return normalize_vision_zip_config(getattr(self.config, "vision_zip_config", None))

    def _filter_valid_visual_tensors(
        self,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
        *visual_tensors: Optional[torch.Tensor],
    ):
        mask_device = batched_num_patches.device
        for tensor in visual_tensors:
            if tensor is not None:
                mask_device = tensor.device
                break
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            num_patches = int(num_patches)
            valid_mask = torch.full(
                (num_patches,),
                modal != "text",
                dtype=torch.bool,
                device=mask_device,
            )
            valid_masks.append(valid_mask)

        valid_mask = torch.cat(valid_masks) if valid_masks else torch.zeros((0,), dtype=torch.bool, device=mask_device)
        outputs = []
        for tensor in visual_tensors:
            outputs.append(None if tensor is None else tensor[valid_mask])
        return outputs if len(outputs) > 1 else outputs[0]

    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        """Filter out text-only samples and keep only valid visual tokens."""
        return self._filter_valid_visual_tensors(
            batched_num_patches,
            modals,
            mm_features,
        )

    def _maybe_truncate_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        compression_mask: torch.BoolTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        extra_tensors: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        """Truncate visual tokens if necessary based on position_ids."""
        if position_ids is None or mm_features.shape[0] == input_ids.eq(self.config.image_token_index).sum():
            if extra_tensors is None:
                return mm_features, compression_mask
            return (mm_features, compression_mask, *extra_tensors)

        truncation_mask = []
        for num_patches, modal in zip(batched_num_patches, modals):
            num_patches = int(num_patches)
            if modal == "text":
                truncation_mask.append(torch.ones((0,), dtype=torch.bool, device=input_ids.device))
            else:
                truncation_mask.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

        seq_end_indices = torch.nonzero(position_ids == 0)[:, 0]
        seq_end_indices = seq_end_indices[seq_end_indices > 0].tolist() + [len(input_ids)]
        seq_start_indices = [0] + seq_end_indices[:-1]
        num_visual_tokens = [
            input_ids[start:end].eq(self.config.image_token_index).sum()
            for start, end in zip(seq_start_indices, seq_end_indices)
        ]

        for n, mask in zip(num_visual_tokens, truncation_mask):
            if len(mask) > 0:
                mask[n:] = False
        truncation_mask = torch.cat(truncation_mask)

        outputs = [mm_features[truncation_mask], compression_mask[truncation_mask]]
        if extra_tensors is not None:
            for tensor in extra_tensors:
                outputs.append(None if tensor is None else tensor[truncation_mask])
        return tuple(outputs)

    def _select_vision_zip_tokens(
        self,
        sample_features: torch.Tensor,
        sample_scores: torch.Tensor,
        sample_keys: torch.Tensor,
        vision_zip_config: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        num_tokens = sample_features.shape[0]
        if num_tokens == 0:
            empty_mask = torch.zeros((0,), dtype=torch.bool, device=sample_features.device)
            return sample_features, empty_mask

        domain_kept = min(
            num_tokens,
            max(1, int(math.floor(num_tokens * vision_zip_config["domain_kept_ratio"]))),
        )
        contextual_candidates = num_tokens - domain_kept
        if contextual_candidates <= 0:
            select_mask = torch.ones(num_tokens, dtype=torch.bool, device=sample_features.device)
            return sample_features, select_mask

        contextual_kept = min(
            contextual_candidates,
            max(1, int(math.floor(num_tokens * vision_zip_config["contextual_kept_ratio"]))),
        )

        _, domain_indices = torch.topk(sample_scores, domain_kept)
        select_mask = torch.zeros(num_tokens, dtype=torch.bool, device=sample_features.device)
        select_mask[domain_indices] = True

        contextual_mask = ~select_mask
        contextual_positions = contextual_mask.nonzero(as_tuple=True)[0]
        contextual_features = sample_features[contextual_mask]
        contextual_keys = sample_keys[contextual_mask]

        if contextual_positions.numel() == 0 or contextual_kept <= 0:
            return sample_features[select_mask], select_mask

        contextual_keys = nn.functional.normalize(contextual_keys, dim=-1, eps=1e-6)
        step = max(1, contextual_positions.numel() // contextual_kept)
        anchor_local_indices = torch.arange(
            0,
            contextual_positions.numel(),
            step,
            device=sample_features.device,
        )[:contextual_kept]
        anchor_global_indices = contextual_positions[anchor_local_indices]
        select_mask[anchor_global_indices] = True

        selected_features = sample_features[select_mask].clone()
        non_anchor_mask = torch.ones(
            contextual_positions.numel(),
            dtype=torch.bool,
            device=sample_features.device,
        )
        non_anchor_mask[anchor_local_indices] = False
        tokens_to_merge = contextual_keys[non_anchor_mask]
        hidden_to_merge = contextual_features[non_anchor_mask]
        anchor_tokens = contextual_keys[anchor_local_indices]
        target_hidden = contextual_features[anchor_local_indices]

        aggregated_hidden = torch.zeros_like(target_hidden)
        if tokens_to_merge.numel() > 0:
            similarity = tokens_to_merge @ anchor_tokens.transpose(0, 1)
            assignments = similarity.argmax(dim=1)
            aggregated_hidden.index_add_(0, assignments, hidden_to_merge)
            counts = torch.zeros(
                anchor_local_indices.numel(),
                device=sample_features.device,
                dtype=sample_features.dtype,
            )
            counts.index_add_(
                0,
                assignments,
                torch.ones(assignments.shape[0], device=sample_features.device, dtype=sample_features.dtype),
            )
            aggregated_hidden = aggregated_hidden / counts.clamp(min=1.0).unsqueeze(-1)

        contextual_features = target_hidden + aggregated_hidden
        selected_positions = select_mask.nonzero(as_tuple=True)[0]
        contextual_positions_in_selected = torch.searchsorted(selected_positions, anchor_global_indices)
        selected_features[contextual_positions_in_selected] = contextual_features.to(selected_features.dtype)
        return selected_features, select_mask

    def _apply_vision_zip(
        self,
        mm_features: torch.FloatTensor,
        attn_scores: torch.Tensor,
        attn_keys: torch.Tensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        vision_zip_config = self._get_vision_zip_config()
        selected_features = []
        compression_masks = []
        start = 0

        for num_patches, grid_size, merge_size, modal in zip(
            batched_num_patches.tolist(),
            grid_sizes.tolist(),
            merge_sizes.tolist(),
            modals,
        ):
            if modal == "text" or num_patches == 0:
                continue

            end = start + num_patches
            sample_features = mm_features[start:end]
            sample_scores = attn_scores[start:end]
            sample_keys = attn_keys[start:end]
            if vision_zip_config["enable"] and modal in {"image", "video"}:
                num_frames, height, width = (int(x) for x in grid_size)
                merge_size = int(merge_size)
                tokens_per_frame = (height // merge_size) * (width // merge_size)

                frame_features = sample_features.view(num_frames, tokens_per_frame, -1)
                frame_scores = sample_scores.view(num_frames, tokens_per_frame)
                frame_keys = sample_keys.view(num_frames, tokens_per_frame, -1)

                frame_outputs = []
                frame_masks = []
                for frame_feature, frame_score, frame_key in zip(frame_features, frame_scores, frame_keys):
                    selected_frame_features, frame_mask = self._select_vision_zip_tokens(
                        frame_feature,
                        frame_score,
                        frame_key,
                        vision_zip_config,
                    )
                    frame_outputs.append(selected_frame_features)
                    frame_masks.append(frame_mask)

                sample_features = torch.cat(frame_outputs, dim=0)
                sample_mask = torch.cat(frame_masks, dim=0)
            else:
                sample_mask = torch.ones(num_patches, dtype=torch.bool, device=mm_features.device)

            selected_features.append(sample_features)
            compression_masks.append(sample_mask)
            start = end

        if not selected_features:
            empty_mask = torch.zeros((0,), dtype=torch.bool, device=mm_features.device)
            return mm_features, empty_mask
        mm_features = torch.cat(selected_features, dim=0)
        compression_mask = torch.cat(compression_masks, dim=0)
        return mm_features, compression_mask

    def _get_compression_mask(
        self,
        pixel_values: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
        threshold: float = 0.1,
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        """Get compression mask for video tokens based on frame differences."""
        batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, num_patches, grid_size, merge_size, modal in zip(
            batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
        ):
            t, h, w = grid_size
            if modal == "image" or (modal == "video" and t == 1):
                compression_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=images.device))

            elif modal == "video":
                # Video token compression based on pixel differences
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = (pixel_diff / 255.0) > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

            else:
                # Pseudo image case
                compression_masks.append(torch.ones((0,), dtype=torch.bool, device=images.device))

        return torch.cat(compression_masks)

    def _compress_visual_tokens(
        self,
        compression_mask: torch.BoolTensor,
        mm_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """Compress visual tokens based on compression mask."""
        mm_features = mm_features[compression_mask]
        image_selected = (input_ids == self.config.image_token_index)

        text_masks = torch.logical_not(image_selected)
        text_masks[image_selected] = compression_mask
        input_ids = input_ids[text_masks]

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None:
            position_ids = position_ids[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([
                torch.arange(end - start, device=input_ids.device) 
                for start, end in zip(pos_start, pos_end)
            ])

        return mm_features, input_ids, attention_mask, position_ids, labels

    def _apply_vision_zip_to_sequence(
        self,
        selected_features: torch.FloatTensor,
        select_mask: torch.BoolTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """Apply VisionZip-selected visual tokens to the flattened multimodal sequence."""
        image_selected = (input_ids == self.config.image_token_index)
        text_masks = torch.logical_not(image_selected)
        text_masks[image_selected] = select_mask
        input_ids = input_ids[text_masks]

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None:
            position_ids = position_ids[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([
                torch.arange(end - start, device=input_ids.device)
                for start, end in zip(pos_start, pos_end)
            ])

        return selected_features, input_ids, attention_mask, position_ids, labels

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
    ):
        """Prepare inputs and labels for multimodal training/inference."""
        vision_encoder = self.get_vision_encoder()
        
        # Text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels

        # 1. Flatten text inputs
        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. Embed visual tokens
        batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
        mm_features, attn_scores, attn_keys = self.encode_images(pixel_values, grid_sizes, merge_sizes)
        mm_features = mm_features.to(input_ids.device)
        if attn_scores is not None:
            attn_scores = attn_scores.to(input_ids.device)
        if attn_keys is not None:
            attn_keys = attn_keys.to(input_ids.device)

        mm_features, attn_scores, attn_keys = self._filter_valid_visual_tensors(
            batched_num_patches,
            modals,
            mm_features,
            attn_scores,
            attn_keys,
        )

        vision_zip_config = self._get_vision_zip_config()
        if vision_zip_config["enable"]:
            assert B == 1, "VisionZip token selection is only supported for batch_size=1"
            if attn_scores is None or attn_keys is None:
                raise ValueError("VisionZip is enabled but the vision encoder did not return attention scores/keys.")
            compression_mask = torch.ones(mm_features.shape[0], dtype=torch.bool, device=input_ids.device)
        else:
            compression_mask = self._get_compression_mask(
                pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals
            )

        maybe_truncated = self._maybe_truncate_visual_tokens(
            mm_features,
            compression_mask,
            batched_num_patches,
            modals,
            input_ids,
            position_ids,
            extra_tensors=[attn_scores, attn_keys],
        )
        mm_features, compression_mask, attn_scores, attn_keys = maybe_truncated

        # 3. Compress visual tokens if enabled
        if vision_zip_config["enable"]:
            mm_features, compression_mask = self._apply_vision_zip(
                mm_features,
                attn_scores,
                attn_keys,
                batched_num_patches,
                grid_sizes,
                merge_sizes,
                modals,
            )
            mm_features, input_ids, attention_mask, position_ids, labels = self._apply_vision_zip_to_sequence(
                mm_features, compression_mask, input_ids, attention_mask, position_ids, labels
            )
        elif self.config.use_token_compression:
            assert B == 1, "Token compression is only supported for batch_size=1"
            mm_features, input_ids, attention_mask, position_ids, labels = self._compress_visual_tokens(
                compression_mask, mm_features, input_ids, attention_mask, position_ids, labels
            )

        # 4. Embed text tokens
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        # 5. Replace multimodal tokens with features
        image_selected = (input_ids == self.config.image_token_index)
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features

        # 6. Reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels


class HulumedQwen2ForCausalLM(Qwen2ForCausalLM, HulumedMetaForCausalLM):
    """HuluMed Qwen2 model for causal language modeling with multimodal support."""
    
    config_class = HulumedQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = HulumedQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # Multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass with multimodal support."""
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        # Multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate with multimodal support."""
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation."""
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
