# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

from ..utils import deprecate
from .controlnets.controlnet.diffsynth import (  # noqa
    DiffsynthControlNetConditioningEmbedding,
    DiffsynthControlNetModel,
    DiffsynthControlNetOutput,
    zero_module,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiffsynthControlNetOutput(DiffsynthControlNetOutput):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `DiffsynthControlNetOutput` from `diffusers.models.controlnet_diffsynth` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetOutput`, instead."
        deprecate("diffusers.models.controlnet_sd3.DiffsynthControlNetOutput", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)


class DiffsynthControlNetModel(DiffsynthControlNetModel):
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        extra_conditioning_channels: int = 0,
    ):
        deprecation_message = "Importing `DiffsynthControlNetModel` from `diffusers.models.controlnet_sd3` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel`, instead."
        deprecate("diffusers.models.controlnet_diffsynth.DiffsynthControlNetModel", "0.34", deprecation_message)
        super().__init__(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            pooled_projection_dim=pooled_projection_dim,
            out_channels=out_channels,
            pos_embed_max_size=pos_embed_max_size,
            extra_conditioning_channels=extra_conditioning_channels,
        )

class DiffsynthMultiControlNetModel(DiffsynthMultiControlNetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `DiffsynthMultiControlNetModel` from `diffusers.models.controlnet_diffsynth` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.controlnet_sd3 import SD3MultiControlNetModel`, instead."
        deprecate("diffusers.models.controlnet_diffsynth.DiffsynthMultiControlNetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)







