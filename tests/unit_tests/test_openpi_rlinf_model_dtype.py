# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from rlinf.models.embodiment.openpi_rlinf import _pi0_config_dtype


@pytest.mark.parametrize(
    ("torch_dtype", "config_dtype"),
    [
        (torch.float32, "float32"),
        (torch.bfloat16, "bfloat16"),
        (torch.float16, "float16"),
        (None, "bfloat16"),
    ],
)
def test_pi0_config_dtype_matches_requested_model_precision(torch_dtype, config_dtype):
    assert _pi0_config_dtype(torch_dtype) == config_dtype


def test_pi0_config_dtype_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported OpenPI model dtype"):
        _pi0_config_dtype(torch.float64)
