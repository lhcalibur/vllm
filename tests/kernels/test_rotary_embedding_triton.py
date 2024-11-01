"""
Copyright 2023 The vLLM team.
Copyright 2024 The SiOrigin Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils import seed_everything

from .allclose_default import get_default_atol, get_default_rtol


def rotary_embedding_opcheck(rot,
                             positions: torch.Tensor,
                             query: torch.Tensor,
                             key: torch.Tensor,
                             offsets: Optional[torch.Tensor] = None):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    if offsets is not None:
        opcheck(torch.ops.triton.batched_rotary_embedding,
                (positions, query, key, rot.head_size, cos_sin_cache,
                 rot.is_neox_style, rot.rotary_dim, offsets))
    else:
        opcheck(torch.ops.triton.rotary_embedding,
                (positions, query, key, rot.head_size, cos_sin_cache,
                 rot.is_neox_style))


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_rotary_embedding_opcheck(dist_init, seed, device, max_position,
                                  is_neox_style, rotary_dim, head_size,
                                  seq_len, batch_size):
    seed_everything(seed)
    torch.set_default_device(device)
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, torch.float32)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn_like(query)

    raw_query = query.clone()
    raw_key = key.clone()
    ref_query, ref_key = rot.forward_native(positions, raw_query, raw_key)
    rot(positions, query, key)
    torch.testing.assert_close(query,
                               ref_query,
                               atol=get_default_atol(ref_query),
                               rtol=get_default_rtol(ref_query))
    torch.testing.assert_close(key,
                               ref_key,
                               atol=get_default_atol(ref_query),
                               rtol=get_default_rtol(ref_query))

    offsets = torch.randint(0,
                            max_position, (batch_size * seq_len, ),
                            device=device)

    rot(positions, raw_query, raw_key, offsets)
    torch.testing.assert_close(query,
                               ref_query,
                               atol=get_default_atol(ref_query),
                               rtol=get_default_rtol(ref_query))
    torch.testing.assert_close(key,
                               ref_key,
                               atol=get_default_atol(ref_query),
                               rtol=get_default_rtol(ref_query))

    rotary_embedding_opcheck(rot, positions, query, key)
    rotary_embedding_opcheck(rot, positions, query, key, offsets)
