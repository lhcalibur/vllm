from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn_triton import (PagedAttention,
                                                  PagedAttentionMetadata)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class TritonFlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "triton-flash-attn"

    @staticmethod
    def get_impl_cls() -> Type["TritonFlashAttentionImpl"]:
        return TritonFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TritonFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["TritonFlashAttentionMetadataBuilder"]:
        return TritonFlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class TritonFlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]
    _cached_prefill_metadata: Optional["TritonFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["TritonFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["TritonFlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = TritonFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["TritonFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = TritonFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata

    def advance_step(self, model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int, num_seqs: int, num_queries: int):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        advance_step_flashattn_triton(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=sampled_token_ids,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables)


class TritonFlashAttentionMetadataBuilder(
        CommonMetadataBuilder[TritonFlashAttentionMetadata]):

    _metadata_cls = TritonFlashAttentionMetadata


def _make_alibi_bias(alibi_slopes: torch.Tensor,
                     dtype: torch.dtype,
                     seq_lens: Optional[List[int]],
                     make_attn_mask: bool = True) -> List[torch.Tensor]:
    attn_biases = []
    if seq_lens:
        for seq_len in seq_lens:
            bias = torch.arange(seq_len, dtype=dtype)
            # NOTE(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(seq_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]

            num_heads = alibi_slopes.shape[0]
            bias = bias[None, :].repeat(
                (num_heads, 1, 1)).to(alibi_slopes.device)
            bias.mul_(alibi_slopes[:, None, None])
            if make_attn_mask:
                inf_mask = torch.empty(
                    (1, seq_len, seq_len),
                    dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1).to(
                        alibi_slopes.device)
                attn_biases.append((bias + inf_mask).to(dtype))
            else:
                attn_biases.append(bias.to(dtype))

    return attn_biases


class TritonFlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|

    Otherwise, the layout is as follows:
    |<------------------ num_generation_tokens (M) ----------------->|
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens ----------->|	
    |<-prompt_0->|...|<-prompt_N-1->|<-generation_0->|...|<-generation_M-1->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "TritonFlashAttention does not support blocksparse attention.")
        if logits_soft_cap is not None:
            raise ValueError(
                "TritonFlashAttention does not support attention logits soft "
                "capping.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

        # NOTE: Allow for switching between Triton and CK. Defaulting to triton.
        self.use_triton_flash_attn = envs.VLLM_USE_TRITON_FLASH_ATTN
        if self.use_triton_flash_attn:
            from vllm.attention.ops.triton_flash_attention import (  # noqa: F401
                triton_attention)
            self.attn_func = triton_attention
            logger.debug("Using Triton FA in TritonBackend")
            if self.sliding_window != (-1, -1):
                logger.warning("Triton FA does not currently support "
                               "sliding window attention")
        else:
            self.attn_func = _sdpa_attention
            logger.debug("Using naive attention in TritonBackend")

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return (x[:, :,
                  None, :].expand(tokens, n_kv_heads, n_rep,
                                  head_dim).reshape(tokens, n_kv_heads * n_rep,
                                                    head_dim))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonFlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonFlashAttentionImpl")

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            assert prefill_meta.seq_lens is not None
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # triton attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                attn_masks = None
                if self.use_triton_flash_attn:
                    if self.alibi_slopes is not None:
                        attn_masks = _make_alibi_bias(
                            self.alibi_slopes,
                            query.dtype,
                            attn_metadata.seq_lens,
                            make_attn_mask=False)  # type: ignore
                    out, _ = self.attn_func(
                        query,
                        key,
                        value,
                        None,
                        prefill_meta.seq_start_loc,
                        prefill_meta.seq_start_loc,
                        prefill_meta.max_prefill_seq_len,
                        prefill_meta.max_prefill_seq_len,
                        True,
                        self.scale,
                        attn_masks[0][None]
                        if attn_masks is not None else None,
                    )
                else:
                    if self.num_kv_heads != self.num_heads:
                        # Interleave for MQA workaround.
                        key = self.repeat_kv(key, self.num_queries_per_kv)
                        value = self.repeat_kv(value, self.num_queries_per_kv)
                    if self.alibi_slopes is not None:
                        attn_masks = _make_alibi_bias(
                            self.alibi_slopes,
                            query.dtype,
                            attn_metadata.seq_lens,
                            make_attn_mask=True)  # type: ignore
                    query = query.movedim(0, query.dim() - 2)
                    key = key.movedim(0, key.dim() - 2)
                    value = value.movedim(0, value.dim() - 2)
                    # sdpa math backend attention
                    out = self.attn_func(
                        query,
                        key,
                        value,
                        prefill_meta.seq_lens,
                        num_tokens,
                        self.num_heads,
                        self.head_size,
                        self.scale,
                        attn_masks,
                    )

                # common code for prefill
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window[0],
                    k_scale,
                    v_scale,
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.seq_lens_tensor,
                decode_meta.max_decode_seq_len,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                k_scale,
                v_scale,
            )

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: List[int],
    num_tokens: int,
    num_heads: int,
    head_size: int,
    scale: float,
    attn_masks: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    start = 0
    output = torch.empty((num_tokens, num_heads, head_size),
                         dtype=query.dtype,
                         device=query.device)

    for i, seq_len in enumerate(seq_lens):
        end = start + seq_len
        with torch.backends.cuda.sdp_kernel(enable_math=True,
                                            enable_flash=False,
                                            enable_mem_efficient=False):
            sub_out = torch.nn.functional.scaled_dot_product_attention(
                query[:, start:end, :],
                key[:, start:end, :],
                value[:, start:end, :],
                dropout_p=0.0,
                is_causal=attn_masks is None,
                attn_mask=attn_masks[i] if attn_masks else None,
                scale=scale).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end

    return output


def verify_tensor(name: str, tensor: torch.Tensor, size_0: Optional[int],
                  size_1: Optional[int], dtype: torch.dtype) -> None:
    """
    Verify tensor properties including shape, contiguity, and data type.
    
    Args:
        name: Tensor name for error messages
        tensor: Tensor to verify
        size_0: Expected size of first dimension (-1 for any size)
        size_1: Expected size of second dimension (-1 for any size)
        dtype: Expected data type
    
    Raises:
        ValueError: If tensor doesn't meet requirements
    """
    # Check first dimension
    if size_0 != -1 and tensor.size(0) != size_0:
        raise ValueError(f"Tensor '{name}' has wrong size_0: "
                         f"expected {size_0}, got {tensor.size(0)}")

    # Check second dimension if tensor is 2D
    if size_1 != -1:
        if tensor.dim() < 2:
            raise ValueError(f"Tensor '{name}' should have at least 2D "
                             f"but got {tensor.dim()}D")
        if tensor.size(1) != size_1:
            raise ValueError(f"Tensor '{name}' has wrong size_1: "
                             f"expected {size_1}, got {tensor.size(1)}")

    # Check contiguity
    if not tensor.is_contiguous():
        raise ValueError(f"Tensor '{name}' must be contiguous")

    # Check data type
    if tensor.dtype != dtype:
        raise ValueError(f"Tensor '{name}' has wrong dtype: "
                         f"expected {dtype}, got {tensor.dtype}")


@triton.jit
def advance_step_kernel(
    # Pointers to input/output tensors
    input_tokens_ptr,  # [num_seqs, max_seq_len]
    sampled_token_ids_ptr,  # [num_queries, 1]
    input_positions_ptr,  # [num_seqs]
    seq_lens_ptr,  # [num_seqs]
    slot_mapping_ptr,  # [num_seqs]
    block_tables_ptr,  # [max_blocks_per_seq]

    # Scalar values
    num_queries,
    block_tables_stride,
    table_block_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)

    query_offsets = pid + tl.arange(0, BLOCK_SIZE)
    query_masks = query_offsets < num_queries

    # Update input tokens with new sampled token
    tl.store(input_tokens_ptr + query_offsets,
             tl.load(sampled_token_ids_ptr + query_offsets, mask=query_masks),
             mask=query_masks)

    # Load current sequence length
    seq_len = tl.load(seq_lens_ptr + query_offsets, mask=query_masks)

    # Calculate next sequence position
    next_seq_len = seq_len + 1
    next_input_pos = next_seq_len - 1

    # Update sequence length
    tl.store(seq_lens_ptr + query_offsets, next_seq_len, mask=query_masks)

    # Update input position
    tl.store(input_positions_ptr + query_offsets,
             next_input_pos,
             mask=query_masks)

    # Calculate KV cache slot mapping
    block_index = next_input_pos // table_block_size
    block_offset = next_input_pos % table_block_size

    block_tables_ptr_offset = query_offsets * block_tables_stride + block_index

    # Load block table entry and calculate slot number
    block_table_entry = tl.load(block_tables_ptr + block_tables_ptr_offset,
                                mask=query_masks)
    slot_num = block_table_entry * table_block_size + block_offset

    # Store slot mapping
    tl.store(slot_mapping_ptr + query_offsets, slot_num, mask=query_masks)


def advance_step_flashattn_triton(
    num_seqs: int,
    num_queries: int,
    block_size: int,
    input_tokens: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    input_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    """
    Triton implementation of advance_step_flashattn with tensor verification
    """
    # Verify all input tensors
    verify_tensor("input_tokens", input_tokens, num_seqs, -1, torch.long)
    verify_tensor("sampled_token_ids", sampled_token_ids, num_queries, 1,
                  torch.long)
    verify_tensor("input_positions", input_positions, num_seqs, -1, torch.long)
    verify_tensor("seq_lens", seq_lens, num_seqs, -1, torch.int32)
    verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, torch.long)
    verify_tensor("block_tables", block_tables, num_seqs, -1, torch.int32)

    # Ensure all tensors are on the same device
    device = input_tokens.device
    if not all(t.device == device for t in [
            sampled_token_ids, input_positions, seq_lens, slot_mapping,
            block_tables
    ]):
        raise ValueError("All tensors must be on the same device")

    # Configure grid and block sizes
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_queries, BLOCK_SIZE), )

    # Launch kernel
    advance_step_kernel[grid](
        input_tokens,
        sampled_token_ids,
        input_positions,
        seq_lens,
        slot_mapping,
        block_tables,
        num_queries,
        block_tables_stride=block_tables.stride(0),
        table_block_size=block_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
