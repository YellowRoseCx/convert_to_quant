import torch
import math

_INT4_GROUP_SIZE = 64
_INT4_MAX = 7

def _pack_int4_row_major(x: torch.Tensor) -> torch.Tensor:
    """Pack signed int8 tensor into row-major int4 representation."""
    if x.dtype != torch.int8:
        raise ValueError(f"Input must be int8, got {x.dtype}")
    x_uint = (x & 0x0F).to(torch.uint8)
    if x.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be even for int4 packing")
    return (x_uint[..., ::2] | (x_uint[..., 1::2] << 4))

def _unpack_int4_row_major(x: torch.Tensor) -> torch.Tensor:
    """Unpack row-major int4 representation into signed int8 tensor."""
    if x.dtype != torch.uint8:
        raise ValueError(f"Input must be uint8, got {x.dtype}")
    low = x & 0x0F
    high = x >> 4
    unpacked = torch.stack((low, high), dim=-1).view(*x.shape[:-1], -1).to(torch.int8)

    # Sign extension from 4-bit to 8-bit
    mask = unpacked & 0x08
    unpacked = unpacked | (mask * 0x1E)  # 0x1E is 0xF0 >> 3, effectively filling upper bits

    # Use PyTorch built-in sign extension trick
    # (x << 4) >> 4 extends the sign bit correctly for 4-bit integers
    return (unpacked << 4) >> 4

def _int4_stochastic_rng(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    return torch.rand(
        x.shape,
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
        generator=generator,
    )

def _round_int4(scaled: torch.Tensor, stochastic_rounding: int | None = 0) -> torch.Tensor:
    if stochastic_rounding is not None and stochastic_rounding > 0:
        rng = _int4_stochastic_rng(scaled, stochastic_rounding)
        scaled.add_(rng)
        return scaled.floor_().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return scaled.round_().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)

def quantize_signed_int4_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes to signed INT4, row-wise.
    Returns:
        tuple of (qdata (uint8, row-major packed), scales (float32, [rows]))
    """
    rows, _ = x.shape
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT4_MAX
    q = _round_int4(x / scales, stochastic_rounding=stochastic_rounding)
    return _pack_int4_row_major(q), scales.reshape(rows).to(torch.float32)

def dequantize_signed_int4_rowwise(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantizes from signed INT4 packed format."""
    w_int = _unpack_int4_row_major(qdata).to(torch.float32)
    w_rot = w_int * scales.to(device=qdata.device, dtype=torch.float32).reshape(-1, 1)
    return w_rot.to(output_dtype)
