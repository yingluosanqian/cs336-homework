import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import math


def ref_forward_kernel(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float = 1e-6,
):
    # RMSNorm does NOT subtract mean
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x / torch.sqrt(rms + eps)
    if norm_weight is not None:
        normalized = normalized * norm_weight
    return normalized


@cute.jit
def warp_reduce_rmsnorm(
    welf_mean: cute.Float32,
    welf_count: cutlass.Float32,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> tuple[cute.Float32, cute.Float32]:
    for offset in cutlass.range_constexpr(4 + 1):
        other_mean = cute.arch.shuffle_sync_down(welf_mean, 1 << (4 - offset))
        other_count = cute.arch.shuffle_sync_down(
            welf_count, 1 << (4 - offset))
        if other_count > 0.0:
            total = welf_count + other_count
            delta = other_mean - welf_mean
            welf_mean += delta * (other_count / total)
            welf_count = total

    return welf_mean, welf_count


@cute.kernel
def rms_norm_kernel(
    x: cute.Tensor,
    norm_weight: cute.Tensor | None,
    out: cute.Tensor,
    eps: cute.Float32,
    tiled_copy_x: cute.TiledCopy,
    tiled_copy_norm: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_id, lane_id = tidx // 32, tidx & 31
    bidx, _, _ = cute.arch.block_idx()
    batch, hidden_dim = x.shape
    TILE_N = 512
    tiler = (1, TILE_N)
    tiler_coord = (bidx, None)

    # Get the working tile for this cta
    gX = cute.local_tile(x, tiler=tiler, coord=tiler_coord)[0, None, None]

    # Scale Residual
    thr_copy_x = tiled_copy_x.get_slice(tidx)
    tXgX = thr_copy_x.partition_S(gX)
    tXrX = cute.make_fragment_like(
        tXgX[None, None, 0], tXgX.element_type)

    cX = cute.make_identity_tensor(hidden_dim)
    ccX = cute.local_tile(cX, tiler=(TILE_N,), coord=(None,))
    tXcX = thr_copy_x.partition_S(ccX)
    tXpX = cute.make_fragment(
        cute.make_layout(
            (tXgX.shape[0][1], tXgX.shape[1]),
            stride=(1, tXgX.shape[0][1]),
        ),
        cutlass.Boolean,
    )
    # print("gX:", gX)
    # print("tXgX:", tXgX)
    # print("tXrX:", tXrX)
    # print("tXcX:", tXcX)
    # print("tXpX:", tXpX)

    smem = cutlass.utils.SmemAllocator()
    welf_mean = cutlass.Float32(0.0)
    welf_count = cutlass.Float32(0.0)
    smem_mean = smem.allocate_tensor(cutlass.Float32, 4)
    smem_count = smem.allocate_tensor(cutlass.Float32, 32)
    for k_iter in range(tXgX.shape[-1]):
        for rest in range(cute.size(tXpX.shape[0])):
            for n in range(cute.size(tXpX.shape[1])):
                tXpX[rest, n] = cute.elem_less(
                    tXcX[rest, n, k_iter], hidden_dim)
        cute.copy(tiled_copy_x, tXgX[None, None, k_iter], tXrX, pred=tXpX)

        for V in range(cute.size(tXrX.shape[0][0])):
            for rest in range(cute.size(tXrX.shape[0][1])):
                for n in range(cute.size(tXrX.shape[1])):
                    if cute.elem_less(tXcX[rest, n, k_iter], hidden_dim):
                        welf_count += 1
                        delta = tXrX[((V, rest), n)].to(cute.Float32) * \
                            tXrX[((V, rest), n)].to(
                                cute.Float32) - welf_mean
                        welf_mean += delta / welf_count

    # Reduce
    welf_mean, welf_count = warp_reduce_rmsnorm(
        welf_mean, welf_count)
    if lane_id == 0:
        smem_mean[warp_id] = welf_mean
        smem_count[warp_id] = welf_count

    cute.arch.sync_threads()
    if warp_id == 0:
        welf_mean = smem_mean[lane_id] if lane_id < 4 else 0.0
        welf_count = smem_count[lane_id] if lane_id < 4 else 0.0
        welf_mean, welf_count = warp_reduce_rmsnorm(welf_mean, welf_count)
        if tidx == 0:
            smem_mean[0] = cute.rsqrt(welf_mean + eps)

    # norm_weight
    if cutlass.const_expr(isinstance(norm_weight, cute.Tensor)):
        thr_copy_norm = tiled_copy_norm.get_slice(tidx)
        gW = cute.local_tile(norm_weight, tiler=(TILE_N,), coord=(None,))
        tWgW = thr_copy_norm.partition_S(gW)
        tWrW = cute.make_fragment_like(
            tWgW[None, None, 0], tWgW.element_type)

    cute.arch.sync_threads()
    inv = smem_mean[0]

    gO = cute.local_tile(out, tiler=tiler, coord=tiler_coord)[0, None, None]
    tOgO = thr_copy_x.partition_S(gO)
    tOrO = cute.make_fragment_like(
        tOgO[None, None, 0], tOgO.element_type)
    for k_iter in range(tXgX.shape[-1]):
        for rest in range(cute.size(tXpX.shape[0])):
            for n in range(cute.size(tXpX.shape[1])):
                tXpX[rest, n] = cute.elem_less(
                    tXcX[rest, n, k_iter], hidden_dim)
        cute.copy(tiled_copy_x, tXgX[None, None, k_iter], tXrX, pred=tXpX)
        norm_x = tXrX.load().to(cute.Float32) * inv
        if cutlass.const_expr(isinstance(norm_weight, cute.Tensor)):
            cute.copy(tiled_copy_norm,
                      tWgW[None, None, k_iter], tWrW, pred=tXpX)
            norm_x = norm_x * tWrW.load().to(cute.Float32)
        tOrO.store(norm_x.to(tOrO.element_type))
        cute.copy(tiled_copy_x, tOrO, tOgO[None, None, k_iter], pred=tXpX)


@cute.jit
def get_tiled_copy(t: cute.Tensor, aligned: cute.Int32) -> cute.TiledCopy:
    num_vectorized = math.gcd(4, aligned)
    atom_sync_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        t.element_type,
        num_bits_per_copy=num_vectorized * t.element_type.width,
    )
    tiled_copy = cute.make_tiled_copy_tv(
        atom_sync_copy,
        cute.make_layout((128,)),
        cute.make_layout((num_vectorized,))
    )
    return tiled_copy


@cute.jit
def rms_norm_forward_jit(
    x: cute.Tensor,
    norm_weight: cute.Tensor | None,
    out: cute.Tensor,
    eps: cute.Float32,
):
    batch, hidden_dim = x.shape

    num_vectorized = x.layout[1].max_alignment
    # [residual, x, modulated, residual_out]
    tiled_copy_x = get_tiled_copy(x, num_vectorized)
    # [weight, bias]
    tiled_copy_norm = None
    if cutlass.const_expr(isinstance(norm_weight, cute.Tensor)):
        tiled_copy_norm = get_tiled_copy(norm_weight, num_vectorized)

    grid = (batch, 1, 1)
    block = (128, 1, 1)

    rms_norm_kernel(
        x, norm_weight, out, eps, tiled_copy_x, tiled_copy_norm,
    ).launch(grid=grid, block=block)

    return out


def rms_norm_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    # CPU fallback: if CUDA is unavailable, run the reference PyTorch version.
    if not torch.cuda.is_available():
        return ref_forward_kernel(x, norm_weight, eps)

    # Device
    origin_device = x.device
    x = x.to("cuda").detach()
    if norm_weight is not None:
        norm_weight = norm_weight.to("cuda").detach()
    # Shape
    origin_x_shape = x.shape
    dtype = x.dtype
    x = x.reshape(-1, x.shape[-1])
    # [x]
    x = from_dlpack(x, assumed_align=16)
    # [norm_weight, norm_bias]
    if norm_weight is not None:
        norm_weight = norm_weight.contiguous()
        norm_weight = from_dlpack(norm_weight, assumed_align=16)
    # [out]
    out = torch.empty(*x.shape, device="cuda", dtype=dtype)

    # CUTE-JIT
    if not hasattr(rms_norm_forward, "kernel_cache"):
        rms_norm_forward.kernel_cache = {}
    key = (
        tuple(x.shape),
        None if norm_weight is None else tuple(norm_weight.shape),
        eps,
    )
    if key not in rms_norm_forward.kernel_cache:
        rms_norm_forward.kernel_cache[key] = cute.compile(
            rms_norm_forward_jit, x, norm_weight,
            from_dlpack(out, assumed_align=16), cutlass.Float32(eps),
        )
    rms_norm_forward.kernel_cache[key](
        x, norm_weight, from_dlpack(
            out, assumed_align=16), cutlass.Float32(eps),
    )
    return out.view(origin_x_shape).to(origin_device)
