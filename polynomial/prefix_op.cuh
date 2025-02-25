// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__SPPARK_POLYNOMIAL_PREFIX_OP_CUH__) && \
    (defined(__CUDACC__) || defined(__HIPCC__))
#define __SPPARK_POLYNOMIAL_PREFIX_OP_CUH__

#include <cassert>
#ifdef __HIPCC__
# include <hip/hip_cooperative_groups.h>
#else
# include <cooperative_groups.h>
#endif
#include <ff/shfl.cuh>

template <typename fr_t>
class Add {
public:
    using T = fr_t;
    static const int CHUNK = 192 / sizeof(fr_t);
    static_assert(CHUNK != 0, "field size is too large");

    __device__ __host__ __forceinline__
    fr_t operator()(const fr_t& a, const fr_t& b) const
    {   return a + b;   }

    __device__ __host__ __forceinline__
    fr_t identity() const
    {   fr_t ret; ret.zero(); return ret;   }
};

template <typename fr_t>
class Multiply {
public:
    using T = fr_t;
    static const int CHUNK = 128 / sizeof(fr_t);
    static_assert(CHUNK != 0, "field size is too large");

    __device__ __host__ __forceinline__
    fr_t operator()(const fr_t& a, const fr_t& b) const
    {   return a * b;   }

    __device__ __host__ __forceinline__
    fr_t identity() const
    {   return fr_t::one();   }
};

template <typename Operation, int CHUNK, typename fr_t = typename Operation::T,
          class OutPtr = fr_t*, class InPtr = const fr_t*>
__global__ __launch_bounds__(sizeof(fr_t)<=16 ? 1024 : 512, 2)
void d_prefix_op(OutPtr out, InPtr inp, size_t len)
{
    struct warp {
        __device__ __forceinline__
        static fr_t& prefix_op(fr_t& x_lane, uint32_t limit = WARP_SZ)
        {
            const uint32_t laneid = threadIdx.x % WARP_SZ;
            const Operation op;

            __builtin_assume(limit > 1);

            #pragma unroll 1
            for (uint32_t offset = 1; offset < limit; offset <<= 1) {
                fr_t temp = shfl_up(x_lane, offset);
                temp = op(temp, x_lane);
                x_lane = fr_t::csel(x_lane, temp, laneid < offset);
            }

            return x_lane;
        }

        /*
         * Compiler apparently doesn't fuse loops, at least not these ones,
         * it has to be done "manually."
         */
        __device__ __forceinline__
        static void prefix_op(fr_t chunk[CHUNK])
        {
            const uint32_t laneid = threadIdx.x % WARP_SZ;
            const Operation op;

            #pragma unroll 1
            for (uint32_t offset = 1; offset < WARP_SZ; offset <<= 1) {
                #pragma unroll
                for (int i = 0; i < CHUNK; i++) {
                    fr_t temp = shfl_up(chunk[i], offset);
                    temp = op(temp, chunk[i]);
                    chunk[i] = fr_t::csel(chunk[i], temp, laneid < offset);
                }
            }
        }
#ifdef __HIPCC__
        static __device__ __noinline__ void noop() { asm(""); }
#else
        static __device__ __forceinline__ void noop() {}
#endif
    };

#if 0
    assert(blockDim.x%WARP_SZ == 0 && gridDim.x <= blockDim.x);
#endif

    const uint32_t tid    = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t warpid = threadIdx.x / WARP_SZ;
    const uint32_t laneid = threadIdx.x % WARP_SZ;
    const uint32_t nwarps = blockDim.x  / WARP_SZ;

    static __shared__ fr_t xchg[1024/WARP_SZ]; // 1024 is maximum for blockDim

    const uint32_t chunk_size = blockDim.x * CHUNK;
    const uint32_t blob_size = gridDim.x * chunk_size;

    constexpr bool coalesce = CHUNK/sizeof(fr_t) > 1;
    const uint32_t lane_off = (tid / WARP_SZ) * WARP_SZ * CHUNK +
                              (coalesce ? laneid : laneid*CHUNK);

    const Operation op;
    const fr_t identity = op.identity();

    fr_t grid_carry = identity;
    fr_t chunk[CHUNK];

    const bool do_prefetch = true;
    fr_t prefetch;

    if (do_prefetch) {
#ifdef __CUDA_ARCH__
        prefetch = identity;
        if (lane_off < len)
            prefetch = inp[lane_off];
#else
        prefetch = inp[lane_off<len ? lane_off : len-1];
#endif
    }

    __builtin_assume(len > 0);

    #pragma unroll 1
    for (size_t blob = 0; blob < len; blob += blob_size) {
        size_t lane_idx = blob + lane_off;
        int top = CHUNK;

        if (do_prefetch) {
            if (lane_idx >= len)
                top = 0;

            chunk[0] = prefetch;
        }

        #pragma unroll
        for (int i = do_prefetch; i < CHUNK; i++) {
            size_t idx = lane_idx + (coalesce ? WARP_SZ*i : i);

            if (top == CHUNK && idx >= len)
                top = i;

#ifdef __CUDA_ARCH__
            chunk[i] = identity;
            if (i < top)
                chunk[i] = inp[idx];
#else
            chunk[i] = inp[i<top ? idx : len-1];
#endif
        }

        if (do_prefetch) {
            size_t idx = lane_idx + blob_size;

#ifdef __CUDA_ARCH__
            prefetch = identity;
            if (idx < len)
                prefetch = inp[idx];
#else
            prefetch = inp[idx<len ? idx : len-1];
#endif
        }

#ifndef __CUDA_ARCH__
        #pragma unroll
        for (int i = 0; i < CHUNK; i++)
            chunk[i] = fr_t::csel(chunk[i], identity, i < top);
#endif

        if (coalesce) {
            warp::prefix_op(chunk);

            #pragma unroll
            for (int i = 1; i < CHUNK; i++)
                chunk[i] = op(chunk[i], shfl_idx(chunk[i-1], WARP_SZ-1));
        } else {
            #pragma unroll
            for (int i = 1; i < CHUNK; i++)
                chunk[i] = op(chunk[i], chunk[i-1]);

            chunk[CHUNK-1] = warp::prefix_op(chunk[CHUNK-1]);
        }

        if (laneid == WARP_SZ-1 && warpid < 1024/WARP_SZ-1)
            xchg[warpid] = chunk[CHUNK - 1];

        __syncthreads();

        fr_t warp_carry = identity;
        if (warpid == 0) {
            fr_t carry = identity;
            if (laneid < nwarps-1)
                carry = xchg[laneid];

            warp_carry = warp::prefix_op(carry);

            if (laneid < nwarps-1)
                xchg[laneid] = carry;
        }

        __syncthreads();

        if (warpid != 0) {
            if (coalesce) {
                fr_t carry = xchg[warpid-1];
                #pragma unroll
                for (int i = 0; i < CHUNK; i++)
                    chunk[i] = op(chunk[i], carry);
            } else {
                auto carry = coalesce ? xchg[warpid-1]
                                      : chunk[CHUNK-1] = op(chunk[CHUNK-1],
                                                            xchg[warpid-1]);
                #pragma unroll
                for (int i = 0; i < CHUNK-1; i++)
                    chunk[i] = op(chunk[i], carry);
            }
        }

        fr_t (&block_carry) = xchg[nwarps - 1];

        /* We are running cooperatively, which effectively means
         * gridDim.x <= #SMs, which is few dozens of blocks.
         */
        if (gridDim.x > 1) {
            cooperative_groups::this_grid().sync();

            if (blockIdx.x == 0) {
                fr_t carry = identity;
                /* This code path is executed by just a few dozen threads,
                 * so there is no need to make laneid==0 prefetch the carry.
                 */
                if (tid < gridDim.x-1) {
                    size_t blocks_chunk = blob / chunk_size;
                    size_t block_id = blocks_chunk * gridDim.x + tid + 1;
                    if (block_id < gridDim.x)
                        carry = out[block_id * chunk_size - 1];
                }

                const uint32_t limit = (gridDim.x + WARP_SZ - 1) / WARP_SZ;
                carry = warp::prefix_op(carry, limit * WARP_SZ);

                if (tid < gridDim.x-1) {
                    size_t blocks_chunk = blob / chunk_size;
                    size_t block_id = blocks_chunk * gridDim.x + tid + 1;
                    if (block_id < gridDim.x)
                        out[block_id * chunk_size - 1] = carry;
                }
            }

            cooperative_groups::this_grid().sync();

            if (blockIdx.x != 0) {
                size_t idx = blockIdx.x * chunk_size - 1;
                if (idx < len && blob == 0) {
                    if (threadIdx.x == 0)
                        grid_carry = out[idx];
                } else {
                    size_t blocks_chunk = blob / chunk_size;
                    size_t block_id = blocks_chunk * gridDim.x + blockIdx.x;
                    if (block_id > 0 && block_id < gridDim.x) {
                        if (threadIdx.x == 0)
                            grid_carry = out[block_id * chunk_size - 1];
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < CHUNK; i++) {
            size_t idx = lane_idx + (coalesce ? WARP_SZ*i : i);
            if (idx < len)
                out[idx] = op(chunk[i], grid_carry);
        }

        if (blob + chunk_size < len && threadIdx.x == 0 && blockIdx.x == gridDim.x-1) {
            out[blob + chunk_size - 1] = op(block_carry, grid_carry);
        }

        cooperative_groups::this_grid().sync();

        if (blockIdx.x == gridDim.x-1 && warpid == nwarps-1 &&
            laneid == WARP_SZ-1 && blob + blob_size < len)
            grid_carry = op(block_carry, grid_carry);
    }
}

template <typename Operation, typename fr_t = typename Operation::T,
          class OutPtr = fr_t*, class InPtr = const fr_t*>
void prefix_op(OutPtr out, InPtr inp, size_t len, cudaStream_t stream = 0)
{
    const int CHUNK = Operation::CHUNK;
    const float bytes_per_ms = 25000;  /* observed bandwidth */
    const float performance = bytes_per_ms*1e-6 / sizeof(fr_t);
    const float time = len / (256 * performance);
    int gridDim;
    if (time < 1.0)
        gridDim = 1;
    else if (time < 2.5)
        gridDim = 2;
    else if (time < 5.0)
        gridDim = 4;
    else if (time < 10.0)
        gridDim = 8;
    else if (time < 20.0)
        gridDim = 16;
    else
        gridDim = 32;

    int tpb = sizeof(fr_t) <= 16 ? 1024 : 512;
    if (tpb > len)
        tpb = (len + 31) / 32 * 32;
    else if (tpb*CHUNK > len)
        tpb /= 2;

    size_t smem = 0;

    dim3 block(tpb);
    dim3 grid(gridDim);

    void* kernel = (void*)d_prefix_op<Operation, CHUNK, fr_t, OutPtr, InPtr>;
    void* args[] = {&out, &inp, &len};

    cudaLaunchCooperativeKernel(kernel, grid, block, args, smem, stream);
}
#endif
