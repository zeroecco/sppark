// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__SPPARK_POLYNOMIAL_DIVISION_CUH__) && \
    (defined(__CUDACC__) || defined(__HIPCC__))
#define __SPPARK_POLYNOMIAL_DIVISION_CUH__

#include <cassert>
#ifdef __HIPCC__
# include <hip/hip_cooperative_groups.h>
#else
# include <cooperative_groups.h>
#endif
#include <ff/shfl.cuh>

template<class fr_t, int N, bool rotate, int BSZ>
__global__ __launch_bounds__(BSZ, 2)
void d_div_by_x_minus_z(fr_t d_inout[], size_t len, fr_t z)
{
    struct my {
        __device__ __forceinline__
        static void madd_up(fr_t& coeff, fr_t& z_pow, uint32_t limit = WARP_SZ)
        {
            const uint32_t laneid = threadIdx.x % WARP_SZ;

            __builtin_assume(limit > 1);

            #pragma unroll 1
            for (uint32_t off = 1; off < limit; off <<= 1) {
                auto temp = shfl_up(coeff, off);
                temp = fr_t::csel(temp, z_pow, laneid != 0);
                z_pow *= temp;          // 0th lane squares z_pow
                temp = coeff + z_pow;
                coeff = fr_t::csel(coeff, temp, laneid < off);
                z_pow = shfl_idx(z_pow, 0);
            }
            /* beware that resulting |z_pow| can be fed to the next madd_up() */
        }

        __device__ __forceinline__
        static fr_t mult_up(fr_t z_lane, uint32_t limit = WARP_SZ)
        {
            const uint32_t laneid = threadIdx.x % WARP_SZ;

            __builtin_assume(limit > 1);

            #pragma unroll 1
            for (uint32_t off = 1; off < limit; off <<= 1) {
                auto temp = shfl_up(z_lane, off);
                temp *= z_lane;
                z_lane = fr_t::csel(z_lane, temp, laneid < off);
            }

            return z_lane;
        }
    };

#if 0
    assert(blockDim.x%WARP_SZ == 0 && gridDim.x <= blockDim.x);
#endif

    const uint32_t tidx   = N * (threadIdx.x + blockDim.x*blockIdx.x);
    const uint32_t laneid = threadIdx.x % WARP_SZ;
    const uint32_t warpid = threadIdx.x / WARP_SZ;
    const uint32_t nwarps = blockDim.x  / WARP_SZ;

    extern __shared__ int xchg_div_by_x_minus_z[];
    fr_t* xchg = reinterpret_cast<decltype(xchg)>(xchg_div_by_x_minus_z);
    static __shared__ fr_t z_pow_carry[WARP_SZ], z_top_block, z_top_carry, z_n;

    /*
     * Calculate ascending powers of |z| in ascending threads across
     * the grid. Since the kernel is invoked cooperatively, gridDim.x
     * would be not larger than the amount of SMs, which would be far
     * from the limit for this part of the implementation, 33*32+1.
     * ["This part" refers to the fact that a stricter limitation is
     * implied elsewhere, gridDim.x <= blockDim.x.]
     */
    fr_t z_pow = z;
    if (N > 1)
        z_pow ^= N;
    z_pow = my::mult_up(z_pow);
    fr_t z_pow_warp = z_pow;        // z^(laneid+1)

    fr_t z_pow_block = z_pow_warp;  // z^(threadIdx.x+1)
    z_pow = shfl_idx(z_pow, WARP_SZ-1);
    z_pow = my::mult_up(z_pow, nwarps);
    if (warpid != 0) {
        z_pow_block = shfl_idx(z_pow, warpid - 1);
        z_pow_block *= z_pow_warp;
    }
    z_pow = shfl_idx(z_pow, nwarps - 1);

    if (threadIdx.x == 0) {
        z_n = z_pow_warp;
        z_top_block = z_pow;
    }

    fr_t z_pow_grid = z_pow_block;  // z^(blockDim.x*blockIdx.x+threadIdx.x+1)
    if (blockIdx.x != 0) {
        z_pow = my::mult_up(z_pow, min(WARP_SZ, gridDim.x));
        z_pow_grid = shfl_idx(z_pow, (blockIdx.x - 1)%WARP_SZ);

        // Offload z^(z_top_block*(laneid+1)) to the shared memory to
        // alleviate register pressure.
        if (warpid == 0)
            z_pow_carry[laneid] = z_pow;

        if (blockIdx.x > WARP_SZ) {
            z_pow = shfl_idx(z_pow, WARP_SZ - 1);
            z_pow = my::mult_up(z_pow, (gridDim.x + WARP_SZ - 1)/WARP_SZ);
            z_pow = shfl_idx(z_pow, (blockIdx.x - 1)/WARP_SZ - 1);
            z_pow_grid *= z_pow;
        }

        if (threadIdx.x == 0)
            z_top_carry = z_pow_grid;

        z_pow_grid *= z_pow_block;
    }

    __syncthreads();

#if 0
    auto check = z^(tidx+N);
    check -= z_pow_grid;
    assert(check.is_zero());
#endif

    /*
     * Given ∑cᵢ⋅xⁱ the goal is to sum up columns as following
     *
     * cf ce cd cc cb ca a9 c8 c7 c6 c5 c4 c3 c2 c1 c0
     *    cf ce cd cc cb ca a9 c8 c7 c6 c5 c4 c3 c2 c1 * z
     *       cf ce cd cc cb ca a9 c8 c7 c6 c5 c4 c3 c2 * z^2
     *          cf ce cd cc cb ca a9 c8 c7 c6 c5 c4 c3 * z^3
     *             cf ce cd cc cb ca a9 c8 c7 c6 c5 c4 * z^4
     *                cf ce cd cc cb ca a9 c8 c7 c6 c5 * z^5
     *                   cf ce cd cc cb ca a9 c8 c7 c6 * z^6
     *                      cf ce cd cc cb ca a9 c8 c7 * z^7
     *                         cf ce cd cc cb ca a9 c8 * z^8
     *                            cf ce cd cc cb ca a9 * z^9
     *                               cf ce cd cc cb ca * z^10
     *                                  cf ce cd cc cb * z^11
     *                                     cf ce cd cc * z^12
     *                                        cf ce cd * z^13
     *                                           cf ce * z^14
     *                                              cf * z^15
     *
     * If |rotate| is false, the first element of the output is
     * the remainder and the rest is the quotient. Otherwise
     * the remainder is stored at the end and the quotient is
     * "shifted" toward the beginning of the |d_inout| vector.
     */
    class rev_ptr_t {
        fr_t* p;
    public:
        __device__ rev_ptr_t(fr_t* ptr, size_t len) : p(ptr + len - 1) {}
        __device__ fr_t& operator[](size_t i)             { return *(p - i); }
        __device__ const fr_t& operator[](size_t i) const { return *(p - i); }
    };
    rev_ptr_t inout{d_inout, len};
    fr_t coeff[N], prefetch;
    uint32_t stride = N*blockDim.x*gridDim.x;
    size_t idx;
    auto __grid = cooperative_groups::this_grid();

    if (tidx < len)
        prefetch = inout[tidx];

    for (size_t chunk = 0; chunk < len; chunk += stride) {
        idx = chunk + tidx;

        #pragma unroll
        for (int i = 1; i < N; i++) {
            if (idx + i < len)
                coeff[i] = inout[idx + i];
        }
        coeff[0] = prefetch;

        if (idx + stride < len)
            prefetch = inout[idx + stride];

        z_pow = z;
        #pragma unroll
        for (int i = 1; i < N; i++)
            coeff[i] += coeff[i-1] * z_pow;

        fr_t carry_over;
        bool tail_sync = false;

        if (sizeof(fr_t) <= 32) {
            my::madd_up(coeff[N-1], z_pow = z_n);

            if (laneid == WARP_SZ-1)
                xchg[warpid] = coeff[N-1];

            __syncthreads();

            carry_over = xchg[laneid];

            my::madd_up(carry_over, z_pow, nwarps);

            if (laneid == nwarps-1)
                xchg[0] = carry_over;

            // Is this thread the designated "synchronizer" for the whole grid?
            tail_sync = blockIdx.x == 0 && threadIdx.x == 0;

            __syncthreads();
            carry_over = xchg[0];
        } else {
            struct {
                __device__ __forceinline__
                static void step(fr_t& carry, fr_t& zpow, fr_t& xchg0,
                                 fr_t& z_top_carry, fr_t* z_pow_carry)
                {
                    const uint32_t laneid = threadIdx.x % WARP_SZ;
                    const uint32_t warpid = threadIdx.x / WARP_SZ;

                    fr_t zpow_temp = zpow;
                    fr_t carry_temp;
                    if (blockIdx.x == 0) {
                        if (laneid == 0)
                            carry_temp = carry;
                        else
                            carry_temp = zpow_temp = fr_t::zero();
                    } else {
                        if (blockIdx.x <= WARP_SZ && warpid == 0) {
                            zpow_temp = z_pow_carry[blockIdx.x-1];
                            if (laneid == 0)
                                carry_temp = xchg0;
                            else
                                carry_temp = fr_t::zero();
                        } else if (blockIdx.x > WARP_SZ) {
                            if (warpid == 0) {
                                zpow_temp = z_pow_carry[WARP_SZ-1];
                                if (blockIdx.x/WARP_SZ * WARP_SZ +
                                    1 + laneid == blockIdx.x) {
                                    zpow_temp *= z_top_carry;
                                    zpow = zpow_temp;
                                }
                            }
                            if (laneid == 0)
                                carry_temp = xchg0;
                            else
                                carry_temp = fr_t::zero();
                        }
                    }
                    carry_temp = shfl_idx(carry_temp, 0);
                    zpow_temp *= carry_temp;
                    carry += zpow_temp;
                }
            };

            // Sync the result of the thread block.
            if (laneid == 0)
                xchg[warpid] = coeff[N-1];

            __syncthreads();

            tail_sync = blockIdx.x == 0 && threadIdx.x == 0;

            // Only one thread per thread block continues.
            if (threadIdx.x == 0) {
                carry_over = xchg[0];
                for (uint32_t j = 1; j < nwarps; j++)
                    carry_over += xchg[j] * (z_n^j);

                xchg[0] = carry_over;
            }

            __syncthreads();
            carry_over = xchg[0];
        }

        // Bring all blocks in line...
        __grid.sync();

        // Store the result.
        if (!rotate) {
            if (tidx == 0) {
                inout[0] = carry_over;
            } else if (idx <= len-N) {
                #pragma unroll
                for (int i = 0; i < N; i++)
                    inout[idx + i] = coeff[i];
            }
        } else if (idx <= len-N) {
            #pragma unroll
            for (int i = 0; i < N; i++)
                inout[idx + i - (idx == 0 ? 0 : 1)] = coeff[i];
            if (idx == 0)
                inout[len-1] = carry_over;
        }

        __grid.sync();
    }
}

template<bool rotate = false, class fr_t, class stream_t>
void div_by_x_minus_z(fr_t d_inout[], size_t len, const fr_t& z,
                      const stream_t& s, int gridDim = 0)
{
    if (gridDim <= 0)
        gridDim = s.sm_count();

    constexpr int N = 2;
    constexpr int BSZ = sizeof(fr_t) <= 16 ? 1024 : 0;
    int blockDim = BSZ;

    if (BSZ == 0) {
        static int saved_blockDim = 0;

        if (saved_blockDim == 0) {
            cudaFuncAttributes attr;
            CUDA_OK(cudaFuncGetAttributes(&attr, d_div_by_x_minus_z<fr_t, N, rotate, BSZ>));
            saved_blockDim = attr.maxThreadsPerBlock;
            assert(saved_blockDim%WARP_SZ == 0);
        }

        blockDim = saved_blockDim;
    }

    if (gridDim > blockDim) // there are no such large GPUs, not for now...
        gridDim = blockDim;

    size_t blocks = (len + blockDim - 1)/blockDim;

    if ((unsigned)gridDim > blocks)
        gridDim = (int)blocks;

    if (gridDim < 3)
        gridDim = 1;

    size_t sharedSz = sizeof(fr_t) * max(blockDim/WARP_SZ, gridDim);

    s.launch_coop(d_div_by_x_minus_z<fr_t, N, rotate, BSZ>,
                  {gridDim, blockDim, sharedSz},
                  d_inout, len, z);
}
#endif
