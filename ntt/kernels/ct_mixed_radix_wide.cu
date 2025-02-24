// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

template <int intermediate_mul, class fr_t>
__launch_bounds__(768, 2) __global__
void _CT_NTT(const unsigned int radix, const unsigned int lg_domain_size,
             const unsigned int stage, const unsigned int iterations,
             fr_t* d_inout, const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
             const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
             const fr_t* d_intermediate_twiddles,
             const unsigned int intermediate_twiddle_shift,
             const bool is_intt, const fr_t d_domain_size_inverse)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11 || defined(__clang__)
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= 10);
    __builtin_assume(iterations <= radix);
    __builtin_assume(stage <= lg_domain_size - iterations);
#endif

    const index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t out_mask = ((index_t)1 << (stage + iterations - 1)) - 1;
#if 1
    const index_t thread_ntt_pos = (tid & out_mask) >> (iterations - 1);
#else
    const index_t inp_mask = ((index_t)1 << stage) - 1;
    const index_t thread_ntt_pos = (tid >> (iterations - 1)) & inp_mask;
#endif

    // rearrange |tid|'s bits
    index_t idx0 = (tid & ~out_mask) | ((tid << stage) & out_mask);
    idx0 = idx0 * 2 + thread_ntt_pos;
    index_t idx1 = idx0 + ((index_t)1 << stage);

    fr_t r0 = d_inout[idx0];
    fr_t r1 = d_inout[idx1];

    if (intermediate_mul == 1) {
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - stage;

        index_t root_idx0 = bit_rev(thread_ntt_idx, nbits) * thread_ntt_pos;
        index_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        fr_t first_root, second_root;
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);

        r0 *= first_root;
        r1 *= second_root;
    } else if (intermediate_mul == 2) {
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int root_idx = (tid & diff_mask) * 2;
        index_t root_pos = thread_ntt_pos << intermediate_twiddle_shift;

        fr_t t0 = d_intermediate_twiddles[root_pos + root_idx];
        fr_t t1 = d_intermediate_twiddles[root_pos + root_idx + 1];

        r0 *= t0;
        r1 *= t1;
    }

    {
        fr_t t = r1;
        r1 = r0 - t;
        r0 = r0 + t;
    }

    for (unsigned int s = 1; s < min(iterations, 6u); s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t tw = d_radix6_twiddles[rank << (6 - (s + 1))];

        fr_t x = fr_t::csel(r1, r0, pos);
        x.shfl_bfly(laneMask);
        r0 = fr_t::csel(r0, x, pos);
        r1 = fr_t::csel(x, r1, pos);

        fr_t t = tw * r1;

        r1 = r0 - t;
        r0 = r0 + t;
    }

    for (unsigned int s = 6; s < iterations; s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t tw = d_radixX_twiddles[rank << (radix - (s + 1))];

        extern __shared__ fr_t shared_exchange[];

        fr_t x = fr_t::csel(r1, r0, pos);
        __syncthreads();
        shared_exchange[threadIdx.x] = x;
        __syncthreads();
        x = shared_exchange[threadIdx.x ^ laneMask];
        r0 = fr_t::csel(r0, x, pos);
        r1 = fr_t::csel(x, r1, pos);

        fr_t t = tw * r1;

        r1 = r0 - t;
        r0 = r0 + t;
    }

    if (is_intt && (stage + iterations) == lg_domain_size) {
        r0 *= d_domain_size_inverse;
        r1 *= d_domain_size_inverse;
    }

    index_t mask = (index_t)((1 << iterations) - 1) << stage;
    index_t rotw0 = idx0 & mask;
    index_t rotw1 = idx1 & mask;

    rotw0 = (rotw0 >> 1) | (rotw0 << (iterations - 1));
    rotw1 = (rotw1 >> 1) | (rotw1 << (iterations - 1));

    idx0 = (idx0 & ~mask) | (rotw0 & mask);
    idx1 = (idx1 & ~mask) | (rotw1 & mask);

    d_inout[idx0] = r0;
    d_inout[idx1] = r1;
}

class CT_launcher {
    fr_t* d_inout;
    const int lg_domain_size;
    bool is_intt;
    int stage;
    const NTTParameters& ntt_parameters;
    const stream_t& stream;
    int min_radix;

public:
    CT_launcher(fr_t* d_ptr, int lg_dsz, bool intt,
                const NTTParameters& params, const stream_t& s)
      : d_inout(d_ptr), lg_domain_size(lg_dsz), is_intt(intt), stage(0),
        ntt_parameters(params), stream(s)
    {   min_radix = lg2(gpu_props(s).warpSize) + 1;   }

    void step(int iterations)
    {
        assert(iterations <= 10);

        const int radix = iterations < min_radix ? min_radix : iterations;

        index_t num_threads = (index_t)1 << (lg_domain_size - 1);
        index_t block_size = 1 << (radix - 1);
        index_t num_blocks;

        block_size = (num_threads <= block_size) ? num_threads : block_size;
        num_blocks = (num_threads + block_size - 1) / block_size;

        assert(num_blocks == (unsigned int)num_blocks);

        fr_t* d_intermediate_twiddles = nullptr;
        unsigned int intermediate_twiddle_shift = 0;

        #define NTT_CONFIGURATION \
                num_blocks, block_size, sizeof(fr_t) * block_size, stream

        #define NTT_ARGUMENTS radix, lg_domain_size, stage, iterations, \
                d_inout, ntt_parameters.partial_twiddles, \
                ntt_parameters.twiddles[0], ntt_parameters.twiddles[radix-6], \
                d_intermediate_twiddles, intermediate_twiddle_shift, \
                is_intt, domain_size_inverse[lg_domain_size]

        switch (stage) {
        case 0:
            _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        case 6:
            if (iterations <= 6) {
                intermediate_twiddle_shift = 6;
                d_intermediate_twiddles = ntt_parameters.radix6_twiddles_6;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            } else {
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            }
            break;
        case 7:
            if (iterations <= 7) {
                intermediate_twiddle_shift = 7;
                d_intermediate_twiddles = ntt_parameters.radix7_twiddles_7;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            } else {
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            }
            break;
        case 8:
            if (iterations <= 8) {
                intermediate_twiddle_shift = 8;
                d_intermediate_twiddles = ntt_parameters.radix8_twiddles_8;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            } else {
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            }
            break;
        case 9:
            if (iterations <= 9) {
                intermediate_twiddle_shift = 9;
                d_intermediate_twiddles = ntt_parameters.radix9_twiddles_9;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            } else {
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            }
            break;
        default:
            _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
            break;
        }

        #undef NTT_CONFIGURATION
        #undef NTT_ARGUMENTS

        CUDA_OK(cudaGetLastError());

        stage += iterations;
    }
};
