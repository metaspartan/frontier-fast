// Feasibility slice v2: direct global->VGPR WMMA, two 16-row stripes per wave,
// scd/scm packed as half2. Correctness reference identical to v1.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define KDIM   5120
#define NSB    (KDIM/256)
#define NSUB   (KDIM/32)
#define NCOLS  16
#define MROWS  17408

typedef __attribute__((__vector_size__(2*sizeof(int)))) int int32x2_t;
typedef __attribute__((__vector_size__(8*sizeof(int)))) int int32x8_t;

__global__ void __launch_bounds__(32, 8) wmma_direct2(
        const uint8_t * __restrict__ qs,     // [M][NSB][128]
        const __half2 * __restrict__ scdm,   // [M][NSUB] {d*sc, dmin*m}
        const int8_t  * __restrict__ y,      // [NCOLS][KDIM]
        const float   * __restrict__ d8,     // [NCOLS][NSUB]
        const float   * __restrict__ su,     // [NCOLS][NSUB]
        float         * __restrict__ out,    // [M][NCOLS]
        int M) {
    const int lane = threadIdx.x;
    const int rl   = lane & 15;              // A-fragment row within stripe; also output COLUMN (C transposed)
    const int hi   = lane >> 4;
    const int r0   = blockIdx.x * 32;        // two stripes
    const uint8_t * qrow0 = qs + (size_t)(r0 + rl)      * NSB * 128;
    const uint8_t * qrow1 = qs + (size_t)(r0 + 16 + rl) * NSB * 128;
    const float   * d8c   = d8 + (size_t)rl * NSUB;
    const float   * suc   = su + (size_t)rl * NSUB;

    float facc[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) facc[e] = 0.f;

    for (int sb = 0; sb < NSB; ++sb) {
        const uint8_t * q0 = qrow0 + sb*128;
        const uint8_t * q1 = qrow1 + sb*128;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            int x0[4], x1[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                x0[t] = *(const int *)(q0 + 32*g + 16*hi + 4*t);
                x1[t] = *(const int *)(q1 + 32*g + 16*hi + 4*t);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int s = sb*8 + 2*g + h;
                const int8_t * yc = y + (size_t)rl * KDIM + (size_t)s * 32 + 16*hi;
                int b0 = *(const int *)(yc + 0);
                int b1 = *(const int *)(yc + 4);
                int b2 = *(const int *)(yc + 8);
                int b3 = *(const int *)(yc + 12);
                int32x2_t bv0 = { b0, b1 };
                int32x2_t bv1 = { b2, b3 };

                const float fd8 = d8c[s];
                const float fsu = suc[s];

#pragma unroll
                for (int st = 0; st < 2; ++st) {          // stripe
                    const int * xx = st ? x1 : x0;
                    int a0 = h ? ((xx[0]>>4)&0x0F0F0F0F) : (xx[0]&0x0F0F0F0F);
                    int a1 = h ? ((xx[1]>>4)&0x0F0F0F0F) : (xx[1]&0x0F0F0F0F);
                    int a2 = h ? ((xx[2]>>4)&0x0F0F0F0F) : (xx[2]&0x0F0F0F0F);
                    int a3 = h ? ((xx[3]>>4)&0x0F0F0F0F) : (xx[3]&0x0F0F0F0F);
                    int32x2_t av0 = { a0, a1 };
                    int32x2_t av1 = { a2, a3 };
                    int32x8_t acc = {0,0,0,0,0,0,0,0};
                    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av0, true, bv0, acc, true);
                    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av1, true, bv1, acc, true);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int rr = r0 + 16*st + 8*hi + e;   // output row (C transposed)
                        const float2 dm = __half22float2(scdm[(size_t)rr*NSUB + s]);
                        facc[8*st + e] += dm.x * fd8 * (float)acc[e] - dm.y * fsu;
                    }
                }
            }
        }
    }
#pragma unroll
    for (int st = 0; st < 2; ++st)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            out[(size_t)(r0 + 16*st + 8*hi + e) * NCOLS + rl] = facc[8*st + e];
}

__global__ void ref_kernel(
        const uint8_t * __restrict__ qs, const __half2 * __restrict__ scdm,
        const int8_t * __restrict__ y, const float * __restrict__ d8, const float * __restrict__ su,
        float * __restrict__ out, int M) {
    const int row = blockIdx.x;
    const int j   = threadIdx.x;
    if (row >= M || j >= NCOLS) return;
    const uint8_t * qrow = qs + (size_t)row * NSB * 128;
    float f = 0.f;
    for (int sb = 0; sb < NSB; ++sb) {
        const uint8_t * qsb = qrow + sb*128;
        for (int g = 0; g < 4; ++g) {
            for (int h = 0; h < 2; ++h) {
                const int s = sb*8 + 2*g + h;
                int dot = 0;
                for (int t = 0; t < 8; ++t) {
                    int xw = *(const int *)(qsb + 32*g + 4*t);
                    int aw = h ? ((xw>>4) & 0x0F0F0F0F) : (xw & 0x0F0F0F0F);
                    int bw = *(const int *)(y + (size_t)j*KDIM + (size_t)s*32 + 4*t);
                    dot = __builtin_amdgcn_sudot4(true, aw, true, bw, dot, false);
                }
                const float2 dm = __half22float2(scdm[(size_t)row*NSUB + s]);
                f += dm.x * d8[(size_t)j*NSUB + s] * (float)dot
                   - dm.y * su[(size_t)j*NSUB + s];
            }
        }
    }
    out[(size_t)row * NCOLS + j] = f;
}

#define HIP_CHECK(x) do { hipError_t e = (x); if (e) { printf("HIP error %d line %d\n", (int)e, __LINE__); exit(1);} } while (0)

int main() {
    srand(12345);
    const size_t qs_bytes = (size_t)MROWS * NSB * 128;
    const size_t sc_n     = (size_t)MROWS * NSUB;
    std::vector<uint8_t> h_qs(qs_bytes);
    for (auto & b : h_qs) b = rand() & 0xFF;
    std::vector<__half2> h_scdm(sc_n);
    for (size_t i = 0; i < sc_n; ++i)
        h_scdm[i] = __floats2half2_rn(0.001f*float(rand()%1000), 0.0005f*float(rand()%1000));
    std::vector<int8_t> h_y((size_t)NCOLS * KDIM);
    for (auto & v : h_y) v = (int8_t)((rand()%255) - 127);
    std::vector<float> h_d8((size_t)NCOLS*NSUB), h_su((size_t)NCOLS*NSUB);
    for (size_t i = 0; i < h_d8.size(); ++i) h_d8[i] = 0.001f*float(rand()%1000);
    for (int c = 0; c < NCOLS; ++c) for (int s = 0; s < NSUB; ++s) {
        int acc = 0;
        for (int o = 0; o < 32; ++o) acc += h_y[(size_t)c*KDIM + s*32 + o];
        h_su[(size_t)c*NSUB + s] = h_d8[(size_t)c*NSUB + s] * (float)acc;
    }
    uint8_t * d_qs; __half2 * d_scdm; int8_t * d_y; float * d_d8; float * d_su; float * d_out; float * d_ref;
    HIP_CHECK(hipMalloc(&d_qs, qs_bytes));
    HIP_CHECK(hipMalloc(&d_scdm, sc_n*sizeof(__half2)));
    HIP_CHECK(hipMalloc(&d_y, h_y.size()));
    HIP_CHECK(hipMalloc(&d_d8, h_d8.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_su, h_su.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, (size_t)MROWS*NCOLS*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ref, (size_t)MROWS*NCOLS*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_qs, h_qs.data(), qs_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scdm, h_scdm.data(), sc_n*sizeof(__half2), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), h_y.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d8, h_d8.data(), h_d8.size()*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_su, h_su.data(), h_su.size()*sizeof(float), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(wmma_direct2, dim3(MROWS/32), dim3(32), 0, 0, d_qs, d_scdm, d_y, d_d8, d_su, d_out, MROWS);
    HIP_CHECK(hipDeviceSynchronize());
    hipLaunchKernelGGL(ref_kernel, dim3(MROWS), dim3(NCOLS), 0, 0, d_qs, d_scdm, d_y, d_d8, d_su, d_ref, MROWS);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> h_out((size_t)MROWS*NCOLS), h_ref((size_t)MROWS*NCOLS);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_out.size()*sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_ref.data(), d_ref, h_ref.size()*sizeof(float), hipMemcpyDeviceToHost));
    size_t mism = 0;
    for (size_t i = 0; i < h_out.size(); ++i) if (h_out[i] != h_ref[i]) ++mism;
    printf("CORRECTNESS: %zu/%zu mismatches\n", mism, h_out.size());

    const int reps = 50;
    hipEvent_t t0, t1; (void)hipEventCreate(&t0); (void)hipEventCreate(&t1);
    hipLaunchKernelGGL(wmma_direct2, dim3(MROWS/32), dim3(32), 0, 0, d_qs, d_scdm, d_y, d_d8, d_su, d_out, MROWS);
    HIP_CHECK(hipDeviceSynchronize());
    (void)hipEventRecord(t0);
    for (int i = 0; i < reps; ++i)
        hipLaunchKernelGGL(wmma_direct2, dim3(MROWS/32), dim3(32), 0, 0, d_qs, d_scdm, d_y, d_d8, d_su, d_out, MROWS);
    (void)hipEventRecord(t1);
    HIP_CHECK(hipEventSynchronize(t1));
    float ms; (void)hipEventElapsedTime(&ms, t0, t1);
    const double per = ms/reps;
    const double stream = (double)(qs_bytes + sc_n*sizeof(__half2));
    printf("TIME: %.3f ms/rep | qs GB/s %.1f | total-stream GB/s %.1f\n",
           per, qs_bytes/per/1e6, stream/per/1e6);
    return 0;
}
