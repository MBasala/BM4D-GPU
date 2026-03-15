// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cstddef>
#include <cassert>
#include <iostream>
#include <vector>

#include "helper_cuda.h"
#include "parameters.h"
#include "stdio.h"

#ifndef idx3
#define idx3(x, y, z, x_size, y_size) ((x) + ((y) + (y_size) * (z)) * (x_size))
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

struct uint3float1 {
  uint x;
  uint y;
  uint z;
  float val;
  __host__ __device__ uint3float1() : x(0), y(0), z(0), val(-1) {};
  __host__ __device__ uint3float1(uint x, uint y, uint z, float val) : x(x), y(y), z(z), val(val) {}
};
inline uint3float1 make_uint3float1(uint x, uint y, uint z, float val) {
  return {x, y, z, val};
}
inline uint3float1 make_uint3float1(uint3 c, float val) { return {c.x, c.y, c.z, val}; }

// ---------------------------------------------------------------------------
// Pipeline context: pre-allocated GPU buffers + CUDA texture for the volume
//
// Everything that can be allocated upfront IS allocated here, so the pipeline
// kernels never call cudaMalloc/cudaFree (which are synchronous GPU stalls).
// ---------------------------------------------------------------------------
struct PipelineContext {
  // Accumulated nstacks (prefix-sum buffer) — reused by WHT and aggregation
  uint *d_accumulated_nstacks = nullptr;
  // Denoised / weights volumes for aggregation
  float *d_denoised_volume = nullptr;
  float *d_weights_volume = nullptr;
  // Group weights for WHT — chunk-scoped workspace sized on demand.
  float *d_group_weights = nullptr;
  std::size_t group_weight_capacity = 0;
  // GPU-side uchar output — fused normalize+convert writes here, then
  // we transfer 70MB (uchar) instead of 280MB (float) over PCIe.
  uchar *d_output_uchar = nullptr;
  // Pinned (page-locked) host output buffer — pre-allocated once so
  // cudaMallocHost doesn't stall mid-pipeline (OS page-lock is slow).
  uchar *h_pinned_output = nullptr;

  // Sizes
  int groups = 0;
  int im_size = 0;

  // CUDA stream for the entire pipeline
  cudaStream_t stream = nullptr;

  // CUDA texture object for the noisy volume
  cudaTextureObject_t tex_volume = 0;
  cudaArray_t d_volume_array = nullptr;

  void allocate(int groups_, int im_size_, int patch_size) {
    groups = groups_;
    im_size = im_size_;
    checkCudaErrors(cudaMalloc((void **)&d_accumulated_nstacks, sizeof(uint) * groups));
    checkCudaErrors(cudaMalloc((void **)&d_denoised_volume, sizeof(float) * im_size));
    checkCudaErrors(cudaMalloc((void **)&d_weights_volume, sizeof(float) * im_size));
    checkCudaErrors(cudaMalloc((void **)&d_output_uchar, sizeof(uchar) * im_size));
    checkCudaErrors(cudaMallocHost((void **)&h_pinned_output, sizeof(uchar) * im_size));
  }

  void ensure_group_weight_capacity(int groups_needed, int patch_size) {
    if (groups_needed <= 0) return;

    const std::size_t stride = static_cast<std::size_t>(patch_size) * patch_size * patch_size;
    const std::size_t required_bytes = sizeof(float) * stride * static_cast<std::size_t>(groups_needed);
    if (required_bytes <= group_weight_capacity) return;

    if (d_group_weights != nullptr) {
      checkCudaErrors(cudaFree(d_group_weights));
      d_group_weights = nullptr;
      group_weight_capacity = 0;
    }

    checkCudaErrors(cudaMalloc((void **)&d_group_weights, required_bytes));
    group_weight_capacity = required_bytes;
  }

  void create_stream() {
    checkCudaErrors(cudaStreamCreate(&stream));
  }

  /// Bind the noisy volume into a 3D CUDA texture object.
  /// Implementation is in kernels.cu (needs nvcc for CUDA API calls).
  void bind_volume_texture(const uchar *h_volume, int width, int height, int depth);

  /// Load DCT/IDCT constant-memory coefficients.  Called once in the
  /// constructor so the synchronous cudaMemcpyToSymbol doesn't happen
  /// mid-pipeline.  Implementation in kernels.cu.
  void load_dct_constants();

  void destroy() {
    if (tex_volume) { cudaDestroyTextureObject(tex_volume); tex_volume = 0; }
    if (d_volume_array) { cudaFreeArray(d_volume_array); d_volume_array = nullptr; }
    if (d_accumulated_nstacks) { cudaFree(d_accumulated_nstacks); d_accumulated_nstacks = nullptr; }
    if (d_denoised_volume) { cudaFree(d_denoised_volume); d_denoised_volume = nullptr; }
    if (d_weights_volume) { cudaFree(d_weights_volume); d_weights_volume = nullptr; }
    if (d_group_weights) { cudaFree(d_group_weights); d_group_weights = nullptr; }
    group_weight_capacity = 0;
    if (d_output_uchar) { cudaFree(d_output_uchar); d_output_uchar = nullptr; }
    if (h_pinned_output) { cudaFreeHost(h_pinned_output); h_pinned_output = nullptr; }
    if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
  }
};

// All kernel-launch functions use DeviceParams (24-byte POD) — no std::string.

void run_block_matching(cudaTextureObject_t tex_volume, const uint3 size, const uint3 tsize,
                        const bm4d_gpu::DeviceParams& params, uint3float1 *d_stacks, uint *d_nstacks,
                        const cudaDeviceProp &d_prop, cudaStream_t stream = nullptr);

void prepare_stacks_for_processing(const uint3 tsize, const bm4d_gpu::DeviceParams& params,
                                   uint3float1 *&d_stacks, uint *d_nstacks,
                                   std::vector<uint> &h_nstacks, uint &gather_stacks_sum,
                                   const cudaDeviceProp &d_prop, cudaStream_t stream = nullptr);

void gather_cubes_chunk(cudaTextureObject_t tex_volume, const uint3 size,
                        const bm4d_gpu::DeviceParams& params, const uint3float1 *d_stacks,
                        uint stack_count, float *d_gathered4dstack,
                        const cudaDeviceProp &d_prop, cudaStream_t stream = nullptr);

void run_dct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size,
               const cudaDeviceProp &d_prop, cudaStream_t stream = nullptr);

void run_wht_ht_iwht_chunk(float *d_gathered4dstack, int patch_size, uint *d_nstacks,
                           int groups, float *d_group_weights,
                           const bm4d_gpu::DeviceParams& params, const cudaDeviceProp &d_prop,
                           PipelineContext &ctx);

void run_idct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size,
                const cudaDeviceProp &d_prop, cudaStream_t stream = nullptr);

void run_aggregation_chunk(const uint3 size, int groups, const float *d_gathered4dstack,
                           uint3float1 *d_stacks, uint *d_nstacks, float *group_weights,
                           const bm4d_gpu::DeviceParams& params, const cudaDeviceProp &d_prop,
                           PipelineContext &ctx);

void finalize_aggregation(int im_size, const cudaDeviceProp &d_prop, PipelineContext &ctx);

// Helper functions
__device__ __host__ float dist(const uchar *__restrict img, const uint3 size, const int3 ref,
                               const int3 cmp, const int k);

__device__ __host__ __inline__ uint lower_power_2(uint x) {
  x = x | (x >> 1);
  x = x | (x >> 2);
  x = x | (x >> 4);
  x = x | (x >> 8);
  x = x | (x >> 16);
  return x - (x >> 1);
}

#ifndef NDEBUG
void debug_kernel(const float *tmp);
#endif
