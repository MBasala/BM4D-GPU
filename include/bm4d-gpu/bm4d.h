// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#pragma once

#include <bm4d-gpu/parameters.h>

#include <bm4d-gpu/kernels.cuh>
#include <bm4d-gpu/stopwatch.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class BM4D {
 public:
  BM4D(bm4d_gpu::Parameters p, const std::vector<uchar> &in_noisy_volume, const int &width,
       const int &height, const int &depth)
      : params(p),
        d_params(p.to_device()),
        width(width),
        height(height),
        depth(depth),
        d_gathered4dstack(nullptr),
        d_stacks(nullptr),
        d_nstacks(nullptr),
        d_group_weights(nullptr) {
    noisy_volume = in_noisy_volume;
    size = width * height * depth;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&d_prop, device));

    twidth = std::floor((width - 1) / params.step_size + 1);
    theight = std::floor((height - 1) / params.step_size + 1);
    tdepth = std::floor((depth - 1) / params.step_size + 1);
    tsize = twidth * theight * tdepth;

    checkCudaErrors(cudaMalloc((void **)&d_stacks, sizeof(uint3float1) * (params.maxN * tsize)));
    checkCudaErrors(cudaMalloc((void **)&d_nstacks, sizeof(uint) * (tsize)));
    checkCudaErrors(cudaMemset(d_nstacks, 0, sizeof(uint) * tsize));

    // Pre-allocate the long-lived pipeline buffers and output staging.
    // Chunk-sized scratch buffers are sized later from the live VRAM budget.
    pipeline_ctx.allocate(/*groups=*/tsize, /*im_size=*/size, /*patch_size=*/params.patch_size);
    pipeline_ctx.create_stream();

    // One-time setup in constructor (not mid-pipeline):
    // 1. Bind volume to 3D texture (hardware spatial cache for block matching)
    pipeline_ctx.bind_volume_texture(noisy_volume.data(), width, height, depth);
    // 2. Load DCT constants to __constant__ memory (synchronous H2D, done once)
    pipeline_ctx.load_dct_constants();
  }

  inline ~BM4D() {
    if (d_stacks != nullptr) {
      checkCudaErrors(cudaFree(d_stacks));
    }
    if (d_nstacks != nullptr) {
      checkCudaErrors(cudaFree(d_nstacks));
    }
    if (d_gathered4dstack != nullptr) {
      // d_gathered4dstack was allocated with cudaMallocAsync — must use cudaFreeAsync
      checkCudaErrors(cudaFreeAsync(d_gathered4dstack, pipeline_ctx.stream));
      checkCudaErrors(cudaStreamSynchronize(pipeline_ctx.stream));
    }
    // d_group_weights is owned by pipeline_ctx and managed as chunk workspace.
    pipeline_ctx.destroy();
  };

  std::vector<unsigned char> run_first_step();

 private:
  // Host variables
  std::vector<uchar> noisy_volume;

  // Device variables
  float *d_gathered4dstack;
  uint3float1 *d_stacks;
  uint *d_nstacks;
  float *d_group_weights;
  int width, height, depth, size;
  int twidth, theight, tdepth, tsize;

  // Pre-allocated pipeline context (stream + texture + persistent buffers)
  PipelineContext pipeline_ctx;

  // Parameters for launching kernels
  dim3 block;
  dim3 grid;

  cudaDeviceProp d_prop;
  bm4d_gpu::Parameters params;
  bm4d_gpu::DeviceParams d_params;  // GPU-safe POD copy
};
