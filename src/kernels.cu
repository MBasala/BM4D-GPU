// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#include <math.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/system/cuda/execution_policy.h>  // thrust::cuda::par.on(stream)

#include <bm4d-gpu/kernels.cuh>

#include "bm4d-gpu/bm4d.h"

// ---------------------------------------------------------------------------
// PipelineContext::bind_volume_texture — must live in a .cu file so nvcc
// compiles the CUDA runtime API calls (cudaMalloc3DArray, etc.)
// ---------------------------------------------------------------------------
void PipelineContext::bind_volume_texture(const uchar *h_volume, int width, int height, int depth) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
  cudaExtent extent = make_cudaExtent(width, height, depth);
  checkCudaErrors(cudaMalloc3DArray(&d_volume_array, &desc, extent));

  cudaMemcpy3DParms p = {0};
  p.srcPtr = make_cudaPitchedPtr((void *)h_volume, width * sizeof(uchar), width, height);
  p.dstArray = d_volume_array;
  p.extent = extent;
  p.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&p));

  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = d_volume_array;

  cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  checkCudaErrors(cudaCreateTextureObject(&tex_volume, &resDesc, &texDesc, nullptr));
}

// ---------------------------------------------------------------------------
// PipelineContext::load_dct_constants — loads DCT coefficients into __constant__
// memory ONCE during construction, not mid-pipeline.
// ---------------------------------------------------------------------------
__constant__ float c_dct[4][4];
__constant__ float c_dct_T[4][4];

void PipelineContext::load_dct_constants() {
  const float h_dct[4][4] = {
      {0.500000000000000f, 0.500000000000000f, 0.500000000000000f, 0.500000000000000f},
      {0.653281482438188f, 0.270598050073099f, -0.270598050073099f, -0.653281482438188f},
      {0.500000000000000f, -0.500000000000000f, -0.500000000000000f, 0.500000000000000f},
      {0.270598050073099f, -0.653281482438188f, 0.653281482438188f, -0.270598050073099f}};
  const float h_dct_T[4][4] = {
      {0.500000000000000f, 0.653281482438188f, 0.500000000000000f, 0.270598050073099f},
      {0.500000000000000f, 0.270598050073099f, -0.500000000000000f, -0.653281482438188f},
      {0.500000000000000f, -0.270598050073099f, -0.500000000000000f, 0.653281482438188f},
      {0.500000000000000f, -0.653281482438188f, 0.500000000000000f, -0.270598050073099f}};
  checkCudaErrors(cudaMemcpyToSymbol(c_dct, h_dct, sizeof(h_dct)));
  checkCudaErrors(cudaMemcpyToSymbol(c_dct_T, h_dct_T, sizeof(h_dct_T)));
}

namespace {

constexpr int kDefaultThreads1D = 256;
constexpr int kBlockMatchingBlockX = 8;
constexpr int kBlockMatchingBlockY = 8;
constexpr int kBlockMatchingBlockZ = 4;

inline int clamp_1d_blocks(const std::uint64_t work_items, const int threads,
                           const cudaDeviceProp &d_prop) {
  if (work_items == 0 || threads <= 0) return 0;
  const std::uint64_t needed_blocks =
      (work_items + static_cast<std::uint64_t>(threads) - 1) / static_cast<std::uint64_t>(threads);
  const std::uint64_t max_blocks = static_cast<std::uint64_t>(d_prop.maxGridSize[0]);
  return static_cast<int>(std::min(needed_blocks, max_blocks));
}

inline int clamp_block_work(const std::uint64_t work_items, const cudaDeviceProp &d_prop) {
  if (work_items == 0) return 0;
  const std::uint64_t max_blocks = static_cast<std::uint64_t>(d_prop.maxGridSize[0]);
  return static_cast<int>(std::min(work_items, max_blocks));
}

}  // namespace

// ---------------------------------------------------------------------------
// Debug kernels — only compiled in debug builds
// ---------------------------------------------------------------------------
#ifndef NDEBUG
__global__ void k_debug_lookup_stacks(const uint3float1 *d_stacks, int total_elements) {
  for (int i = 0; i < 150; ++i) {
    printf("%i: %d %d %d %f\n", i, d_stacks[i].x, d_stacks[i].y, d_stacks[i].z, d_stacks[i].val);
  }
}
__global__ void k_debug_lookup_4dgathered_stack(const float *gathered_stack4d) {
  for (int i = 0; i < 64 * 3; ++i) {
    if (!(i % 4)) printf("\n");
    if (!(i % 16)) printf("------------\n");
    if (!(i % 64)) printf("------------\n");
    printf("%f ", gathered_stack4d[i]);
  }
}
void debug_kernel(const float *tmp) {
  k_debug_lookup_4dgathered_stack<<<1, 1>>>(tmp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
#endif  // NDEBUG

// ---------------------------------------------------------------------------
// Stack insertion (sorted by distance, descending — worst match at index 0)
// ---------------------------------------------------------------------------
__device__ __host__ void add_stack(uint3float1 *d_stacks, uint *d_nstacks, const uint3float1 val,
                                   const int maxN) {
  int k{};
  uint num = (*d_nstacks);
  if (num < maxN) {
    k = num++;
    while (k > 0 && val.val > d_stacks[k - 1].val) {
      d_stacks[k] = d_stacks[k - 1];
      --k;
    }
    d_stacks[k] = val;
    *d_nstacks = num;
  } else if (val.val >= d_stacks[0].val) {
    return;
  } else {
    k = 1;
    while (k < maxN && val.val < d_stacks[k].val) {
      d_stacks[k - 1] = d_stacks[k];
      k++;
    }
    d_stacks[k - 1] = val;
  }
}

// ---------------------------------------------------------------------------
// Distance functions — plain global-memory version for host-side / tests
// ---------------------------------------------------------------------------
__device__ __host__ float dist(const uchar *__restrict img, const uint3 size, const int3 ref,
                               const int3 cmp, const int k) {
  const float normalizer = 1.0f / (k * k * k);
  const int3 isize = make_int3(size.x, size.y, size.z);
  float diff{0.f};
  for (int z = 0; z < k; ++z) {
    for (int y = 0; y < k; ++y) {
      for (int x = 0; x < k; ++x) {
        const int rx = max(0, min(x + ref.x, isize.x - 1));
        const int ry = max(0, min(y + ref.y, isize.y - 1));
        const int rz = max(0, min(z + ref.z, isize.z - 1));
        const int cx = max(0, min(x + cmp.x, isize.x - 1));
        const int cy = max(0, min(y + cmp.y, isize.y - 1));
        const int cz = max(0, min(z + cmp.z, isize.z - 1));
        const float tmp = (img[(rx) + (ry)*isize.x + (rz)*isize.x * isize.y] -
                           img[(cx) + (cy)*isize.x + (cz)*isize.x * isize.y]);
        diff += tmp * tmp * normalizer;
      }
    }
  }
  return diff;
}

// ---------------------------------------------------------------------------
// Texture-based distance with early exit.
// tex3D reads go through the texture cache (separate from L1/L2), boundary
// clamped in hardware.  Loops unrolled for patch_size=4.
// ---------------------------------------------------------------------------
__device__ float dist_tex_thresholded(cudaTextureObject_t tex, const int3 ref,
                                      const int3 cmp, const int k, const float sim_th) {
  const float max_diff = sim_th * static_cast<float>(k * k * k);
  const float normalizer = 1.0f / static_cast<float>(k * k * k);
  float diff_sum = 0.0f;
  #pragma unroll
  for (int z = 0; z < 4; ++z) {
    #pragma unroll
    for (int y = 0; y < 4; ++y) {
      #pragma unroll
      for (int x = 0; x < 4; ++x) {
        float r = static_cast<float>(tex3D<uchar>(tex, x + ref.x, y + ref.y, z + ref.z));
        float c = static_cast<float>(tex3D<uchar>(tex, x + cmp.x, y + cmp.y, z + cmp.z));
        float d = r - c;
        diff_sum += d * d;
        if (diff_sum > max_diff) {
          return sim_th + 1.0f;
        }
      }
    }
  }
  return diff_sum * normalizer;
}

// ---------------------------------------------------------------------------
// Block matching — texture reads + DeviceParams + __launch_bounds__
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2)
k_block_matching(cudaTextureObject_t tex, const uint3 size, const uint3 tsize,
                 const bm4d_gpu::DeviceParams params, uint3float1 *d_stacks,
                 uint *d_nstacks) {
  for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < tsize.z;
       Idz += blockDim.z * gridDim.z)
    for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < tsize.y;
         Idy += blockDim.y * gridDim.y)
      for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < tsize.x;
           Idx += blockDim.x * gridDim.x) {
        int x = Idx * params.step_size;
        int y = Idy * params.step_size;
        int z = Idz * params.step_size;
        if (x >= size.x || y >= size.y || z >= size.z) return;

        const int wxb = max(0, x - params.window_size);
        const int wyb = max(0, y - params.window_size);
        const int wzb = max(0, z - params.window_size);
        const int wxe = min(static_cast<int>(size.x) - 1, x + params.window_size);
        const int wye = min(static_cast<int>(size.y) - 1, y + params.window_size);
        const int wze = min(static_cast<int>(size.z) - 1, z + params.window_size);

        const int group_id = Idx + (Idy + Idz * tsize.y) * tsize.x;
        const int stack_offset = group_id * params.maxN;
        const int3 ref = make_int3(x, y, z);
        uint local_nstacks = 0;
        uint3float1 local_stack[16];

        for (int wz = wzb; wz <= wze; wz++)
          for (int wy = wyb; wy <= wye; wy++)
            for (int wx = wxb; wx <= wxe; wx++) {
              const float active_threshold =
                  local_nstacks >= static_cast<uint>(params.maxN)
                      ? min(params.sim_th, local_stack[0].val)
                      : params.sim_th;
              float w = dist_tex_thresholded(tex, ref, make_int3(wx, wy, wz),
                                             params.patch_size, active_threshold);
              if (w < active_threshold) {
                add_stack(local_stack, &local_nstacks, uint3float1(wx, wy, wz, w), params.maxN);
              }
            }

        d_nstacks[group_id] = local_nstacks;
        for (uint i = 0; i < local_nstacks; ++i) {
          d_stacks[stack_offset + i] = local_stack[i];
        }
      }
}

void run_block_matching(cudaTextureObject_t tex_volume, const uint3 size, const uint3 tsize,
                        const bm4d_gpu::DeviceParams& params, uint3float1 *d_stacks, uint *d_nstacks,
                        const cudaDeviceProp &d_prop, cudaStream_t stream) {
  const dim3 block(kBlockMatchingBlockX, kBlockMatchingBlockY, kBlockMatchingBlockZ);
  const dim3 grid(std::min(static_cast<unsigned int>(d_prop.maxGridSize[0]),
                           (tsize.x + block.x - 1) / block.x),
                  std::min(static_cast<unsigned int>(d_prop.maxGridSize[1]),
                           (tsize.y + block.y - 1) / block.y),
                  std::min(static_cast<unsigned int>(d_prop.maxGridSize[2]),
                           (tsize.z + block.z - 1) / block.z));

  std::cout << "Total number of reference patches " << (tsize.x * tsize.y * tsize.z) << std::endl;

  k_block_matching<<<grid, block, 0, stream>>>(tex_volume, size, tsize, params, d_stacks, d_nstacks);
  checkCudaErrors(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Nstack-to-power-of-2 + Gather cubes
// ---------------------------------------------------------------------------

__global__ void k_nstack_to_pow(uint3float1 *d_stacks, uint *d_nstacks, const uint groups,
                                const uint maxN) {
  for (uint groupId = blockIdx.x * blockDim.x + threadIdx.x; groupId < groups;
       groupId += blockDim.x * gridDim.x) {
    const uint n = d_nstacks[groupId];
    const uint normalized_n = lower_power_2(n);
    const uint diff = n - normalized_n;
    const uint group_offset = groupId * maxN;

    d_nstacks[groupId] = normalized_n;

    for (uint in_group_id = 0; in_group_id < maxN; ++in_group_id) {
      if (in_group_id < diff || in_group_id >= n) {
        d_stacks[group_offset + in_group_id].val = -1;
      }
    }
  }
}

// Gather cubes — 1 thread per stack entry, serial 4×4×4 voxel loop.
// Uses texture for hardware-cached 3D reads + boundary clamp.
__global__ void k_gather_cubes(cudaTextureObject_t tex, const uint3 size,
                               const int patch_size,
                               const uint3float1 *__restrict d_stacks,
                               const uint array_size,
                               float *d_gathered4dstack) {
  const int cube_size = patch_size * patch_size * patch_size;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < array_size;
       i += blockDim.x * gridDim.x) {
    uint3float1 ref = d_stacks[i];
    for (int z = 0; z < patch_size; ++z)
      for (int y = 0; y < patch_size; ++y)
        for (int x = 0; x < patch_size; ++x) {
          int stack_idx = i * cube_size + x + (y + z * patch_size) * patch_size;
          d_gathered4dstack[stack_idx] = static_cast<float>(
              tex3D<uchar>(tex, x + ref.x, y + ref.y, z + ref.z));
        }
  }
}

struct is_not_empty {
  __host__ __device__ bool operator()(const uint3float1 x) { return (x.val != -1); }
};

void prepare_stacks_for_processing(const uint3 tsize, const bm4d_gpu::DeviceParams& params,
                                   uint3float1 *&d_stacks, uint *d_nstacks,
                                   std::vector<uint> &h_nstacks, uint &gather_stacks_sum,
                                   const cudaDeviceProp &d_prop, cudaStream_t stream) {
  const uint array_size = (tsize.x * tsize.y * tsize.z);
  const int threads = kDefaultThreads1D;
  const int bs_x = clamp_1d_blocks(array_size, threads, d_prop);
  k_nstack_to_pow<<<bs_x, threads, 0, stream>>>(d_stacks, d_nstacks, array_size, params.maxN);
  checkCudaErrors(cudaGetLastError());

  h_nstacks.resize(array_size);
  checkCudaErrors(cudaMemcpyAsync(h_nstacks.data(), d_nstacks, sizeof(uint) * array_size,
                                  cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  gather_stacks_sum = static_cast<uint>(
      std::accumulate(h_nstacks.begin(), h_nstacks.end(), std::uint64_t{0}));

  if (gather_stacks_sum == 0) return;

  auto policy = thrust::cuda::par.on(stream);
  checkCudaErrors(cudaGetLastError());

  uint3float1 *d_stacks_compacted;
  checkCudaErrors(
      cudaMallocAsync((void **)&d_stacks_compacted, sizeof(uint3float1) * gather_stacks_sum, stream));

  thrust::device_ptr<uint3float1> dt_stacks = thrust::device_pointer_cast(d_stacks);
  thrust::device_ptr<uint3float1> dt_stacks_compacted =
      thrust::device_pointer_cast(d_stacks_compacted);
  thrust::copy_if(policy, dt_stacks, dt_stacks + params.maxN * tsize.x * tsize.y * tsize.z,
                  dt_stacks_compacted, is_not_empty());
  d_stacks_compacted = thrust::raw_pointer_cast(dt_stacks_compacted);
  uint3float1 *tmp = d_stacks;
  d_stacks = d_stacks_compacted;
  checkCudaErrors(cudaFreeAsync(tmp, stream));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(stream));
}

void gather_cubes_chunk(cudaTextureObject_t tex_volume, const uint3 size,
                        const bm4d_gpu::DeviceParams& params, const uint3float1 *d_stacks,
                        uint stack_count, float *d_gathered4dstack,
                        const cudaDeviceProp &d_prop, cudaStream_t stream) {
  const int threads = kDefaultThreads1D;
  const int gather_blocks = clamp_1d_blocks(stack_count, threads, d_prop);
  k_gather_cubes<<<gather_blocks, threads, 0, stream>>>(
      tex_volume, size, params.patch_size, d_stacks, stack_count, d_gathered4dstack);
  checkCudaErrors(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// DCT3D / IDCT3D — constant-memory coefficients (loaded once in constructor)
// ---------------------------------------------------------------------------

__global__ void dct3d(float *d_gathered4dstack, int patch_size, uint gather_stacks_sum) {
  for (uint cuIdx = blockIdx.x; cuIdx < gather_stacks_sum; cuIdx += gridDim.x) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int stride = patch_size * patch_size * patch_size;

    __shared__ float cube[4][4][4];
    int idx = (cuIdx * stride) + (x + y * patch_size + z * patch_size * patch_size);
    cube[z][y][x] = d_gathered4dstack[idx];
    __syncthreads();
    float tmp = c_dct[y][0] * cube[z][0][x] + c_dct[y][1] * cube[z][1][x] +
                c_dct[y][2] * cube[z][2][x] + c_dct[y][3] * cube[z][3][x];
    __syncthreads();
    cube[z][y][x] = tmp;
    __syncthreads();
    tmp = c_dct_T[0][x] * cube[z][y][0] + c_dct_T[1][x] * cube[z][y][1] +
          c_dct_T[2][x] * cube[z][y][2] + c_dct_T[3][x] * cube[z][y][3];
    __syncthreads();
    cube[z][y][x] = tmp;
    __syncthreads();
    float z_vec[4];
    for (int i = 0; i < 4; ++i) {
      z_vec[i] = cube[i][y][x];
    }
    __syncthreads();
    cube[z][y][x] = z_vec[0] * c_dct[z][0] + z_vec[1] * c_dct[z][1] +
                    z_vec[2] * c_dct[z][2] + z_vec[3] * c_dct[z][3];
    __syncthreads();
    d_gathered4dstack[idx] = cube[z][y][x];
  }
}

void run_dct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size,
               const cudaDeviceProp &d_prop, cudaStream_t stream) {
  // DCT constants now loaded in constructor — no mid-pipeline H2D transfer
  const int blocks = clamp_block_work(gather_stacks_sum, d_prop);
  dct3d<<<blocks, dim3(patch_size, patch_size, patch_size), 0, stream>>>(
      d_gathered4dstack, patch_size, gather_stacks_sum);
  checkCudaErrors(cudaGetLastError());
}

__global__ void idct3d(float *d_gathered4dstack, int patch_size, uint gather_stacks_sum) {
  for (uint cuIdx = blockIdx.x; cuIdx < gather_stacks_sum; cuIdx += gridDim.x) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int stride = patch_size * patch_size * patch_size;

    __shared__ float cube[4][4][4];
    int idx = (cuIdx * stride) + (x + y * patch_size + z * patch_size * patch_size);
    cube[z][y][x] = d_gathered4dstack[idx];
    __syncthreads();
    float z_vec[4];
    for (int i = 0; i < 4; ++i) {
      z_vec[i] = cube[i][y][x];
    }
    __syncthreads();
    cube[z][y][x] = z_vec[0] * c_dct_T[z][0] + z_vec[1] * c_dct_T[z][1] +
                    z_vec[2] * c_dct_T[z][2] + z_vec[3] * c_dct_T[z][3];
    __syncthreads();
    float tmp = c_dct_T[y][0] * cube[z][0][x] + c_dct_T[y][1] * cube[z][1][x] +
                c_dct_T[y][2] * cube[z][2][x] + c_dct_T[y][3] * cube[z][3][x];
    __syncthreads();
    cube[z][y][x] = tmp;
    tmp = c_dct[0][x] * cube[z][y][0] + c_dct[1][x] * cube[z][y][1] +
          c_dct[2][x] * cube[z][y][2] + c_dct[3][x] * cube[z][y][3];
    __syncthreads();
    cube[z][y][x] = tmp;
    __syncthreads();
    d_gathered4dstack[idx] = cube[z][y][x];
  }
}

void run_idct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size,
                const cudaDeviceProp &d_prop, cudaStream_t stream) {
  const int blocks = clamp_block_work(gather_stacks_sum, d_prop);
  idct3d<<<blocks, dim3(patch_size, patch_size, patch_size), 0, stream>>>(
      d_gathered4dstack, patch_size, gather_stacks_sum);
  checkCudaErrors(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// WHT + Hard Thresholding + IWHT
// ---------------------------------------------------------------------------

__device__ __host__ void whrotate(float &a, float &b) {
  float t = a;
  a = a + b;
  b = t - b;
}

__device__ __host__ long ilog2(long x) {
  long l2 = 0;
  for (; x; x >>= 1) ++l2;
  return l2;
}

__device__ __host__ void fwht(float *data, int size) {
  const long l2 = ilog2(size) - 1;
  for (long i = 0; i < l2; ++i) {
    for (long j = 0; j < (1 << l2); j += 1 << (i + 1))
      for (long k = 0; k < (1 << i); ++k) whrotate(data[j + k], data[j + k + (1 << i)]);
  }
}

__global__ void k_run_wht_ht_iwht(float *d_gathered4dstack, uint groups, int patch_size,
                                  uint *d_nstacks, uint *accumulated_nstacks,
                                  float *d_group_weights, const float hard_th) {
  for (uint cuIdx = blockIdx.x; cuIdx < groups; cuIdx += gridDim.x) {
    if (cuIdx >= groups) return;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int stride = patch_size * patch_size * patch_size;
    float group_vector[16];
    int size = d_nstacks[cuIdx];
    int group_start = accumulated_nstacks[cuIdx];

    for (int i = 0; i < size; i++) {
      long long int gl_idx =
          (group_start * stride) + (x + y * patch_size + z * patch_size * patch_size + i * stride);
      group_vector[i] = d_gathered4dstack[gl_idx];
    }

    fwht(group_vector, size);
    float threshold = hard_th * sqrtf((float)size);
    d_group_weights[cuIdx * stride + x + y * patch_size + z * patch_size * patch_size] = 0;
    for (int i = 0; i < size; i++) {
      group_vector[i] /= size;
      if (fabs(group_vector[i]) > threshold) {
        d_group_weights[cuIdx * stride + x + y * patch_size + z * patch_size * patch_size] += 1;
      } else {
        group_vector[i] = 0;
      }
    }
    fwht(group_vector, size);
    for (int i = 0; i < size; i++) {
      long long int gl_idx =
          (group_start * stride) + (x + y * patch_size + z * patch_size * patch_size + i * stride);
      d_gathered4dstack[gl_idx] = group_vector[i];
    }
  }
}

__global__ void k_sum_group_weights(float *d_group_weights, uint groups, int patch_size) {
  extern __shared__ float partial_sums[];
  const int stride = patch_size * patch_size * patch_size;
  for (uint cuIdx = blockIdx.x; cuIdx < groups; cuIdx += gridDim.x) {
    float counter = 0.0f;
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
      counter += d_group_weights[cuIdx * stride + i];
    }
    partial_sums[threadIdx.x] = counter;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      d_group_weights[cuIdx * stride] = partial_sums[0] > 0.0f ? 1.0f / partial_sums[0] : 0.0f;
    }
    __syncthreads();
  }
}

void run_wht_ht_iwht_chunk(float *d_gathered4dstack, int patch_size, uint *d_nstacks,
                           int groups, float *d_group_weights,
                           const bm4d_gpu::DeviceParams& params, const cudaDeviceProp &d_prop,
                           PipelineContext &ctx) {
  auto policy = thrust::cuda::par.on(ctx.stream);
  thrust::device_ptr<uint> dt_accumulated_nstacks = thrust::device_pointer_cast(ctx.d_accumulated_nstacks);
  thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks);
  thrust::exclusive_scan(policy, dt_nstacks, dt_nstacks + groups, dt_accumulated_nstacks);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemsetAsync(d_group_weights, 0,
                             sizeof(float) * groups * patch_size * patch_size * patch_size, ctx.stream));
  const int wht_blocks = clamp_block_work(groups, d_prop);
  k_run_wht_ht_iwht<<<wht_blocks, dim3(params.patch_size, params.patch_size, params.patch_size), 0, ctx.stream>>>(
      d_gathered4dstack, groups, patch_size, d_nstacks, ctx.d_accumulated_nstacks, d_group_weights,
      params.hard_th);
  checkCudaErrors(cudaGetLastError());

  const int reduction_threads = std::min(patch_size * patch_size * patch_size, kDefaultThreads1D);
  const int reduction_blocks = clamp_block_work(groups, d_prop);
  k_sum_group_weights<<<reduction_blocks, reduction_threads, reduction_threads * sizeof(float), ctx.stream>>>(
      d_group_weights, groups, patch_size);
  checkCudaErrors(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Aggregation + float→uchar on GPU (eliminates 70M-iteration CPU loop)
// ---------------------------------------------------------------------------
__global__ void k_aggregation(float *d_denoised_volume, float *d_weights_volume, const uint3 size,
                              const uint groups, const float *d_gathered4dstack,
                              uint3float1 *d_stacks, uint *d_nstacks, float *group_weights,
                              const bm4d_gpu::DeviceParams params,
                              const uint *d_accumulated_nstacks) {
  int stride = params.patch_size * params.patch_size * params.patch_size;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < groups; i += blockDim.x * gridDim.x) {
    if (i >= groups) return;

    float weight = group_weights[i * stride];
    int patches = d_nstacks[i];
    int group_beginning = d_accumulated_nstacks[i];
    for (int p = 0; p < patches; ++p) {
      uint3float1 ref = d_stacks[group_beginning + p];

      for (int z = 0; z < params.patch_size; ++z)
        for (int y = 0; y < params.patch_size; ++y)
          for (int x = 0; x < params.patch_size; ++x) {
            int rx = x + ref.x;
            int ry = y + ref.y;
            int rz = z + ref.z;

            if (rx < 0 || rx >= size.x) continue;
            if (ry < 0 || ry >= size.y) continue;
            if (rz < 0 || rz >= size.z) continue;

            int img_idx = rx + ry*size.x + rz*size.x * size.y;
            long long int stack_idx = group_beginning * stride + x +
                                      (y + z * params.patch_size) * params.patch_size + p * stride;
            float tmp = d_gathered4dstack[stack_idx];
            atomicAdd(&d_denoised_volume[img_idx], tmp * weight);
            atomicAdd(&d_weights_volume[img_idx], weight);
          }
    }
  }
}

// Fused normalize + float→uchar conversion on GPU.
// Eliminates the 70M-iteration serial CPU loop AND reduces the D2H transfer
// from 280MB (float) to 70MB (uchar).
__global__ void k_normalize_and_convert(const float *d_denoised_volume,
                                        const float *__restrict d_weights_volume,
                                        uchar *d_output_uchar,
                                        const int im_size) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < im_size; i += blockDim.x * gridDim.x) {
    float weight = d_weights_volume[i];
    float val = weight > 0.0f ? d_denoised_volume[i] / weight : 0.0f;
    // Clamp to [0, 255] and convert
    val = fminf(fmaxf(val, 0.0f), 255.0f);
    d_output_uchar[i] = static_cast<uchar>(val);
  }
}

void run_aggregation_chunk(const uint3 size, int groups, const float *d_gathered4dstack,
                           uint3float1 *d_stacks, uint *d_nstacks, float *d_group_weights,
                           const bm4d_gpu::DeviceParams& params, const cudaDeviceProp &d_prop,
                           PipelineContext &ctx) {
  int threads = kDefaultThreads1D;
  int bs_x = clamp_1d_blocks(groups, threads, d_prop);
  k_aggregation<<<bs_x, threads, 0, ctx.stream>>>(
      ctx.d_denoised_volume, ctx.d_weights_volume, size, static_cast<uint>(groups),
      d_gathered4dstack, d_stacks, d_nstacks, d_group_weights, params,
      ctx.d_accumulated_nstacks);
  checkCudaErrors(cudaGetLastError());
}

void finalize_aggregation(int im_size, const cudaDeviceProp &d_prop, PipelineContext &ctx) {
  int threads = kDefaultThreads1D;
  int bs_x = clamp_1d_blocks(im_size, threads, d_prop);
  k_normalize_and_convert<<<bs_x, threads, 0, ctx.stream>>>(
      ctx.d_denoised_volume, ctx.d_weights_volume,
      ctx.d_output_uchar, im_size);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(
      cudaMemcpyAsync(ctx.h_pinned_output, ctx.d_output_uchar,
                      sizeof(uchar) * im_size, cudaMemcpyDeviceToHost, ctx.stream));
}
