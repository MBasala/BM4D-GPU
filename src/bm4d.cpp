// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#include <bm4d-gpu/bm4d.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t kMinChunkReserveBytes = 1ull << 30;

struct ChunkRange {
  std::size_t group_begin = 0;
  std::size_t group_end = 0;
  std::size_t stack_begin = 0;
  std::size_t stack_count = 0;
};

std::size_t gathered_stack_bytes(const bm4d_gpu::DeviceParams &params) {
  const std::size_t patch_volume =
      static_cast<std::size_t>(params.patch_size) * params.patch_size * params.patch_size;
  return sizeof(float) * patch_volume;
}

std::size_t select_chunk_stack_budget(const bm4d_gpu::DeviceParams &params) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));

  const std::size_t reserve_bytes = (std::max)(kMinChunkReserveBytes, total_bytes / 6);
  const std::size_t free_after_reserve =
      free_bytes > reserve_bytes ? free_bytes - reserve_bytes : free_bytes / 4;
  const std::size_t chunk_bytes = (std::min)(free_bytes / 2, free_after_reserve);
  const std::size_t stack_budget = chunk_bytes / gathered_stack_bytes(params);
  return (std::max)(static_cast<std::size_t>(params.maxN), stack_budget);
}

std::vector<ChunkRange> plan_group_chunks(const std::vector<uint> &h_nstacks,
                                          const std::size_t max_chunk_stacks) {
  std::vector<ChunkRange> chunks;
  std::size_t group_begin = 0;
  std::size_t stack_begin = 0;

  while (group_begin < h_nstacks.size()) {
    std::size_t group_end = group_begin;
    std::size_t stack_count = 0;

    while (group_end < h_nstacks.size()) {
      const std::size_t next_stack_count = static_cast<std::size_t>(h_nstacks[group_end]);
      if (stack_count > 0 && stack_count + next_stack_count > max_chunk_stacks) break;
      stack_count += next_stack_count;
      ++group_end;
      if (stack_count >= max_chunk_stacks) break;
    }

    if (group_end == group_begin) {
      stack_count = static_cast<std::size_t>(h_nstacks[group_end]);
      ++group_end;
    }

    chunks.push_back({group_begin, group_end, stack_begin, stack_count});
    stack_begin += stack_count;
    group_begin = group_end;
  }

  return chunks;
}

}  // namespace

std::vector<uchar> BM4D::run_first_step() {
  assert(size == noisy_volume.size());

  const uint3 im_size = make_uint3(width, height, depth);
  const uint3 tr_size = make_uint3(twidth, theight, tdepth);

  // -----------------------------------------------------------------------
  // Long-lived pipeline state is pre-allocated in the constructor:
  //   - 3D texture for the volume
  //   - DCT constants in __constant__ memory
  //   - Persistent GPU buffers for block-matching and accumulation
  //   - Pinned host output buffer
  //   - CUDA stream
  // Chunk-sized scratch buffers are sized after block matching so large 3D
  // volumes stay below the VRAM paging cliff.
  // -----------------------------------------------------------------------

  Stopwatch blockmatching(true);
  run_block_matching(pipeline_ctx.tex_volume, im_size, tr_size, d_params,
                     d_stacks, d_nstacks, d_prop, pipeline_ctx.stream);
  // Chunk planning needs the normalized stack counts on the host.
  checkCudaErrors(cudaStreamSynchronize(pipeline_ctx.stream));
  blockmatching.stop();
  std::cout << "Blockmatching took: " << blockmatching.getSeconds() << std::endl;

  std::vector<uint> h_nstacks;
  uint gather_stacks_sum = 0;
  prepare_stacks_for_processing(tr_size, d_params, d_stacks, d_nstacks, h_nstacks,
                                gather_stacks_sum, d_prop, pipeline_ctx.stream);
  if (gather_stacks_sum == 0) return noisy_volume;

  const std::size_t max_chunk_stacks_budget = select_chunk_stack_budget(d_params);
  const std::vector<ChunkRange> chunks = plan_group_chunks(h_nstacks, max_chunk_stacks_budget);

  std::size_t max_chunk_stacks = 0;
  std::size_t max_chunk_groups = 0;
  for (const ChunkRange &chunk : chunks) {
    max_chunk_stacks = (std::max)(max_chunk_stacks, chunk.stack_count);
    max_chunk_groups = (std::max)(max_chunk_groups, chunk.group_end - chunk.group_begin);
  }

  pipeline_ctx.ensure_group_weight_capacity(static_cast<int>(max_chunk_groups), d_params.patch_size);
  d_group_weights = pipeline_ctx.d_group_weights;

  if (d_gathered4dstack != nullptr) {
    checkCudaErrors(cudaFreeAsync(d_gathered4dstack, pipeline_ctx.stream));
    checkCudaErrors(cudaStreamSynchronize(pipeline_ctx.stream));
    d_gathered4dstack = nullptr;
  }

  if (max_chunk_stacks > 0) {
    checkCudaErrors(cudaMallocAsync(
        (void **)&d_gathered4dstack,
        gathered_stack_bytes(d_params) * max_chunk_stacks,
        pipeline_ctx.stream));
  }

  if (chunks.size() > 1) {
    const std::size_t workspace_mib =
        (gathered_stack_bytes(d_params) * max_chunk_stacks) / (1024ull * 1024ull);
    std::cout << "Chunking post-processing into " << chunks.size()
              << " chunks using up to " << workspace_mib
              << " MiB of gathered-stack workspace" << std::endl;
  }

  checkCudaErrors(cudaMemsetAsync(pipeline_ctx.d_denoised_volume, 0,
                                  sizeof(float) * size, pipeline_ctx.stream));
  checkCudaErrors(cudaMemsetAsync(pipeline_ctx.d_weights_volume, 0,
                                  sizeof(float) * size, pipeline_ctx.stream));

  Stopwatch aggregation(true);
  for (const ChunkRange &chunk : chunks) {
    const int chunk_groups = static_cast<int>(chunk.group_end - chunk.group_begin);
    const uint chunk_stack_count = static_cast<uint>(chunk.stack_count);
    if (chunk_groups == 0 || chunk_stack_count == 0) continue;

    uint3float1 *d_chunk_stacks = d_stacks + chunk.stack_begin;
    uint *d_chunk_nstacks = d_nstacks + chunk.group_begin;

    gather_cubes_chunk(pipeline_ctx.tex_volume, im_size, d_params, d_chunk_stacks,
                       chunk_stack_count, d_gathered4dstack, d_prop, pipeline_ctx.stream);
    run_dct3d(d_gathered4dstack, chunk_stack_count, d_params.patch_size, d_prop,
              pipeline_ctx.stream);
    run_wht_ht_iwht_chunk(d_gathered4dstack, d_params.patch_size, d_chunk_nstacks,
                          chunk_groups, d_group_weights, d_params, d_prop,
                          pipeline_ctx);
    run_idct3d(d_gathered4dstack, chunk_stack_count, d_params.patch_size, d_prop,
               pipeline_ctx.stream);
    run_aggregation_chunk(im_size, chunk_groups, d_gathered4dstack, d_chunk_stacks,
                          d_chunk_nstacks, d_group_weights, d_params, d_prop,
                          pipeline_ctx);
  }

  finalize_aggregation(size, d_prop, pipeline_ctx);
  checkCudaErrors(cudaStreamSynchronize(pipeline_ctx.stream));
  aggregation.stop();
  std::cout << "Aggregation took: " << aggregation.getSeconds() << std::endl;

  std::memcpy(noisy_volume.data(), pipeline_ctx.h_pinned_output, size);
  return noisy_volume;
}
