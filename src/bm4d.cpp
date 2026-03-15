// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#include <bm4d-gpu/bm4d.h>

std::vector<uchar> BM4D::run_first_step() {
  uchar *d_noisy_volume;
  assert(size == noisy_volume.size());
  checkCudaErrors(cudaMalloc((void **)&d_noisy_volume, sizeof(uchar) * size));
  checkCudaErrors(cudaMemcpy((void *)d_noisy_volume, (void *)noisy_volume.data(),
                             sizeof(uchar) * size, cudaMemcpyHostToDevice));

  uint3 im_size = make_uint3(width, height, depth);
  uint3 tr_size =
      make_uint3(twidth, theight, tdepth);  // Truncated size, with some step for ref patches

  // Do block matching
  Stopwatch blockmatching(true);
  run_block_matching(d_noisy_volume, im_size, tr_size, params, d_stacks, d_nstacks, d_prop);
  blockmatching.stop();
  std::cout << "Blockmatching took: " << blockmatching.getSeconds() << std::endl;

  // Gather cubes together
  uint gather_stacks_sum;
  gather_cubes(d_noisy_volume, im_size, tr_size, params, d_stacks, d_nstacks, d_gathered4dstack,
               gather_stacks_sum, d_prop);
  cudaFree(d_noisy_volume);

  // Perform 3D DCT
  run_dct3d(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_prop);

  // Do WHT in 4th dim + Hard Thresholding + IWHT
  float *d_group_weights;
  run_wht_ht_iwht(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_nstacks, tr_size,
                  d_group_weights, params, d_prop);

  // Perform inverse 3D DCT
  run_idct3d(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_prop);

  // Aggregate
  std::vector<float> final_image(width * height * depth, 0.0f);
  run_aggregation(final_image.data(), im_size, tr_size, d_gathered4dstack, d_stacks, d_nstacks,
                  d_group_weights, params, gather_stacks_sum, d_prop);
  for (int i = 0; i < size; i++) {
    noisy_volume[i] = static_cast<uchar>(final_image[i]);
  }
  return noisy_volume;
}
