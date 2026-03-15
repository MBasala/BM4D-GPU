// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#pragma once
#include <cmath>
#include <vector>

namespace bm4d_gpu {

template <typename T>
constexpr T sqr(const T &val) {
  return val * val;
}

inline float psnr(const std::vector<unsigned char> &gt, const std::vector<unsigned char> &noisy) {
   constexpr float max_signal{255.f};
  float sqr_err{0.f};
  for (std::vector<unsigned char>::size_type i = 0; i < gt.size(); ++i) {
    float sqr_diff = sqr(static_cast<float>(gt[i]) - static_cast<float>(noisy[i]));
    sqr_err += sqr_diff;
  }
  float mse = sqr_err / static_cast<float>(gt.size());
  float psnr = 10.f * std::log10(max_signal * max_signal / mse);
  return psnr;
}

}  // namespace bm4d_gpu
