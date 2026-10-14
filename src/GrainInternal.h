// Shared between the CPU renderer and the Metal backend: per-frame plan,
// stock lookup tables and resampling taps. Both backends run the same stages
// (GrainShared.h holds the per-pixel math) so their output matches.
#pragma once

#include <vector>

#include "GrainEngine.h"
#include "GrainShared.h"

namespace spektragrain {

constexpr float kMinBlurSigma = 0.3f;

struct StockLuts {
  int count = 0;
  float logE0 = 0.0f, step = 1.0f, logEMax = 0.0f;
  std::vector<float> layer;  // [9][count], index c * 3 + l
  std::vector<float> slope;  // [3][count], signed, magnitude floored
};

struct FramePlan {
  SGParams k{};               // kernel constants (image strides filled by the caller)
  bool identity = true;       // grain grid == image grid
  float blurSigma[3] = {};    // dye clouds + scan softness, in cells (0 = none)
  const StockLuts *luts = nullptr;
};

FramePlan makeFramePlan(const GrainSettings &settings, int width, int height);

std::vector<float> gaussianKernel(float sigma);  // normalized, radius = ceil(3 sigma)

struct ResampleTaps {
  std::vector<int> start, count, offset;
  std::vector<float> weight;
};
ResampleTaps resampleTaps(int srcCount, int dstCount);

}  // namespace spektragrain
