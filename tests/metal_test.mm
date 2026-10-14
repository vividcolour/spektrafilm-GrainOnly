// Metal vs CPU parity and GPU timing.
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/GrainEngine.h"
#include "../src/GrainMetal.h"
#include "../src/StockData.h"

using namespace spektragrain;

static std::vector<float> makeImage(int W, int H) {
  std::vector<float> img(static_cast<size_t>(W) * H * 4);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      float *p = &img[(static_cast<size_t>(y) * W + x) * 4];
      const float t = static_cast<float>(x) / W, u = static_cast<float>(y) / H;
      p[0] = 0.05f + 0.9f * t;
      p[1] = 0.05f + 0.9f * u;
      p[2] = 0.5f + 0.4f * std::sin(10.0f * t + 7.0f * u);
      p[3] = 0.75f;
    }
  return img;
}

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::printf("device: %s\n", device.name.UTF8String);
    MetalScratch *scratch = createMetalScratch();
    int failures = 0;

    struct Case {
      const char *name;
      int stock;
      float frame, aspect, sat, size;
      int inSpace;
      int outSpace;
      ViewMode view;
    };
    const Case cases[] = {
        {"250D S35 709", kDefaultStock, 24.89f, 1.0f, 1.0f, 1.0f, 16, kOutputMatchInput, ViewMode::Final},
        {"500T S16 aspect 1.7 sat .5", 3, 12.52f, 1.7f, 0.5f, 1.0f, kSpaceDWG, kOutputMatchInput, ViewMode::Final},
        {"Portra S8 aspect .3 size 3 DWG->709", 6, 5.79f, 0.3f, 1.0f, 3.0f, kSpaceDWG, 16, ViewMode::Final},
        {"Velvia LogC4 grain only", 19, 36.0f, 1.0f, 1.2f, 1.0f, 6, kOutputMatchInput, ViewMode::GrainOnly},
        {"Ektar ACEScct->DWG aspect 0", 10, 24.89f, 0.0f, 0.0f, 1.0f, 2, 0, ViewMode::Final},
    };
    const int W = 1031, H = 577;  // odd sizes on purpose
    for (const Case &c : cases) {
      GrainSettings s;
      s.stock = c.stock;
      s.rmsTarget = kStocks[c.stock].rmsTarget;
      s.frameWidthMm = c.frame;
      s.aspect = c.aspect;
      s.saturation = c.sat;
      s.size = c.size;
      s.inputSpace = c.inSpace;
      s.outputSpace = c.outSpace;
      s.view = c.view;
      s.seed = 42;
      std::vector<float> img = makeImage(W, H), cpu(img.size());
      renderGrain(s, {img.data(), W * 16, 4}, {cpu.data(), W * 16, 4}, W, H);

      id<MTLBuffer> src = [device newBufferWithBytes:img.data() length:img.size() * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> dst = [device newBufferWithLength:img.size() * 4 options:MTLResourceStorageModeShared];
      MetalImageLayout layout;
      layout.strideFloats = static_cast<size_t>(W) * 4;
      std::string err;
      if (!renderGrainMetal(s, scratch, (__bridge void *)queue, (__bridge void *)src, layout, (__bridge void *)dst,
                            layout, W, H, true, &err)) {
        std::printf("FAIL %s: %s\n", c.name, err.c_str());
        ++failures;
        continue;
      }
      const float *gpu = static_cast<const float *>(dst.contents);
      double maxDiff = 0.0, sumDiff = 0.0, grainRms = 0.0;
      size_t big = 0;
      for (size_t i = 0; i < img.size(); ++i) {
        const double d = std::fabs(static_cast<double>(gpu[i]) - cpu[i]);
        maxDiff = std::max(maxDiff, d);
        sumDiff += d;
        if (d > 1e-3) ++big;
        if (i % 4 != 3) grainRms += (cpu[i] - img[i]) * (cpu[i] - img[i]);
      }
      grainRms = std::sqrt(grainRms / (img.size() * 0.75));
      const bool ok = big < img.size() / 10000 && sumDiff / img.size() < 1e-5;
      std::printf("%s %-30s max %.2e  mean %.2e  >1e-3: %zu  (grain rms in code values %.4f)\n", ok ? "ok  " : "FAIL",
                  c.name, maxDiff, sumDiff / img.size(), big, grainRms);
      if (!ok) ++failures;
    }

    // Timing at UHD.
    const int TW = 3840, TH = 2160;
    std::vector<float> img = makeImage(TW, TH);
    id<MTLBuffer> src = [device newBufferWithBytes:img.data() length:img.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst = [device newBufferWithLength:img.size() * 4 options:MTLResourceStorageModePrivate];
    MetalImageLayout layout;
    layout.strideFloats = static_cast<size_t>(TW) * 4;
    for (float aspect : {1.0f, 1.6f}) {
      GrainSettings s;
      s.stock = kDefaultStock;
      s.aspect = aspect;
      std::string err;
      renderGrainMetal(s, scratch, (__bridge void *)queue, (__bridge void *)src, layout, (__bridge void *)dst, layout,
                       TW, TH, true, &err);
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 10; ++i) {
        s.seed = i;
        renderGrainMetal(s, scratch, (__bridge void *)queue, (__bridge void *)src, layout, (__bridge void *)dst,
                         layout, TW, TH, i == 9, &err);
      }
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10;
      std::printf("UHD aspect %.1f: %.2f ms/frame %s\n", aspect, ms, err.c_str());
    }
    destroyMetalScratch(scratch);
    std::printf(failures ? "%d FAILURES\n" : "all Metal checks passed\n", failures);
    return failures ? 1 : 0;
  }
}
