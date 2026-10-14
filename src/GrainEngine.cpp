// RMS-calibrated film grain, after spektrafilm's sublayer particle model
// (spektrafilm/model/grain.py, apply_grain_to_density_layers).
//
// Pipeline per frame:
//   1. convert pixels to linear DaVinci Wide Gamut and map to log exposure on
//      the stock
//   2. on a "grain grid" (stretched by the aspect ratio) evaluate the 3 dye
//      layers x 3 sublayers: developed fraction p from the stock's sublayer
//      curves, Poisson particle count -> density deviation
//   3. dye-cloud + scan blur, chroma mix (saturation), resample to the image
//   4. convert the density deviation to exposure through the stock's local
//      curve slope, back to the output colour space
//
// Strength is calibrated analytically so the diffuse RMS granularity at
// density 1.0 through a 48 um aperture equals the stock's data-sheet value.
#include "GrainEngine.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include "GrainInternal.h"
#include "StockData.h"

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace spektragrain {
namespace {

// spektrafilm GrainParams defaults.
constexpr float kParticleScale[3] = {1.6f, 1.6f, 3.2f};       // per dye layer (r, g, b)
constexpr float kParticleScaleLayers[3] = {2.0f, 1.0f, 0.5f};  // per sublayer (fast, mid, slow)
constexpr float kDensityMin[3] = {0.03f, 0.03f, 0.03f};
constexpr float kUniformity[3] = {0.97f, 0.99f, 0.97f};
constexpr float kDyeCloudUm = 1.0f;

constexpr double kApertureAreaUm2 = 3.14159265358979 * 24.0 * 24.0;  // 48 um diameter
constexpr float kSlopeFloorFraction = 0.25f;  // of the mid-gray slope; tames toe and shoulder

// Parallel helper

template <class F>
void parallelFor(int count, int chunk, const F &fn) {
  if (count <= 0) {
    return;
  }
#if defined(__APPLE__)
  const int chunks = (count + chunk - 1) / chunk;
  struct Ctx {
    const F *fn;
    int count;
    int chunk;
  } ctx{&fn, count, chunk};
  dispatch_apply_f(static_cast<size_t>(chunks), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                   [](void *p, size_t i) {
                     const Ctx *c = static_cast<const Ctx *>(p);
                     const int begin = static_cast<int>(i) * c->chunk;
                     const int end = std::min(c->count, begin + c->chunk);
                     for (int k = begin; k < end; ++k) {
                       (*c->fn)(k);
                     }
                   });
#else
  (void)chunk;
  for (int k = 0; k < count; ++k) {
    fn(k);
  }
#endif
}

// Stock model

struct StockModel {
  StockLuts luts;
  bool positive = false;
  std::vector<float> curve[3];  // density above base
  float dtot[3][3] = {};        // sublayer max density incl. its share of base fog
  float frac[3][3] = {};        // sublayer share of the channel's max density
  float dmin[3][3] = {};        // sublayer share of base fog
  double selwyn[3][3] = {};     // per-layer Selwyn term at density 1.0 (per unit particle area)
  double selwynSum[3] = {};
  float calibLogE[3] = {};      // exposure where the channel reaches density 1.0
};

inline float lutEval(const float *lut, const StockLuts &l, float logE) {
  float t = (logE - l.logE0) / l.step;
  t = std::min(std::max(t, 0.0f), static_cast<float>(l.count - 1));
  const int i = std::min(static_cast<int>(t), l.count - 2);
  const float f = t - static_cast<float>(i);
  return lut[i] + (lut[i + 1] - lut[i]) * f;
}

StockModel buildStockModel(int index) {
  const StockProfile &p = kStocks[index];
  StockModel m;
  StockLuts &L = m.luts;
  L.count = p.count;
  L.logE0 = p.logE0;
  L.step = p.logEStep;
  L.logEMax = p.logE0 + p.logEStep * static_cast<float>(p.count - 1);
  L.layer.resize(static_cast<size_t>(9) * L.count);
  L.slope.resize(static_cast<size_t>(3) * L.count);
  m.positive = p.positive;
  const int n = L.count;

  for (int c = 0; c < 3; ++c) {
    m.curve[c].resize(n);
    for (int i = 0; i < n; ++i) {
      m.curve[c][i] = p.curves[i * 3 + c];
    }
    for (int l = 0; l < 3; ++l) {
      for (int i = 0; i < n; ++i) {
        L.layer[static_cast<size_t>(c * 3 + l) * n + i] = p.layers[i * 9 + l * 3 + c];
      }
    }

    // Smoothed slope (5-sample central difference), floored in magnitude.
    std::vector<float> raw(n);
    for (int i = 0; i < n; ++i) {
      const int a = std::max(0, i - 2), b = std::min(n - 1, i + 2);
      raw[i] = (m.curve[c][b] - m.curve[c][a]) / (L.step * static_cast<float>(b - a));
    }
    const float mid = std::fabs(lutEval(raw.data(), L, 0.0f));
    const float floorMag = std::max(kSlopeFloorFraction * mid, 0.05f);
    const float sign = m.positive ? -1.0f : 1.0f;
    for (int i = 0; i < n; ++i) {
      L.slope[static_cast<size_t>(c) * n + i] = sign * std::max(std::fabs(raw[i]), floorMag);
    }

    float dmaxTotal = 0.0f;
    float dmaxLayer[3];
    for (int l = 0; l < 3; ++l) {
      const float *lay = &L.layer[static_cast<size_t>(c * 3 + l) * n];
      dmaxLayer[l] = *std::max_element(lay, lay + n);
      dmaxTotal += dmaxLayer[l];
    }
    for (int l = 0; l < 3; ++l) {
      m.frac[c][l] = dmaxTotal > 0.0f ? dmaxLayer[l] / dmaxTotal : 0.0f;
      m.dmin[c][l] = m.frac[c][l] * kDensityMin[c];
      m.dtot[c][l] = dmaxLayer[l] + m.dmin[c][l];
    }

    // Exposure where this channel reaches density 1.0 above base.
    const float cmax = *std::max_element(m.curve[c].begin(), m.curve[c].end());
    const float target = std::min(1.0f, 0.9f * cmax);
    int best = 0;
    float bestErr = 1e9f;
    for (int i = 0; i < n; ++i) {
      const float err = std::fabs(m.curve[c][i] - target);
      if (err < bestErr) {
        bestErr = err;
        best = i;
      }
    }
    m.calibLogE[c] = L.logE0 + L.step * static_cast<float>(best);

    // Selwyn constant per unit particle area: Dtot^2 p s scale / frac.
    double sum = 0.0;
    for (int l = 0; l < 3; ++l) {
      if (m.frac[c][l] <= 1e-4f) {
        continue;
      }
      const float lay = L.layer[static_cast<size_t>(c * 3 + l) * n + best];
      const double pl = std::clamp((lay + m.dmin[c][l]) / m.dtot[c][l], 1e-6f, 1.0f - 1e-6f);
      const double sl = 1.0 - pl * kUniformity[c];
      m.selwyn[c][l] = static_cast<double>(m.dtot[c][l]) * m.dtot[c][l] * pl * sl * kParticleScale[c] *
                       kParticleScaleLayers[l] / m.frac[c][l];
      sum += m.selwyn[c][l];
    }
    m.selwynSum[c] = sum;
  }
  return m;
}

const StockModel &stockModel(int index) {
  static std::once_flag once;
  static std::vector<StockModel> models;
  std::call_once(once, [] {
    models.reserve(kStockCount);
    for (int i = 0; i < kStockCount; ++i) {
      models.push_back(buildStockModel(i));
    }
  });
  return models[std::clamp(index, 0, kStockCount - 1)];
}

// Base particle area (um^2) giving the target RMS (mean over dye layers).
double calibratedParticleArea(const StockModel &m, float rmsTarget) {
  const double meanSum = (m.selwynSum[0] + m.selwynSum[1] + m.selwynSum[2]) / 3.0;
  const double rms = std::max(static_cast<double>(rmsTarget), 0.01) * 1e-3;
  return meanSum > 0.0 ? rms * rms * kApertureAreaUm2 / meanSum : 0.0;
}

// Planes

void blurPlane(std::vector<float> &plane, std::vector<float> &scratch, int w, int h, float sigma) {
  if (sigma < kMinBlurSigma || w <= 0 || h <= 0) {
    return;
  }
  const std::vector<float> k = gaussianKernel(sigma);
  const int r = static_cast<int>(k.size() / 2);
  scratch.resize(static_cast<size_t>(w) * h);

  parallelFor(h, 16, [&](int y) {
    const float *in = plane.data() + static_cast<size_t>(y) * w;
    float *out = scratch.data() + static_cast<size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      if (x >= r && x < w - r) {
        for (int j = -r; j <= r; ++j) acc += k[j + r] * in[x + j];
      } else {
        for (int j = -r; j <= r; ++j) acc += k[j + r] * in[std::clamp(x + j, 0, w - 1)];
      }
      out[x] = acc;
    }
  });
  parallelFor(h, 8, [&](int y) {
    float *out = plane.data() + static_cast<size_t>(y) * w;
    std::fill(out, out + w, 0.0f);
    for (int j = -r; j <= r; ++j) {
      const float kj = k[j + r];
      const float *in = scratch.data() + static_cast<size_t>(std::clamp(y + j, 0, h - 1)) * w;
      for (int x = 0; x < w; ++x) out[x] += kj * in[x];
    }
  });
}

// src (sw x sh) -> dst (dw x dh)
void resamplePlane(const std::vector<float> &src, int sw, int sh, std::vector<float> &dst, int dw, int dh,
                   const ResampleTaps &tx, const ResampleTaps &ty) {
  std::vector<float> mid(static_cast<size_t>(dw) * sh);
  parallelFor(sh, 16, [&](int y) {
    const float *in = src.data() + static_cast<size_t>(y) * sw;
    float *out = mid.data() + static_cast<size_t>(y) * dw;
    for (int x = 0; x < dw; ++x) {
      const float *w = tx.weight.data() + tx.offset[x];
      const float *s = in + tx.start[x];
      float acc = 0.0f;
      for (int k = 0; k < tx.count[x]; ++k) acc += w[k] * s[k];
      out[x] = acc;
    }
  });
  dst.assign(static_cast<size_t>(dw) * dh, 0.0f);
  parallelFor(dh, 8, [&](int y) {
    float *out = dst.data() + static_cast<size_t>(y) * dw;
    const float *w = ty.weight.data() + ty.offset[y];
    for (int k = 0; k < ty.count[y]; ++k) {
      const float *in = mid.data() + static_cast<size_t>(ty.start[y] + k) * dw;
      const float wk = w[k];
      for (int x = 0; x < dw; ++x) out[x] += wk * in[x];
    }
  });
}

struct GridSize {
  int w = 0, h = 0;
  bool identity = true;
};

GridSize grainGrid(int width, int height, float aspect) {
  GridSize g{width, height, true};
  const float a = std::clamp(aspect, 0.05f, 2.0f);
  if (std::fabs(a - 1.0f) < 1e-4f) {
    return g;
  }
  const float sa = std::sqrt(a);
  g.w = std::max(1, static_cast<int>(std::lround(width / sa)));
  g.h = std::max(1, static_cast<int>(std::lround(height * sa)));
  g.identity = false;
  return g;
}

// Stretching the grain grid changes per-pixel variance through resampling;
// measure that on white noise and return the compensating gain.
float aspectGain(float aspect, float sigma) {
  constexpr int kTest = 256;
  const GridSize g = grainGrid(kTest, kTest, aspect);
  if (g.identity) {
    return 1.0f;
  }
  static std::mutex cacheMutex;
  static float cachedAspect = -1.0f, cachedSigma = -1.0f, cachedGain = 1.0f;
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (cachedAspect == aspect && cachedSigma == sigma) {
      return cachedGain;
    }
  }
  auto variance = [&](int gw, int gh, bool resample) {
    std::vector<float> plane(static_cast<size_t>(gw) * gh), scratch;
    for (int y = 0; y < gh; ++y)
      for (int x = 0; x < gw; ++x) plane[static_cast<size_t>(y) * gw + x] = sg_normal(sg_cellKey(0x5EEDul, static_cast<sg_u32>(x), static_cast<sg_u32>(y)));
    blurPlane(plane, scratch, gw, gh, sigma);
    std::vector<float> img;
    if (resample) {
      resamplePlane(plane, gw, gh, img, kTest, kTest, resampleTaps(gw, kTest), resampleTaps(gh, kTest));
    } else {
      img = plane;
    }
    const int border = static_cast<int>(std::ceil(3.0f * sigma * 4.5f)) + 2;
    double s = 0.0, s2 = 0.0;
    int n = 0;
    for (int y = border; y < kTest - border; ++y)
      for (int x = border; x < kTest - border; ++x) {
        const double v = img[static_cast<size_t>(y) * kTest + x];
        s += v;
        s2 += v * v;
        ++n;
      }
    const double mean = s / n;
    return s2 / n - mean * mean;
  };
  const double ref = variance(kTest, kTest, false);
  const double asp = variance(g.w, g.h, true);
  const float gain = asp > 0.0 ? static_cast<float>(std::sqrt(ref / asp)) : 1.0f;
  std::lock_guard<std::mutex> lock(cacheMutex);
  cachedAspect = aspect;
  cachedSigma = sigma;
  cachedGain = gain;
  return gain;
}

inline float sampleBilinear(const float *plane, int w, int h, float fx, float fy) {
  fx = std::clamp(fx, 0.0f, static_cast<float>(w - 1));
  fy = std::clamp(fy, 0.0f, static_cast<float>(h - 1));
  const int x0 = std::min(static_cast<int>(fx), w - 1), y0 = std::min(static_cast<int>(fy), h - 1);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const float tx = fx - x0, ty = fy - y0;
  const float a = plane[static_cast<size_t>(y0) * w + x0], b = plane[static_cast<size_t>(y0) * w + x1];
  const float c = plane[static_cast<size_t>(y1) * w + x0], d = plane[static_cast<size_t>(y1) * w + x1];
  const float top = a + (b - a) * tx, bottom = c + (d - c) * tx;
  return top + (bottom - top) * ty;
}

inline const float *rowPtr(const void *base, ptrdiff_t rowBytes, int y) {
  return reinterpret_cast<const float *>(static_cast<const char *>(base) + rowBytes * y);
}
inline float *rowPtr(void *base, ptrdiff_t rowBytes, int y) {
  return reinterpret_cast<float *>(static_cast<char *>(base) + rowBytes * y);
}

}  // namespace

// Shared helpers (GrainInternal.h)

std::vector<float> gaussianKernel(float sigma) {
  const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
  std::vector<float> k(2 * radius + 1);
  float sum = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    k[i + radius] = std::exp(-0.5f * static_cast<float>(i * i) / (sigma * sigma));
    sum += k[i + radius];
  }
  for (float &v : k) {
    v /= sum;
  }
  return k;
}

// Area-weighted box when shrinking, linear when enlarging. Cells are unit
// intervals [j, j + 1).
ResampleTaps resampleTaps(int srcCount, int dstCount) {
  ResampleTaps t;
  t.start.resize(dstCount);
  t.count.resize(dstCount);
  t.offset.resize(dstCount);
  const double scale = static_cast<double>(srcCount) / dstCount;  // src cells per dst cell
  for (int i = 0; i < dstCount; ++i) {
    t.offset[i] = static_cast<int>(t.weight.size());
    if (scale >= 1.0) {
      const double lo = i * scale, hi = (i + 1) * scale;
      const int j0 = static_cast<int>(std::floor(lo));
      const int j1 = std::min(srcCount, static_cast<int>(std::ceil(hi)));
      t.start[i] = j0;
      t.count[i] = j1 - j0;
      for (int j = j0; j < j1; ++j) {
        const double overlap = std::min(hi, j + 1.0) - std::max(lo, static_cast<double>(j));
        t.weight.push_back(static_cast<float>(overlap / scale));
      }
    } else {
      const double s = (i + 0.5) * scale - 0.5;
      const int j0 = static_cast<int>(std::floor(s));
      const float f = static_cast<float>(s - j0);
      const int a = std::clamp(j0, 0, srcCount - 1), b = std::clamp(j0 + 1, 0, srcCount - 1);
      t.start[i] = a;
      if (a == b) {
        t.count[i] = 1;
        t.weight.push_back(1.0f);
      } else {
        t.count[i] = 2;
        t.weight.push_back(1.0f - f);
        t.weight.push_back(f);
      }
    }
  }
  return t;
}

FramePlan makeFramePlan(const GrainSettings &s, int width, int height) {
  const StockModel &m = stockModel(s.stock);
  const StockLuts &L = m.luts;
  FramePlan p;
  SGParams &k = p.k;
  const GridSize grid = grainGrid(width, height, s.aspect);
  p.identity = grid.identity;
  p.luts = &L;
  k.width = static_cast<sg_u32>(width);
  k.height = static_cast<sg_u32>(height);
  k.gw = static_cast<sg_u32>(grid.w);
  k.gh = static_cast<sg_u32>(grid.h);
  k.srcComponents = k.dstComponents = 4;
  k.srcStride = k.dstStride = static_cast<sg_u32>(width) * 4;
  k.lutCount = static_cast<sg_u32>(L.count);
  k.identity = grid.identity ? 1 : 0;
  k.grainOnly = s.view == ViewMode::GrainOnly ? 1 : 0;
  setupColourSpaces(k, std::clamp(s.inputSpace, 0, kColourSpaceCount - 1),
                    s.outputSpace < 0 ? kOutputMatchInput : std::clamp(s.outputSpace, 0, kColourSpaceCount - 1));
  const sg_u64 seedKey = sg_mix64(0xC0FFEEul ^ (static_cast<sg_u64>(s.seed) * 0x9E3779B97F4A7C15ul));
  k.seedLo = static_cast<sg_u32>(seedKey & 0xFFFFFFFFul);
  k.seedHi = static_cast<sg_u32>(seedKey >> 32);
  k.logE0 = L.logE0;
  k.logEStep = L.step;
  k.logEMax = L.logEMax;
  k.expBias = s.exposureStops * 0.30103f;
  k.scaleX = static_cast<float>(width) / grid.w;
  k.scaleY = static_cast<float>(height) / grid.h;
  k.satMix = std::clamp(s.saturation, 0.0f, 4.0f);

  const float sizeScale = std::clamp(s.size, 0.05f, 20.0f);
  const int framePx = s.frameWidthPx > 0 ? s.frameWidthPx : width;
  const float pixelUm = s.frameWidthMm * 1000.0f / (static_cast<float>(std::max(framePx, 1)) * sizeScale);
  const float cellArea = pixelUm * pixelUm;  // stretching the grid preserves cell area
  const double a0 = calibratedParticleArea(m, s.rmsTarget);
  const float softness = std::max(s.softnessPx, 0.0f) * std::clamp(s.renderScale, 0.01f, 1.0f);

  for (int c = 0; c < 3; ++c) {
    k.uniformity[c] = kUniformity[c];
    double dyeVar = 0.0;
    for (int l = 0; l < 3; ++l) {
      const int i = c * 3 + l;
      const double particle = a0 * kParticleScale[c] * kParticleScaleLayers[l];
      k.invLambda[i] = m.frac[c][l] > 1e-4f ? static_cast<float>(particle / (cellArea * m.frac[c][l])) : 0.0f;
      k.dtot[i] = m.dtot[c][l];
      k.dmin[i] = m.dmin[c][l];
      const double dye = kDyeCloudUm * std::sqrt(kParticleScale[c] * kParticleScaleLayers[l]) / pixelUm;
      dyeVar += m.selwyn[c][l] * dye * dye;
    }
    // One blur per dye layer: sublayer dye clouds (variance-weighted) + scan.
    const float dyeSigma2 = m.selwynSum[c] > 0.0 ? static_cast<float>(dyeVar / m.selwynSum[c]) : 0.0f;
    const float sigma = std::sqrt(dyeSigma2 + softness * softness);
    p.blurSigma[c] = sigma >= kMinBlurSigma ? sigma : 0.0f;
    k.blurRadius[c] = p.blurSigma[c] > 0.0f ? static_cast<sg_u32>(gaussianKernel(p.blurSigma[c]).size() / 2) : 0;
  }

  k.gain = std::max(s.amount, 0.0f);
  if (!grid.identity) {
    k.gain *= aspectGain(s.aspect, p.blurSigma[1]);
  }
  return p;
}

double modelRmsAtUnitDensity(const GrainSettings &settings, int channel) {
  const StockModel &m = stockModel(settings.stock);
  const double a0 = calibratedParticleArea(m, settings.rmsTarget);
  return std::sqrt(a0 * m.selwynSum[std::clamp(channel, 0, 2)] / kApertureAreaUm2) * 1000.0;
}

float calibrationLogExposure(int stock, int channel) {
  return stockModel(stock).calibLogE[std::clamp(channel, 0, 2)];
}

float curveSlope(int stock, int channel, float logE) {
  const StockLuts &L = stockModel(stock).luts;
  return lutEval(&L.slope[static_cast<size_t>(std::clamp(channel, 0, 2)) * L.count], L, logE);
}

// CPU renderer

void renderGrain(const GrainSettings &s, const ImageRef &src, const MutableImageRef &dst, int width, int height) {
  if (width <= 0 || height <= 0) {
    return;
  }
  const FramePlan plan = makeFramePlan(s, width, height);
  const SGParams &p = plan.k;
  const StockLuts &L = *plan.luts;
  const int gw = static_cast<int>(p.gw), gh = static_cast<int>(p.gh);
  const size_t pixels = static_cast<size_t>(width) * height;

  // 1. log exposure on the image grid.
  std::vector<float> logE[3];
  for (auto &plane : logE) plane.resize(pixels);
  parallelFor(height, 8, [&](int y) {
    const float *in = rowPtr(src.data, src.rowBytes, y);
    for (int x = 0; x < width; ++x) {
      const float *px = in + static_cast<size_t>(x) * src.components;
      const SGRgb lin = sg_toWorking(p, px[0], px[1], px[2]);
      for (int c = 0; c < 3; ++c) {
        logE[c][static_cast<size_t>(y) * width + x] = sg_pixelLogExposure(p, sg_comp(lin, c));
      }
    }
  });

  // 2. density deviation per grain cell.
  const size_t cells = static_cast<size_t>(gw) * gh;
  std::vector<float> noise[3];
  for (auto &plane : noise) plane.resize(cells);
  const sg_u64 seedKey = sg_seedKey(p);
  parallelFor(gh, 4, [&](int gy) {
    for (int gx = 0; gx < gw; ++gx) {
      const sg_u64 key = sg_cellKey(seedKey, static_cast<sg_u32>(gx), static_cast<sg_u32>(gy));
      const size_t idx = static_cast<size_t>(gy) * gw + gx;
      for (int c = 0; c < 3; ++c) {
        const float le = plan.identity ? logE[c][idx]
                                    : sampleBilinear(logE[c].data(), width, height, (gx + 0.5f) * p.scaleX - 0.5f,
                                                     (gy + 0.5f) * p.scaleY - 0.5f);
        float sum = 0.0f;
        for (int l = 0; l < 3; ++l) {
          const int i = c * 3 + l;
          sum += sg_layerDeviation(p, i, sg_lutEval(&L.layer[static_cast<size_t>(i) * L.count], p, le), key);
        }
        noise[c][idx] = sum;
      }
    }
  });

  // 3. blur, chroma mix, back to the image grid.
  std::vector<float> scratch;
  for (int c = 0; c < 3; ++c) {
    blurPlane(noise[c], scratch, gw, gh, plan.blurSigma[c]);
  }
  if (std::fabs(p.satMix - 1.0f) > 1e-4f) {
    parallelFor(gh, 16, [&](int gy) {
      const size_t base = static_cast<size_t>(gy) * gw;
      for (int gx = 0; gx < gw; ++gx) {
        const size_t i = base + gx;
        const float mean = (noise[0][i] + noise[1][i] + noise[2][i]) * (1.0f / 3.0f);
        for (int c = 0; c < 3; ++c) noise[c][i] = mean + p.satMix * (noise[c][i] - mean);
      }
    });
  }
  if (!plan.identity) {
    const ResampleTaps tx = resampleTaps(gw, width), ty = resampleTaps(gh, height);
    std::vector<float> resampled;
    for (int c = 0; c < 3; ++c) {
      resamplePlane(noise[c], gw, gh, resampled, width, height, tx, ty);
      noise[c].swap(resampled);
    }
  }

  // 4. density deviation -> exposure -> output colour space.
  parallelFor(height, 8, [&](int y) {
    const float *in = rowPtr(src.data, src.rowBytes, y);
    float *out = rowPtr(dst.data, dst.rowBytes, y);
    for (int x = 0; x < width; ++x) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      const float *pi = in + static_cast<size_t>(x) * src.components;
      float *po = out + static_cast<size_t>(x) * dst.components;
      float slope[3];
      for (int c = 0; c < 3; ++c) {
        slope[c] = sg_lutEval(&L.slope[static_cast<size_t>(c) * L.count], p, logE[c][idx]);
      }
      const SGRgb lin = sg_toWorking(p, pi[0], pi[1], pi[2]);
      const SGRgb rgb = sg_applyGrain(p, sg_rgb(pi[0], pi[1], pi[2]), lin,
                                      sg_rgb(noise[0][idx], noise[1][idx], noise[2][idx]),
                                      sg_rgb(slope[0], slope[1], slope[2]));
      const float alpha = src.components == 4 ? pi[3] : 1.0f;
      po[0] = rgb.r;
      po[1] = rgb.g;
      po[2] = rgb.b;
      if (dst.components == 4) {
        po[3] = alpha;
      }
    }
  });
}

}  // namespace spektragrain
