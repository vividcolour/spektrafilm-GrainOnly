// Per-pixel grain math shared by the C++ renderer and the Metal kernels.
// This file compiles as C++17 and as Metal Shading Language (it is embedded
// verbatim into the Metal library source at build time), so keep it to the
// common subset: plain structs of 4-byte scalars, no std:: containers.
#ifndef SPEKTRAGRAIN_SHARED_H
#define SPEKTRAGRAIN_SHARED_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
typedef ulong sg_u64;
typedef uint sg_u32;
#define SG_INLINE inline
#define SG_PARAMS constant SGParams &
#define SG_LUT device const float *
#else
#include <cmath>
#include <cstdint>
namespace spektragrain {
typedef uint64_t sg_u64;
typedef uint32_t sg_u32;
using std::exp;
using std::exp2;
using std::fabs;
using std::floor;
using std::fmax;
using std::log;
using std::log10;
using std::log2;
using std::pow;
using std::sqrt;
#define SG_INLINE inline
#define SG_PARAMS const SGParams &
#define SG_LUT const float *
#endif

// Transfer function ids (same as SaturnOFX).
#define SG_TF_LINEAR 0
#define SG_TF_DAVINCI_INT 1
#define SG_TF_ACESCCT 2
#define SG_TF_LOGC3 3
#define SG_TF_LOGC4 4
#define SG_TF_SLOG3 5
#define SG_TF_VLOG 6
#define SG_TF_LOG3G10 7
#define SG_TF_CLOG3 8
#define SG_TF_FLOG2 9
#define SG_TF_APPLELOG 10
#define SG_TF_BMDGEN5 11
#define SG_TF_DLOG 12
#define SG_TF_GAMMA24 13
#define SG_TF_SRGB 14

#define SG_MID_GRAY 0.184f
#define SG_POISSON_EXACT_BELOW 3.0f

// Per-frame constants. 4-byte members only so C++ and Metal layouts match.
struct SGParams {
  sg_u32 width, height;      // image grid
  sg_u32 gw, gh;             // grain grid
  sg_u32 srcStride, dstStride;  // floats per row
  sg_u32 srcOffset, dstOffset;  // floats to the region's first pixel (GPU only)
  sg_u32 srcComponents, dstComponents;
  sg_u32 lutCount, identity, grainOnly;
  sg_u32 inTF, outTF;         // transfer functions of the input / output colour space
  sg_u32 matchInput;          // output space == input space: add grain as an offset
  float inMat[9];            // input-gamut linear -> DWG linear (row-major)
  float outMat[9];           // DWG linear -> output-gamut linear
  sg_u32 seedLo, seedHi;
  sg_u32 blurRadius[3];
  float logE0, logEStep, logEMax, expBias;
  float scaleX, scaleY;      // image pixels per grain cell
  float gain, satMix;
  float invLambda[9];        // 1 / particles per cell (0 = layer unused), index c*3+l
  float dtot[9];
  float dmin[9];
  float uniformity[3];
};

SG_INLINE float sg_min(float a, float b) { return a < b ? a : b; }
SG_INLINE float sg_max(float a, float b) { return a > b ? a : b; }
SG_INLINE float sg_clamp(float v, float lo, float hi) { return sg_min(sg_max(v, lo), hi); }
SG_INLINE float sg_spow(float x, float e) { return x < 0.0f ? -pow(-x, e) : pow(x, e); }

// Colour. Grain is computed in linear DaVinci Wide Gamut.

struct SGRgb {
  float r, g, b;
};

SG_INLINE SGRgb sg_rgb(float r, float g, float b) {
  SGRgb c;
  c.r = r;
  c.g = g;
  c.b = b;
  return c;
}

SG_INLINE float sg_comp(SGRgb c, int i) { return i == 0 ? c.r : (i == 1 ? c.g : c.b); }

#define SG_LOG2_10 3.32192809488736f
#define SG_INV_LOG2_10 0.30102999566398f
SG_INLINE float sg_log10(float x) { return log2(x) * SG_INV_LOG2_10; }
SG_INLINE float sg_pow10(float x) { return exp2(x * SG_LOG2_10); }

// Transfer functions. decode: code value -> scene linear. encode: inverse.
SG_INLINE float sg_decode(sg_u32 tf, float x) {
  if (tf == SG_TF_LINEAR) return x;
  if (tf == SG_TF_DAVINCI_INT) {
    const float A = 0.0075f, B = 7.0f, C = 0.07329248f, M = 10.44426855f, LOGCUT = 0.02740668f;
    return x <= LOGCUT ? x / M : exp2(x / C - B) - A;
  }
  if (tf == SG_TF_ACESCCT) {
    return x <= 0.155251141552511f ? (x - 0.0729055341958355f) / 10.5402377416545f : exp2(x * 17.52f - 9.72f);
  }
  if (tf == SG_TF_LOGC3) {  // EI 800
    const float cut = 0.010591f, a = 5.555556f, b = 0.052272f, c = 0.247190f, d = 0.385537f, e = 5.367655f,
                f = 0.092809f;
    return x > e * cut + f ? (sg_pow10((x - d) / c) - b) / a : (x - f) / e;
  }
  if (tf == SG_TF_LOGC4) {
    const float a = 2231.82630906769f, b = 0.9071358748778104f, c = 0.0928641251221896f, s = 0.1135972086105891f,
                t = -0.0180569961199113f;
    return x >= 0.0f ? (exp2(14.0f * (x - c) / b + 6.0f) - 64.0f) / a : x * s + t;
  }
  if (tf == SG_TF_SLOG3) {
    return x >= 171.2102946929f / 1023.0f ? sg_pow10((x * 1023.0f - 420.0f) / 261.5f) * 0.19f - 0.01f
                                          : (x * 1023.0f - 95.0f) * 0.01125f / (171.2102946929f - 95.0f);
  }
  if (tf == SG_TF_VLOG) {
    const float b = 0.00873f, c = 0.241514f, d = 0.598206f;
    return x < 0.181f ? (x - 0.125f) / 5.6f : sg_pow10((x - d) / c) - b;
  }
  if (tf == SG_TF_LOG3G10) {
    const float a = 0.224282f, b = 155.975327f, c = 0.01f, g = 15.1927f;
    return (x < 0.0f ? x / g : (sg_pow10(x / a) - 1.0f) / b) - c;
  }
  if (tf == SG_TF_CLOG3) {
    // Canon Log 3, full-range code values, reflection (x0.9) convention
    float y;
    if (x < 0.097465473f)
      y = -(sg_pow10((0.12783901f - x) / 0.36726845f) - 1.0f) / 14.98325f;
    else if (x <= 0.15277891f)
      y = (x - 0.12512219f) / 1.9754798f;
    else
      y = (sg_pow10((x - 0.12240537f) / 0.36726845f) - 1.0f) / 14.98325f;
    return y * 0.9f;
  }
  if (tf == SG_TF_FLOG2) {
    const float a = 5.555556f, b = 0.064829f, c = 0.245281f, d = 0.384316f, e = 8.799461f, f = 0.092864f,
                cut2 = 0.100686685370811f;
    return x < cut2 ? (x - f) / e : sg_pow10((x - d) / c) / a - b / a;
  }
  if (tf == SG_TF_APPLELOG) {
    const float R0 = -0.05641088f, Rt = 0.01f, c = 47.28711236f, beta = 0.00964052f, gamma = 0.08550479f,
                delta = 0.69336945f;
    const float Pt = c * (Rt - R0) * (Rt - R0);
    if (x >= Pt) return exp2((x - delta) / gamma) - beta;
    if (x < 0.0f) return R0;
    return sqrt(x / c) + R0;
  }
  if (tf == SG_TF_BMDGEN5) {
    const float A = 0.08692876065f, B = 0.005494072432f, C = 0.5300133392f, D = 8.283605932f, E = 0.09246575342f;
    const float LOGCUT = D * 0.005f + E;
    return x < LOGCUT ? (x - E) / D : exp2((x - C) / A * 1.4426950408889634f) - B;
  }
  if (tf == SG_TF_DLOG) {
    return x <= 0.14f ? (x - 0.0929f) / 6.025f : (sg_pow10(3.89616f * x - 2.27752f) - 0.0108f) / 0.9892f;
  }
  if (tf == SG_TF_GAMMA24) return x > 0.0f ? pow(x, 2.4f) : x;
  if (tf == SG_TF_SRGB) {
    const float ax = fabs(x);
    const float y = ax <= 0.04045f ? ax / 12.92f : pow((ax + 0.055f) / 1.055f, 2.4f);
    return x < 0.0f ? -y : y;
  }
  return x;
}

SG_INLINE float sg_encode(sg_u32 tf, float y) {
  if (tf == SG_TF_LINEAR) return y;
  if (tf == SG_TF_DAVINCI_INT) {
    const float A = 0.0075f, B = 7.0f, C = 0.07329248f, M = 10.44426855f, LINCUT = 0.00262409f;
    return y <= LINCUT ? y * M : (log2(y + A) + B) * C;
  }
  if (tf == SG_TF_ACESCCT) {
    return y <= 0.0078125f ? 10.5402377416545f * y + 0.0729055341958355f : (log2(y) + 9.72f) / 17.52f;
  }
  if (tf == SG_TF_LOGC3) {
    const float cut = 0.010591f, a = 5.555556f, b = 0.052272f, c = 0.247190f, d = 0.385537f, e = 5.367655f,
                f = 0.092809f;
    return y > cut ? c * sg_log10(a * y + b) + d : e * y + f;
  }
  if (tf == SG_TF_LOGC4) {
    const float a = 2231.82630906769f, b = 0.9071358748778104f, c = 0.0928641251221896f, s = 0.1135972086105891f,
                t = -0.0180569961199113f;
    return y >= t ? (log2(a * y + 64.0f) - 6.0f) / 14.0f * b + c : (y - t) / s;
  }
  if (tf == SG_TF_SLOG3) {
    return y >= 0.01125f ? (420.0f + sg_log10((y + 0.01f) / 0.19f) * 261.5f) / 1023.0f
                         : (y * (171.2102946929f - 95.0f) / 0.01125f + 95.0f) / 1023.0f;
  }
  if (tf == SG_TF_VLOG) {
    const float b = 0.00873f, c = 0.241514f, d = 0.598206f;
    return y < 0.01f ? 5.6f * y + 0.125f : c * sg_log10(y + b) + d;
  }
  if (tf == SG_TF_LOG3G10) {
    const float a = 0.224282f, b = 155.975327f, c = 0.01f, g = 15.1927f;
    const float x = y + c;
    return x < 0.0f ? x * g : a * sg_log10(x * b + 1.0f);
  }
  if (tf == SG_TF_CLOG3) {
    const float x = y / 0.9f;
    if (x < -0.014f) return -0.36726845f * sg_log10(-x * 14.98325f + 1.0f) + 0.12783901f;
    if (x <= 0.014f) return 1.9754798f * x + 0.12512219f;
    return 0.36726845f * sg_log10(x * 14.98325f + 1.0f) + 0.12240537f;
  }
  if (tf == SG_TF_FLOG2) {
    const float a = 5.555556f, b = 0.064829f, c = 0.245281f, d = 0.384316f, e = 8.799461f, f = 0.092864f,
                cut1 = 0.000889f;
    return y < cut1 ? e * y + f : c * sg_log10(a * y + b) + d;
  }
  if (tf == SG_TF_APPLELOG) {
    const float R0 = -0.05641088f, Rt = 0.01f, c = 47.28711236f, beta = 0.00964052f, gamma = 0.08550479f,
                delta = 0.69336945f;
    if (y >= Rt) return gamma * log2(y + beta) + delta;
    if (y >= R0) return c * (y - R0) * (y - R0);
    return 0.0f;
  }
  if (tf == SG_TF_BMDGEN5) {
    const float A = 0.08692876065f, B = 0.005494072432f, C = 0.5300133392f, D = 8.283605932f, E = 0.09246575342f;
    return y < 0.005f ? D * y + E : A * log2(y + B) * 0.6931471805599453f + C;
  }
  if (tf == SG_TF_DLOG) {
    return y <= 0.0078f ? 6.025f * y + 0.0929f : (sg_log10(y * 0.9892f + 0.0108f)) * 0.256663f + 0.584555f;
  }
  if (tf == SG_TF_GAMMA24) return y > 0.0f ? pow(y, 1.0f / 2.4f) : y;
  if (tf == SG_TF_SRGB) {
    const float ay = fabs(y);
    const float x = ay <= 0.0031308f ? ay * 12.92f : 1.055f * pow(ay, 1.0f / 2.4f) - 0.055f;
    return y < 0.0f ? -x : x;
  }
  return y;
}

// Incoming code values -> linear DaVinci Wide Gamut.
SG_INLINE SGRgb sg_toWorking(SG_PARAMS p, float r, float g, float b) {
  const float lr = sg_decode(p.inTF, r), lg = sg_decode(p.inTF, g), lb = sg_decode(p.inTF, b);
  return sg_rgb(p.inMat[0] * lr + p.inMat[1] * lg + p.inMat[2] * lb,
                p.inMat[3] * lr + p.inMat[4] * lg + p.inMat[5] * lb,
                p.inMat[6] * lr + p.inMat[7] * lg + p.inMat[8] * lb);
}

// Linear DaVinci Wide Gamut -> output code values.
SG_INLINE SGRgb sg_fromWorking(SG_PARAMS p, SGRgb c) {
  const float r = p.outMat[0] * c.r + p.outMat[1] * c.g + p.outMat[2] * c.b;
  const float g = p.outMat[3] * c.r + p.outMat[4] * c.g + p.outMat[5] * c.b;
  const float b = p.outMat[6] * c.r + p.outMat[7] * c.g + p.outMat[8] * c.b;
  return sg_rgb(sg_encode(p.outTF, r), sg_encode(p.outTF, g), sg_encode(p.outTF, b));
}

// Counter-based RNG: grain depends only on (seed, cell), so renders are
// deterministic, tile/thread independent, and identical on CPU and GPU.

SG_INLINE sg_u64 sg_mix64(sg_u64 x) {
  x += 0x9E3779B97F4A7C15ul;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;
  return x ^ (x >> 31);
}

SG_INLINE sg_u64 sg_cellKey(sg_u64 seedKey, sg_u32 gx, sg_u32 gy) {
  const sg_u64 row = sg_mix64(seedKey + (sg_u64)gy * 0xD1B54A32D192ED03ul);
  return sg_mix64(row ^ ((sg_u64)gx * 0xA24BAED4963EE407ul));
}

// Irwin-Hall(4) standard normal approximation from one 64-bit hash.
SG_INLINE float sg_normal(sg_u64 h) {
  const float s = ((float)(sg_u32)(h & 0xFFFFul) + (float)(sg_u32)((h >> 16) & 0xFFFFul) +
                   (float)(sg_u32)((h >> 32) & 0xFFFFul) + (float)(sg_u32)(h >> 48) + 2.0f) *
                  (1.0f / 65536.0f);
  return (s - 2.0f) * 1.7320508f;
}

SG_INLINE float sg_uniform(sg_u64 h) { return ((float)(sg_u32)(h >> 40) + 0.5f) * (1.0f / 16777216.0f); }

// Poisson(mu) - mu: exact CDF inversion for small means; above that a
// Cornish-Fisher corrected Gaussian, which keeps Poisson's 1/sqrt(mu) skew.
SG_INLINE float sg_poissonDeviation(float mu, sg_u64 h) {
  if (mu >= SG_POISSON_EXACT_BELOW) {
    const float z = sg_normal(h);
    return sqrt(mu) * z + (z * z - 1.0f) * (1.0f / 6.0f);
  }
  const float u = sg_uniform(h);
  float pk = exp(-mu);
  float cdf = pk;
  int k = 0;
  while (u > cdf && k < 32) {
    ++k;
    pk *= mu / (float)k;
    cdf += pk;
  }
  return (float)k - mu;
}

// Film model

SG_INLINE float sg_lutEval(SG_LUT lut, SG_PARAMS p, float logE) {
  const float t = sg_clamp((logE - p.logE0) / p.logEStep, 0.0f, (float)(p.lutCount - 1));
  int i = (int)t;
  if (i > (int)p.lutCount - 2) i = (int)p.lutCount - 2;
  const float f = t - (float)i;
  return lut[i] + (lut[i + 1] - lut[i]) * f;
}

SG_INLINE float sg_pixelLogExposure(SG_PARAMS p, float lin) {
  const float le = log10(sg_max(lin, 1e-7f) / SG_MID_GRAY) + p.expBias;
  return sg_clamp(le, p.logE0, p.logEMax);
}

SG_INLINE sg_u64 sg_seedKey(SG_PARAMS p) { return ((sg_u64)p.seedHi << 32) | (sg_u64)p.seedLo; }

// Density deviation of sublayer i (= c*3+l) at one grain cell, from the
// sublayer's developed density.
SG_INLINE float sg_layerDeviation(SG_PARAMS p, int i, float layerDensity, sg_u64 cellKey) {
  const float invLambda = p.invLambda[i];
  if (invLambda <= 0.0f) {
    return 0.0f;
  }
  const float dtot = p.dtot[i];
  const float prob = sg_clamp((layerDensity + p.dmin[i]) / dtot, 1e-6f, 1.0f - 1e-6f);
  const float sat = 1.0f - prob * p.uniformity[i / 3];
  const float mu = prob / (sat * invLambda);
  const float od = dtot * invLambda;
  const sg_u64 h = sg_mix64(cellKey + (sg_u64)(i + 1) * 0x9E3779B97F4A7C15ul);
  return sg_poissonDeviation(mu, h) * od * sat;
}

// Density deviation -> exposure ratio via the curve slope, applied in linear
// DWG. With Match Input the grain goes on as an offset so the image itself is
// never re-encoded (zero grain is a bit-exact passthrough).
SG_INLINE SGRgb sg_applyGrain(SG_PARAMS p, SGRgb in, SGRgb lin, SGRgb dev, SGRgb slope) {
  const SGRgb ratio = sg_rgb(exp2(p.gain * dev.r / slope.r * SG_LOG2_10),
                             exp2(p.gain * dev.g / slope.g * SG_LOG2_10),
                             exp2(p.gain * dev.b / slope.b * SG_LOG2_10));
  if (p.grainOnly != 0) {
    return sg_fromWorking(p, sg_rgb(SG_MID_GRAY * ratio.r, SG_MID_GRAY * ratio.g, SG_MID_GRAY * ratio.b));
  }
  const SGRgb grained = sg_fromWorking(p, sg_rgb(lin.r * ratio.r, lin.g * ratio.g, lin.b * ratio.b));
  if (p.matchInput != 0) {
    const SGRgb plain = sg_fromWorking(p, lin);
    return sg_rgb(in.r + (grained.r - plain.r), in.g + (grained.g - plain.g), in.b + (grained.b - plain.b));
  }
  return grained;
}

#ifndef __METAL_VERSION__
}  // namespace spektragrain
#endif

#endif  // SPEKTRAGRAIN_SHARED_H
