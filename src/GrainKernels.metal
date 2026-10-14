// Metal kernels for the grain node. GrainShared.h is prepended to this file at
// build time, so SGParams and all sg_* functions are available here.
//
// Buffers: planes are stored channel-major ([3][rows][cols]).
//   A, B  scratch planes, capacity >= 3 * max(gw*gh, W*gh, W*H)

// Per-channel log exposure of one source pixel, in linear DWG.
static inline SGRgb sg_srcLogE(constant SGParams &p, device const float *src, uint x, uint y) {
  device const float *px = src + p.srcOffset + y * p.srcStride + x * p.srcComponents;
  const SGRgb lin = sg_toWorking(p, px[0], px[1], px[2]);
  return sg_rgb(sg_pixelLogExposure(p, lin.r), sg_pixelLogExposure(p, lin.g), sg_pixelLogExposure(p, lin.b));
}

static inline SGRgb sg_lerpRgb(SGRgb a, SGRgb b, float t) {
  return sg_rgb(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t);
}

// Bilinear log exposure at a grain cell centre (non-identity grid).
static inline SGRgb sg_cellLogE(constant SGParams &p, device const float *src, uint gx, uint gy) {
  const float fx = sg_clamp(((float)gx + 0.5f) * p.scaleX - 0.5f, 0.0f, (float)(p.width - 1));
  const float fy = sg_clamp(((float)gy + 0.5f) * p.scaleY - 0.5f, 0.0f, (float)(p.height - 1));
  const uint x0 = min((uint)fx, p.width - 1), y0 = min((uint)fy, p.height - 1);
  const uint x1 = min(x0 + 1, p.width - 1), y1 = min(y0 + 1, p.height - 1);
  const float tx = fx - (float)x0, ty = fy - (float)y0;
  const SGRgb top = sg_lerpRgb(sg_srcLogE(p, src, x0, y0), sg_srcLogE(p, src, x1, y0), tx);
  const SGRgb bottom = sg_lerpRgb(sg_srcLogE(p, src, x0, y1), sg_srcLogE(p, src, x1, y1), tx);
  return sg_lerpRgb(top, bottom, ty);
}

kernel void sg_grain(constant SGParams &p [[buffer(0)]],
                     device const float *src [[buffer(1)]],
                     device const float *layerLut [[buffer(2)]],
                     device float *noise [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.gw || gid.y >= p.gh) return;
  const sg_u64 key = sg_cellKey(sg_seedKey(p), gid.x, gid.y);
  const uint cells = p.gw * p.gh;
  const uint idx = gid.y * p.gw + gid.x;
  const SGRgb les = p.identity != 0 ? sg_srcLogE(p, src, gid.x, gid.y) : sg_cellLogE(p, src, gid.x, gid.y);
  for (uint c = 0; c < 3; ++c) {
    const float le = sg_comp(les, (int)c);
    float sum = 0.0f;
    for (int l = 0; l < 3; ++l) {
      const int i = (int)c * 3 + l;
      sum += sg_layerDeviation(p, i, sg_lutEval(layerLut + i * p.lutCount, p, le), key);
    }
    noise[c * cells + idx] = sum;
  }
}

// Separable Gaussian, clamp-to-edge. weights: [3][taps], centred.
kernel void sg_blurH(constant SGParams &p [[buffer(0)]],
                     device const float *in [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     device const float *weights [[buffer(3)]],
                     constant uint &taps [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.gw || gid.y >= p.gh) return;
  const uint c = gid.z;
  const uint cells = p.gw * p.gh;
  device const float *row = in + c * cells + gid.y * p.gw;
  const int r = (int)p.blurRadius[c];
  device const float *w = weights + c * taps + (taps / 2);
  float acc = 0.0f;
  for (int j = -r; j <= r; ++j) {
    const int x = clamp((int)gid.x + j, 0, (int)p.gw - 1);
    acc += w[j] * row[x];
  }
  out[c * cells + gid.y * p.gw + gid.x] = r == 0 ? row[gid.x] : acc;
}

kernel void sg_blurV(constant SGParams &p [[buffer(0)]],
                     device const float *in [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     device const float *weights [[buffer(3)]],
                     constant uint &taps [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.gw || gid.y >= p.gh) return;
  const uint c = gid.z;
  const uint cells = p.gw * p.gh;
  device const float *plane = in + c * cells;
  const int r = (int)p.blurRadius[c];
  device const float *w = weights + c * taps + (taps / 2);
  float acc = 0.0f;
  for (int j = -r; j <= r; ++j) {
    const int y = clamp((int)gid.y + j, 0, (int)p.gh - 1);
    acc += w[j] * plane[y * p.gw + gid.x];
  }
  out[c * cells + gid.y * p.gw + gid.x] = r == 0 ? plane[gid.y * p.gw + gid.x] : acc;
}

// Chroma mix (grain saturation), in place.
kernel void sg_mix(constant SGParams &p [[buffer(0)]],
                   device float *noise [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.gw || gid.y >= p.gh) return;
  const uint cells = p.gw * p.gh;
  const uint idx = gid.y * p.gw + gid.x;
  const float r = noise[idx], g = noise[cells + idx], b = noise[2 * cells + idx];
  const float mean = (r + g + b) * (1.0f / 3.0f);
  noise[idx] = mean + p.satMix * (r - mean);
  noise[cells + idx] = mean + p.satMix * (g - mean);
  noise[2 * cells + idx] = mean + p.satMix * (b - mean);
}

// Resampling taps: ints = [start[n] | count[n] | offset[n]], weights flat.
kernel void sg_resampleH(constant SGParams &p [[buffer(0)]],
                         device const float *in [[buffer(1)]],   // [3][gh][gw]
                         device float *out [[buffer(2)]],        // [3][gh][W]
                         device const int *taps [[buffer(3)]],
                         device const float *weights [[buffer(4)]],
                         uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.gh) return;
  const uint c = gid.z;
  const uint n = p.width;
  const int start = taps[gid.x], count = taps[n + gid.x], offset = taps[2 * n + gid.x];
  device const float *row = in + c * p.gw * p.gh + gid.y * p.gw + start;
  float acc = 0.0f;
  for (int k = 0; k < count; ++k) acc += weights[offset + k] * row[k];
  out[c * p.width * p.gh + gid.y * p.width + gid.x] = acc;
}

kernel void sg_resampleV(constant SGParams &p [[buffer(0)]],
                         device const float *in [[buffer(1)]],   // [3][gh][W]
                         device float *out [[buffer(2)]],        // [3][H][W]
                         device const int *taps [[buffer(3)]],
                         device const float *weights [[buffer(4)]],
                         uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) return;
  const uint c = gid.z;
  const uint n = p.height;
  const int start = taps[gid.y], count = taps[n + gid.y], offset = taps[2 * n + gid.y];
  device const float *plane = in + c * p.width * p.gh;
  float acc = 0.0f;
  for (int k = 0; k < count; ++k) acc += weights[offset + k] * plane[(start + k) * p.width + gid.x];
  out[c * p.width * p.height + gid.y * p.width + gid.x] = acc;
}

// Density deviation -> exposure -> output colour space. noise is [3][H][W].
kernel void sg_final(constant SGParams &p [[buffer(0)]],
                     device const float *src [[buffer(1)]],
                     device float *dst [[buffer(2)]],
                     device const float *noise [[buffer(3)]],
                     device const float *slopeLut [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) return;
  device const float *pi = src + p.srcOffset + gid.y * p.srcStride + gid.x * p.srcComponents;
  device float *po = dst + p.dstOffset + gid.y * p.dstStride + gid.x * p.dstComponents;
  const uint plane = p.width * p.height;
  const uint idx = gid.y * p.width + gid.x;
  const SGRgb lin = sg_toWorking(p, pi[0], pi[1], pi[2]);
  const SGRgb slope = sg_rgb(sg_lutEval(slopeLut, p, sg_pixelLogExposure(p, lin.r)),
                             sg_lutEval(slopeLut + p.lutCount, p, sg_pixelLogExposure(p, lin.g)),
                             sg_lutEval(slopeLut + 2 * p.lutCount, p, sg_pixelLogExposure(p, lin.b)));
  const SGRgb dev = sg_rgb(noise[idx], noise[plane + idx], noise[2 * plane + idx]);
  const SGRgb rgb = sg_applyGrain(p, sg_rgb(pi[0], pi[1], pi[2]), lin, dev, slope);
  const float alpha = p.srcComponents == 4 ? pi[3] : 1.0f;
  po[0] = rgb.r;
  po[1] = rgb.g;
  po[2] = rgb.b;
  if (p.dstComponents == 4) po[3] = alpha;
}
