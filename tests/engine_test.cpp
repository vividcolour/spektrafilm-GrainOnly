// Engine checks: colour space round trips, Match Input neutrality, RMS
// calibration vs the stock target, aspect ratio, and a few sample renders.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/GrainEngine.h"
#include "../src/StockData.h"

using namespace spektragrain;

static int failures = 0;
#define CHECK(cond, ...)              \
  do {                                \
    if (!(cond)) {                    \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");              \
      ++failures;                     \
    }                                 \
  } while (0)

static int spaceIndex(const char *label) {
  for (int i = 0; i < kColourSpaceCount; ++i)
    if (std::string(kColourSpaces[i].label) == label) return i;
  std::printf("FAIL: no colour space %s\n", label);
  ++failures;
  return 0;
}

static void testColourSpaces() {
  // Transfer function round trips, every space.
  for (int e = 0; e < kColourSpaceCount; ++e) {
    const sg_u32 tf = kColourSpaces[e].tf;
    double worst = 0.0;
    for (int i = 1; i < 100; ++i) {
      const float v = 0.05f + 0.9f * i / 100.0f;
      worst = std::max(worst, static_cast<double>(std::fabs(encode(tf, decode(tf, v)) - v)));
    }
    CHECK(worst < 2e-4, "%s transfer round trip error %g", kColourSpaces[e].label, worst);
    // Gamut round trip: space -> DWG -> space.
    double a[9], b[9], m[9];
    cs_rgbToRgb(e, kSpaceDWG, a);
    cs_rgbToRgb(kSpaceDWG, e, b);
    cs_mul(b, a, m);
    double off = 0.0;
    for (int i = 0; i < 9; ++i) off = std::max(off, std::fabs(m[i] - (i % 4 == 0 ? 1.0 : 0.0)));
    CHECK(off < 1e-9, "%s gamut round trip error %g", kColourSpaces[e].label, off);
  }
  // Mid gray sanity: linear 0.18 should land on documented code values.
  CHECK(std::fabs(encode(SG_TF_ACESCCT, 0.18f) - 0.4135f) < 2e-3, "ACEScct mid gray");
  CHECK(std::fabs(encode(SG_TF_LOGC3, 0.18f) - 0.3910f) < 2e-3, "LogC3 mid gray");
  CHECK(std::fabs(encode(SG_TF_SLOG3, 0.18f) - 0.4106f) < 2e-3, "S-Log3 mid gray");
  CHECK(std::fabs(encode(SG_TF_DAVINCI_INT, 0.18f) - 0.3360f) < 2e-3, "DI mid gray");
  CHECK(std::fabs(encode(SG_TF_LOGC4, 0.18f) - 0.2784f) < 2e-3, "LogC4 mid gray");
  CHECK(std::fabs(encode(SG_TF_VLOG, 0.18f) - 0.4233f) < 2e-3, "V-Log mid gray");
  CHECK(std::fabs(encode(SG_TF_LOG3G10, 0.18f) - 0.3333f) < 2e-3, "Log3G10 mid gray");
}

// Match Input: exact passthrough with no grain, unchanged average with grain.
static void testNeutral() {
  const int W = 256, H = 256;
  const float patches[][3] = {{0.02f, 0.05f, 0.10f}, {0.336f, 0.336f, 0.336f}, {0.45f, 0.30f, 0.20f},
                              {0.80f, 0.65f, 0.90f}, {1.30f, 0.95f, 0.60f},   {-0.03f, 0.01f, 0.20f}};
  GrainSettings s;
  s.stock = kDefaultStock;
  s.rmsTarget = kStocks[s.stock].rmsTarget;
  s.frameWidthPx = W;
  double worstPass = 0.0, worstMean = 0.0;
  for (int space = 0; space < kColourSpaceCount; ++space) {
    s.inputSpace = space;
    s.outputSpace = kOutputMatchInput;
    for (const auto &pc : patches) {
      std::vector<float> img(static_cast<size_t>(W) * H * 4), out(img.size());
      for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) {
        for (int c = 0; c < 3; ++c) img[i * 4 + c] = pc[c];
        img[i * 4 + 3] = 0.7f;
      }
      ImageRef in{img.data(), static_cast<ptrdiff_t>(W * 4 * sizeof(float)), 4};
      MutableImageRef o{out.data(), static_cast<ptrdiff_t>(W * 4 * sizeof(float)), 4};
      s.amount = 0.0f;
      renderGrain(s, in, o, W, H);
      for (size_t i = 0; i < out.size(); ++i)
        worstPass = std::max(worstPass, static_cast<double>(std::fabs(out[i] - img[i])));
      if (space == kSpaceDWG) {
        s.amount = 1.0f;
        renderGrain(s, in, o, W, H);
        for (int c = 0; c < 3; ++c) {
          double sum = 0.0;
          for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) sum += out[i * 4 + c];
          worstMean = std::max(worstMean, std::fabs(sum / (static_cast<double>(W) * H) - pc[c]));
        }
        CHECK(out[3] == 0.7f, "alpha changed");
      }
    }
  }
  std::printf("  match input: passthrough max error %g, mean colour max drift %g\n", worstPass, worstMean);
  CHECK(worstPass == 0.0, "no-grain passthrough not exact (%g)", worstPass);
  CHECK(worstMean < 1.5e-3, "average colour drift %g", worstMean);

  // Output != Input converts: DWG/Intermediate mid gray -> Rec.709 Gamma 2.4.
  s.inputSpace = kSpaceDWG;
  s.outputSpace = spaceIndex("Rec.709 / Gamma 2.4");
  s.amount = 0.0f;
  std::vector<float> g(16 * 16 * 4, 0.336f), go(g.size());
  renderGrain(s, ImageRef{g.data(), 16 * 4 * sizeof(float), 4}, MutableImageRef{go.data(), 16 * 4 * sizeof(float), 4},
              16, 16);
  const float want = std::pow(0.18f, 1.0f / 2.4f);
  CHECK(std::fabs(go[0] - want) < 2e-3 && std::fabs(go[1] - want) < 2e-3 && std::fabs(go[2] - want) < 2e-3,
        "DWG -> 709 mid gray %g %g %g (want %g)", go[0], go[1], go[2], want);
}

// Flat field at the calibration exposure; per-pixel density noise converted to
// a 48 um aperture via the Selwyn relation (white noise, no blur).
static void testCalibration() {
  const int W = 1024, H = 1024;
  for (int stock = 0; stock < kStockCount; ++stock) {
    GrainSettings s;
    s.stock = stock;
    s.rmsTarget = kStocks[stock].rmsTarget;
    s.softnessPx = 0.0f;
    s.inputSpace = kSpaceDWGLinear;
    s.frameWidthMm = 24.89f;
    const float pixelUm = s.frameWidthMm * 1000.0f / W;
    double meanVar = 0.0;
    for (int c = 0; c < 3; ++c) {
      const float le = calibrationLogExposure(stock, c);
      const float lin = 0.184f * std::pow(10.0f, le);
      std::vector<float> img(static_cast<size_t>(W) * H * 4, lin), out(img.size());
      renderGrain(s, {img.data(), W * 16, 4}, {out.data(), W * 16, 4}, W, H);
      const float slope = std::fabs(curveSlope(stock, c, le));
      double sum = 0, sum2 = 0;
      for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) {
        const double d = std::log10(out[i * 4 + c] / lin) * slope;
        sum += d;
        sum2 += d * d;
      }
      const double n = static_cast<double>(W) * H;
      const double var = sum2 / n - (sum / n) * (sum / n);
      const double rms48 = std::sqrt(var * pixelUm * pixelUm / (3.14159265 * 24 * 24)) * 1000.0;
      const double analytic = modelRmsAtUnitDensity(s, c);
      CHECK(std::fabs(rms48 / analytic - 1.0) < 0.03, "%s ch%d MC rms %.3f vs analytic %.3f", kStocks[stock].name, c,
            rms48, analytic);
      meanVar += rms48 * rms48 / 3.0;
    }
    const double rmsMean = std::sqrt(meanVar);
    std::printf("  %-28s target %5.2f  measured %5.2f\n", kStocks[stock].name, kStocks[stock].rmsTarget, rmsMean);
    CHECK(std::fabs(rmsMean / kStocks[stock].rmsTarget - 1.0) < 0.03, "%s calibration", kStocks[stock].name);
  }
}

// Per-pixel std should stay put across aspect ratios; correlation should
// stretch along x for aspect > 1 and along y for aspect < 1.
static void testAspect() {
  const int W = 768, H = 768;
  double baseStd = 0.0;
  for (float aspect : {1.0f, 0.0f, 0.3f, 0.6f, 1.5f, 2.0f}) {
    GrainSettings s;
    s.stock = kDefaultStock;
    s.rmsTarget = kStocks[s.stock].rmsTarget;
    s.inputSpace = kSpaceDWGLinear;
    s.aspect = aspect;
    s.softnessPx = 0.6f;
    s.saturation = 0.0f;
    std::vector<float> img(static_cast<size_t>(W) * H * 4, 0.184f), out(img.size());
    renderGrain(s, {img.data(), W * 16, 4}, {out.data(), W * 16, 4}, W, H);
    std::vector<double> g(static_cast<size_t>(W) * H);
    double sum = 0, sum2 = 0;
    for (size_t i = 0; i < g.size(); ++i) {
      g[i] = std::log10(out[i * 4 + 1] / 0.184f);
      sum += g[i];
      sum2 += g[i] * g[i];
    }
    const double n = static_cast<double>(g.size());
    const double mean = sum / n, var = sum2 / n - mean * mean;
    auto corr = [&](int dx, int dy) {
      double acc = 0;
      int cnt = 0;
      for (int y = 8; y < H - 8; ++y)
        for (int x = 8; x < W - 8; ++x) {
          acc += (g[static_cast<size_t>(y) * W + x] - mean) * (g[static_cast<size_t>(y + dy) * W + x + dx] - mean);
          ++cnt;
        }
      return acc / cnt / var;
    };
    const double cx = corr(1, 0), cy = corr(0, 1);
    const double sd = std::sqrt(var);
    if (aspect == 1.0f) baseStd = sd;
    std::printf("  aspect %.2f  std %.5f (x%.3f)  corr x %.3f  y %.3f\n", aspect, sd, sd / baseStd, cx, cy);
    CHECK(std::fabs(sd / baseStd - 1.0) < 0.06, "aspect %.2f changes grain strength by %.3f", aspect, sd / baseStd);
    if (aspect > 1.01f) CHECK(cx > cy + 0.05, "aspect %.2f should stretch horizontally", aspect);
    if (aspect < 0.99f) CHECK(cy > cx + 0.05, "aspect %.2f should stretch vertically", aspect);
  }
}

static void writeTestRender(const char *path, const GrainSettings &s, int W, int H) {
  // Rec.709-encoded chart: gray ramp on top, colour patches below, a soft
  // portrait-like gradient at the bottom.
  std::vector<float> img(static_cast<size_t>(W) * H * 4);
  const float patches[6][3] = {{0.75f, 0.3f, 0.25f}, {0.3f, 0.6f, 0.3f}, {0.25f, 0.35f, 0.7f},
                               {0.85f, 0.7f, 0.55f}, {0.2f, 0.2f, 0.2f}, {0.9f, 0.9f, 0.9f}};
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      float *p = &img[(static_cast<size_t>(y) * W + x) * 4];
      const float t = static_cast<float>(x) / (W - 1);
      const int band = y * 3 / H;  // OFX rows are bottom-up; written top-down below
      if (band == 2) {
        p[0] = p[1] = p[2] = t;
      } else if (band == 1) {
        const float *c = patches[std::min(5, x * 6 / W)];
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
      } else {
        const float v = 0.15f + 0.7f * (0.5f + 0.5f * std::sin(t * 6.283f)) * (static_cast<float>(y) / (H / 3));
        p[0] = v;
        p[1] = v * 0.82f;
        p[2] = v * 0.7f;
      }
      p[3] = 1.0f;
    }
  // Linear 0..1 test chart lives in Rec.709 gamma 2.4 code values.
  const auto t0 = std::chrono::steady_clock::now();
  renderGrain(s, {img.data(), W * 16, 4}, {img.data(), W * 16, 4}, W, H);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  %s  %dx%d  %.1f ms\n", path, W, H, ms);
  FILE *f = std::fopen(path, "wb");
  std::fprintf(f, "P6\n%d %d\n255\n", W, H);
  for (int y = H - 1; y >= 0; --y)
    for (int x = 0; x < W; ++x)
      for (int c = 0; c < 3; ++c) {
        const float v = std::min(std::max(img[(static_cast<size_t>(y) * W + x) * 4 + c], 0.0f), 1.0f);
        std::fputc(static_cast<int>(v * 255.0f + 0.5f), f);
      }
  std::fclose(f);
}

int main(int argc, char **argv) {
  const std::string outDir = argc > 1 ? argv[1] : ".";
  std::printf("colour spaces\n");
  testColourSpaces();
  std::printf("neutrality\n");
  testNeutral();
  std::printf("calibration (diffuse RMS x1000)\n");
  testCalibration();
  std::printf("aspect ratio\n");
  testAspect();

  std::printf("renders\n");
  GrainSettings s;
  s.stock = kDefaultStock;
  s.rmsTarget = kStocks[s.stock].rmsTarget;
  s.inputSpace = spaceIndex("Rec.709 / Gamma 2.4");
  writeTestRender((outDir + "/render_250d_s35_4k.ppm").c_str(), s, 3840, 2160);
  s.frameWidthMm = 12.52f;
  writeTestRender((outDir + "/render_250d_s16_hd.ppm").c_str(), s, 1920, 1080);
  s.aspect = 2.0f;
  writeTestRender((outDir + "/render_250d_s16_hd_aspect2.ppm").c_str(), s, 1920, 1080);
  s.aspect = 0.3f;
  writeTestRender((outDir + "/render_250d_s16_hd_aspect03.ppm").c_str(), s, 1920, 1080);
  s.aspect = 1.0f;
  s.stock = 13;  // Ektachrome
  s.rmsTarget = kStocks[s.stock].rmsTarget;
  writeTestRender((outDir + "/render_e100_s16_hd.ppm").c_str(), s, 1920, 1080);
  s.frameWidthMm = 5.79f;
  s.stock = 3;  // 500T
  s.rmsTarget = kStocks[s.stock].rmsTarget;
  writeTestRender((outDir + "/render_500t_s8_hd.ppm").c_str(), s, 1920, 1080);

  std::printf(failures ? "\n%d FAILURES\n" : "\nall checks passed\n", failures);
  return failures ? 1 : 0;
}
