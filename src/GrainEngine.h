// RMS-calibrated film grain, after spektrafilm's sublayer particle model.
#pragma once

#include <cstddef>
#include <cstdint>

#include "Transfer.h"

namespace spektragrain {

struct FilmFormat {
  const char *label;
  float frameWidthMm;
};

static const FilmFormat kFilmFormats[] = {
  {"Super 8 (5.79 mm)", 5.79f},
  {"16 mm (10.26 mm)", 10.26f},
  {"Super 16 (12.52 mm)", 12.52f},
  {"35 mm Academy (21.95 mm)", 21.95f},
  {"Super 35 (24.89 mm)", 24.89f},
  {"35 mm Full Frame (36 mm)", 36.0f},
  {"65 mm (52.48 mm)", 52.48f},
  {"IMAX 15/70 (70.41 mm)", 70.41f},
};
static const int kFilmFormatCount = sizeof(kFilmFormats) / sizeof(kFilmFormats[0]);
static const int kDefaultFilmFormat = 4;

enum class ViewMode : int { Final = 0, GrainOnly = 1 };

struct GrainSettings {
  int stock = 0;
  float rmsTarget = 5.0f;       // diffuse RMS granularity x1000 at D=1.0, 48 um aperture
  float amount = 1.0f;          // multiplier on the calibrated grain
  float aspect = 1.0f;          // grain width / height (0..2, 1 = round)
  float size = 1.0f;            // >1 enlarges grain relative to the frame
  float softnessPx = 0.6f;      // scanner blur sigma, full-resolution pixels
  float renderScale = 1.0f;     // host proxy scale (softness follows it)
  float saturation = 1.0f;      // 0 = monochrome grain
  float exposureStops = 0.0f;   // where the image sits on the film curve
  float frameWidthMm = 24.89f;
  int frameWidthPx = 0;         // full image width in pixels (0 = render width)
  uint32_t seed = 0;
  int inputSpace = kSpaceDWG;            // index into kColourSpaces
  int outputSpace = kOutputMatchInput;   // index into kColourSpaces, or Match Input
  ViewMode view = ViewMode::Final;
};

// Interleaved float image rows. rowBytes may be negative.
struct ImageRef {
  const void *data = nullptr;
  ptrdiff_t rowBytes = 0;
  int components = 4;  // 3 (RGB) or 4 (RGBA)
};

struct MutableImageRef {
  void *data = nullptr;
  ptrdiff_t rowBytes = 0;
  int components = 4;
};

// Renders grain over a width x height region. src and dst point at the
// region's first row (bottom row in OFX order) and may alias.
void renderGrain(const GrainSettings &settings, const ImageRef &src, const MutableImageRef &dst, int width,
                 int height);

// Diagnostics for tests: analytic diffuse RMS (x1000, 48 um aperture) at the
// calibration exposure, that exposure (log10, 0 = mid gray), and the signed
// curve slope used to convert density noise to exposure.
double modelRmsAtUnitDensity(const GrainSettings &settings, int channel);
float calibrationLogExposure(int stock, int channel);
float curveSlope(int stock, int channel, float logE);

}  // namespace spektragrain
