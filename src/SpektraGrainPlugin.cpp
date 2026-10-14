// Spektra RMS Grain: OpenFX plugin for DaVinci Resolve (macOS).
//
// A standalone grain node built on spektrafilm's sublayer particle grain model
// (https://github.com/andreavolpato/spektrafilm), calibrated to each stock's
// RMS granularity. Renders with Metal when the host offers it, else on the CPU.
//
// Copyright (C) 2026 vividcolour. Licensed under the GNU GPL v3 (see LICENSE).
// Film profile data: spektrafilm by Andrea Volpato, CC BY-SA 4.0.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "ofxCore.h"
#include "ofxGPURender.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#include "GrainEngine.h"
#include "GrainMetal.h"
#include "StockData.h"

using namespace spektragrain;

#define PLUGIN_ID "com.vividcolour.SpektraRMSGrain"
#define PLUGIN_VERSION_MAJOR 1
#define PLUGIN_VERSION_MINOR 1

namespace {

OfxHost *gHost = nullptr;
const OfxImageEffectSuiteV1 *gEffect = nullptr;
const OfxPropertySuiteV1 *gProp = nullptr;
const OfxParameterSuiteV1 *gParam = nullptr;
const OfxMessageSuiteV1 *gMessage = nullptr;

// Parameter names (stable: changing them breaks saved projects).
constexpr const char *kStock = "stock";
constexpr const char *kFormat = "filmFormat";
constexpr const char *kInSpace = "inputSpace";
constexpr const char *kOutSpace = "outputSpace";
constexpr const char *kAmount = "amount";
constexpr const char *kAspect = "aspectRatio";
constexpr const char *kSize = "grainSize";
constexpr const char *kSoftness = "softness";
constexpr const char *kSaturation = "saturation";
constexpr const char *kExposure = "exposureOffset";
constexpr const char *kCustomRms = "customRms";
constexpr const char *kRms = "rmsGranularity";
constexpr const char *kAnimate = "animate";
constexpr const char *kSeed = "seed";
constexpr const char *kView = "view";

constexpr const char *kGroupFilm = "groupFilm";
constexpr const char *kGroupGrain = "groupGrain";
constexpr const char *kGroupColour = "groupColourSpace";
constexpr const char *kGroupOutput = "groupOutput";

const char *const kViewLabels[] = {"Final Image", "Grain Only (on mid gray)"};

struct Instance {
  OfxImageClipHandle src = nullptr;
  OfxImageClipHandle dst = nullptr;
  OfxParamHandle stock = nullptr, format = nullptr, inSpace = nullptr, outSpace = nullptr, amount = nullptr, aspect = nullptr,
                 size = nullptr, softness = nullptr, saturation = nullptr, exposure = nullptr, customRms = nullptr,
                 rms = nullptr, animate = nullptr, seed = nullptr, view = nullptr;
  MetalScratch *metal = nullptr;
};

Instance *getInstance(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props = nullptr;
  gEffect->getPropertySet(effect, &props);
  void *ptr = nullptr;
  gProp->propGetPointer(props, kOfxPropInstanceData, 0, &ptr);
  return static_cast<Instance *>(ptr);
}

// Parameter definition helpers

OfxPropertySetHandle defineParam(OfxParamSetHandle set, const char *type, const char *name, const char *label,
                                 const char *hint, const char *parent) {
  OfxPropertySetHandle props = nullptr;
  gParam->paramDefine(set, type, name, &props);
  gProp->propSetString(props, kOfxPropLabel, 0, label);
  gProp->propSetString(props, kOfxParamPropScriptName, 0, name);
  if (hint) gProp->propSetString(props, kOfxParamPropHint, 0, hint);
  if (parent) gProp->propSetString(props, kOfxParamPropParent, 0, parent);
  return props;
}

void defineGroup(OfxParamSetHandle set, const char *name, const char *label, bool open) {
  OfxPropertySetHandle props = defineParam(set, kOfxParamTypeGroup, name, label, nullptr, nullptr);
  gProp->propSetInt(props, kOfxParamPropGroupOpen, 0, open ? 1 : 0);
}

void defineDouble(OfxParamSetHandle set, const char *name, const char *label, const char *hint, const char *parent,
                  double def, double min, double max, double displayMin, double displayMax, double increment,
                  int digits) {
  OfxPropertySetHandle props = defineParam(set, kOfxParamTypeDouble, name, label, hint, parent);
  gProp->propSetString(props, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypePlain);
  gProp->propSetDouble(props, kOfxParamPropDefault, 0, def);
  gProp->propSetDouble(props, kOfxParamPropMin, 0, min);
  gProp->propSetDouble(props, kOfxParamPropMax, 0, max);
  gProp->propSetDouble(props, kOfxParamPropDisplayMin, 0, displayMin);
  gProp->propSetDouble(props, kOfxParamPropDisplayMax, 0, displayMax);
  gProp->propSetDouble(props, kOfxParamPropIncrement, 0, increment);
  gProp->propSetInt(props, kOfxParamPropDigits, 0, digits);
}

void defineChoice(OfxParamSetHandle set, const char *name, const char *label, const char *hint, const char *parent,
                  const char *const *options, int count, int def) {
  OfxPropertySetHandle props = defineParam(set, kOfxParamTypeChoice, name, label, hint, parent);
  for (int i = 0; i < count; ++i) {
    gProp->propSetString(props, kOfxParamPropChoiceOption, i, options[i]);
  }
  gProp->propSetInt(props, kOfxParamPropDefault, 0, def);
  gProp->propSetInt(props, kOfxParamPropAnimates, 0, 0);
}

void defineBool(OfxParamSetHandle set, const char *name, const char *label, const char *hint, const char *parent,
                bool def) {
  OfxPropertySetHandle props = defineParam(set, kOfxParamTypeBoolean, name, label, hint, parent);
  gProp->propSetInt(props, kOfxParamPropDefault, 0, def ? 1 : 0);
  gProp->propSetInt(props, kOfxParamPropAnimates, 0, 0);
}

void defineInt(OfxParamSetHandle set, const char *name, const char *label, const char *hint, const char *parent,
               int def, int min, int max) {
  OfxPropertySetHandle props = defineParam(set, kOfxParamTypeInteger, name, label, hint, parent);
  gProp->propSetInt(props, kOfxParamPropDefault, 0, def);
  gProp->propSetInt(props, kOfxParamPropMin, 0, min);
  gProp->propSetInt(props, kOfxParamPropMax, 0, max);
  gProp->propSetInt(props, kOfxParamPropDisplayMin, 0, min);
  gProp->propSetInt(props, kOfxParamPropDisplayMax, 0, max);
}

// Parameter access

double getDouble(OfxParamHandle h, OfxTime t, double fallback) {
  double v = fallback;
  if (h) gParam->paramGetValueAtTime(h, t, &v);
  return v;
}

int getInt(OfxParamHandle h, OfxTime t, int fallback) {
  int v = fallback;
  if (h) gParam->paramGetValueAtTime(h, t, &v);
  return v;
}

void setEnabled(OfxParamHandle h, bool enabled) {
  OfxPropertySetHandle props = nullptr;
  if (h && gParam->paramGetPropertySet(h, &props) == kOfxStatOK) {
    gProp->propSetInt(props, kOfxParamPropEnabled, 0, enabled ? 1 : 0);
  }
}

// With "Custom RMS" off the RMS field mirrors the stock's data-sheet value.
void syncRmsParam(Instance *inst) {
  int custom = 0, stock = kDefaultStock;
  gParam->paramGetValue(inst->customRms, &custom);
  gParam->paramGetValue(inst->stock, &stock);
  setEnabled(inst->rms, custom != 0);
  if (!custom) {
    const double stockRms = kStocks[std::clamp(stock, 0, kStockCount - 1)].rmsTarget;
    double current = 0.0;
    gParam->paramGetValue(inst->rms, &current);
    if (std::fabs(current - stockRms) > 1e-6) {
      gParam->paramSetValue(inst->rms, stockRms);
    }
  }
}

GrainSettings readSettings(Instance *inst, OfxTime time) {
  GrainSettings s;
  s.stock = std::clamp(getInt(inst->stock, time, kDefaultStock), 0, kStockCount - 1);
  s.frameWidthMm = kFilmFormats[std::clamp(getInt(inst->format, time, kDefaultFilmFormat), 0, kFilmFormatCount - 1)]
                       .frameWidthMm;
  s.inputSpace = std::clamp(getInt(inst->inSpace, time, kSpaceDWG), 0, kColourSpaceCount - 1);
  // Output menu: 0 = Match Input, then the colour spaces.
  const int outChoice = getInt(inst->outSpace, time, 0);
  s.outputSpace = outChoice <= 0 ? kOutputMatchInput : std::min(outChoice - 1, kColourSpaceCount - 1);
  s.amount = static_cast<float>(getDouble(inst->amount, time, 1.0));
  s.aspect = static_cast<float>(getDouble(inst->aspect, time, 1.0));
  s.size = static_cast<float>(getDouble(inst->size, time, 1.0));
  s.softnessPx = static_cast<float>(getDouble(inst->softness, time, 0.6));
  s.saturation = static_cast<float>(getDouble(inst->saturation, time, 1.0));
  s.exposureStops = static_cast<float>(getDouble(inst->exposure, time, 0.0));
  s.rmsTarget = getInt(inst->customRms, time, 0) ? static_cast<float>(getDouble(inst->rms, time, 5.0))
                                                 : kStocks[s.stock].rmsTarget;
  s.view = getInt(inst->view, time, 0) == 1 ? ViewMode::GrainOnly : ViewMode::Final;
  const uint32_t userSeed = static_cast<uint32_t>(std::max(0, getInt(inst->seed, time, 0)));
  if (getInt(inst->animate, time, 1)) {
    const int64_t frame = static_cast<int64_t>(std::floor(time + 0.5));
    s.seed = userSeed * 1000003u + static_cast<uint32_t>(frame) * 2654435761u + 1u;
  } else {
    s.seed = userSeed * 1000003u;
  }
  return s;
}

// Actions

OfxStatus onLoad() {
  if (!gHost) return kOfxStatErrMissingHostFeature;
  gEffect = static_cast<const OfxImageEffectSuiteV1 *>(gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1));
  gProp = static_cast<const OfxPropertySuiteV1 *>(gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1));
  gParam = static_cast<const OfxParameterSuiteV1 *>(gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1));
  gMessage = static_cast<const OfxMessageSuiteV1 *>(gHost->fetchSuite(gHost->host, kOfxMessageSuite, 1));
  return (gEffect && gProp && gParam) ? kOfxStatOK : kOfxStatErrMissingHostFeature;
}

OfxStatus describe(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props = nullptr;
  gEffect->getPropertySet(effect, &props);
  gProp->propSetString(props, kOfxPropLabel, 0, "Spektra RMS Grain");
  gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "VividColour");
  gProp->propSetString(props, kOfxPropPluginDescription, 0,
                       "RMS-calibrated film grain after spektrafilm's sublayer particle model. "
                       "Film modeling inspired by spektrafilm (Andrea Volpato). GPL-3.0.");
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsMultiResolution, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPluginPropSingleInstance, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsMultipleClipDepths, 0, 0);
  gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);
  gProp->propSetString(props, kOfxImageEffectPropMetalRenderSupported, 0, "true");
  return kOfxStatOK;
}

OfxStatus describeInContext(OfxImageEffectHandle effect) {
  OfxPropertySetHandle clip = nullptr;
  gEffect->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &clip);
  gProp->propSetString(clip, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  gProp->propSetString(clip, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
  gProp->propSetInt(clip, kOfxImageClipPropIsMask, 0, 0);
  gProp->propSetInt(clip, kOfxImageEffectPropSupportsTiles, 0, 0);

  gEffect->clipDefine(effect, kOfxImageEffectOutputClipName, &clip);
  gProp->propSetString(clip, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  gProp->propSetString(clip, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
  gProp->propSetInt(clip, kOfxImageEffectPropSupportsTiles, 0, 0);

  OfxParamSetHandle set = nullptr;
  gEffect->getParamSet(effect, &set);

  defineGroup(set, kGroupFilm, "Film", true);
  const char *stocks[kStockCount];
  for (int i = 0; i < kStockCount; ++i) stocks[i] = kStocks[i].name;
  defineChoice(set, kStock, "Film Stock",
               "Stock whose characteristic curves and RMS granularity drive the grain.", kGroupFilm, stocks,
               kStockCount, kDefaultStock);
  const char *formats[kFilmFormatCount];
  for (int i = 0; i < kFilmFormatCount; ++i) formats[i] = kFilmFormats[i].label;
  defineChoice(set, kFormat, "Film Format",
               "Negative frame width. Smaller formats enlarge the grain relative to the picture.", kGroupFilm,
               formats, kFilmFormatCount, kDefaultFilmFormat);
  defineBool(set, kCustomRms, "Custom RMS",
             "Override the stock's data-sheet RMS granularity with your own value.", kGroupFilm, false);
  defineDouble(set, kRms, "RMS Granularity (x1000)",
               "Diffuse RMS granularity at density 1.0 through a 48 um aperture. Follows the stock unless "
               "Custom RMS is on.",
               kGroupFilm, kStocks[kDefaultStock].rmsTarget, 0.5, 40.0, 1.0, 20.0, 0.1, 2);
  defineDouble(set, kExposure, "Exposure Offset (stops)",
               "Shifts where the image sits on the stock's curve. Overexposing a negative gives finer looking "
               "grain, underexposing coarser.",
               kGroupFilm, 0.0, -6.0, 6.0, -4.0, 4.0, 0.1, 2);

  defineGroup(set, kGroupGrain, "Grain", true);
  defineDouble(set, kAmount, "Grain Amount", "Multiplier on the RMS-calibrated grain. 1 = stock data sheet.",
               kGroupGrain, 1.0, 0.0, 10.0, 0.0, 3.0, 0.01, 2);
  defineDouble(set, kAspect, "Grain Aspect Ratio",
               "Grain width relative to height. 1.00 = round, above 1 stretches horizontally, below 1 vertically.",
               kGroupGrain, 1.0, 0.0, 2.0, 0.0, 2.0, 0.01, 2);
  defineDouble(set, kSize, "Grain Size", "Scales grain relative to the frame, as if shot on a smaller format.",
               kGroupGrain, 1.0, 0.25, 4.0, 0.25, 4.0, 0.01, 2);
  defineDouble(set, kSoftness, "Grain Softness", "Scanner blur on the grain, in pixels.", kGroupGrain, 0.6, 0.0,
               4.0, 0.0, 3.0, 0.01, 2);
  defineDouble(set, kSaturation, "Grain Saturation", "0 = monochrome grain, 1 = dye-layer colour grain.",
               kGroupGrain, 1.0, 0.0, 2.0, 0.0, 2.0, 0.01, 2);
  defineBool(set, kAnimate, "Animate Grain", "New grain pattern every frame.", kGroupGrain, true);
  defineInt(set, kSeed, "Seed", "Changes the grain pattern.", kGroupGrain, 0, 0, 9999);

  defineGroup(set, kGroupColour, "Colour Space", true);
  const char *inOpts[kColourSpaceCount];
  for (int i = 0; i < kColourSpaceCount; ++i) inOpts[i] = kColourSpaces[i].label;
  defineChoice(set, kInSpace, "Input",
               "Colour space of the incoming image. Internally the grain is always computed in linear "
               "DaVinci Wide Gamut.",
               kGroupColour, inOpts, kColourSpaceCount, kSpaceDWG);
  const char *outOpts[kColourSpaceCount + 1];
  outOpts[0] = "Match Input";
  for (int i = 0; i < kColourSpaceCount; ++i) outOpts[i + 1] = kColourSpaces[i].label;
  defineChoice(set, kOutSpace, "Output", "Colour space written out. Match Input keeps the node colour-neutral.",
               kGroupColour, outOpts, kColourSpaceCount + 1, 0);

  defineGroup(set, kGroupOutput, "Output", true);
  defineChoice(set, kView, "View", "Final image, or the grain alone on mid gray for checking or compositing.",
               kGroupOutput, kViewLabels, 2, 0);
  return kOfxStatOK;
}

OfxStatus createInstance(OfxImageEffectHandle effect) {
  Instance *inst = new Instance();
  gEffect->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &inst->src, nullptr);
  gEffect->clipGetHandle(effect, kOfxImageEffectOutputClipName, &inst->dst, nullptr);
  OfxParamSetHandle set = nullptr;
  gEffect->getParamSet(effect, &set);
  gParam->paramGetHandle(set, kStock, &inst->stock, nullptr);
  gParam->paramGetHandle(set, kFormat, &inst->format, nullptr);
  gParam->paramGetHandle(set, kInSpace, &inst->inSpace, nullptr);
  gParam->paramGetHandle(set, kOutSpace, &inst->outSpace, nullptr);
  gParam->paramGetHandle(set, kAmount, &inst->amount, nullptr);
  gParam->paramGetHandle(set, kAspect, &inst->aspect, nullptr);
  gParam->paramGetHandle(set, kSize, &inst->size, nullptr);
  gParam->paramGetHandle(set, kSoftness, &inst->softness, nullptr);
  gParam->paramGetHandle(set, kSaturation, &inst->saturation, nullptr);
  gParam->paramGetHandle(set, kExposure, &inst->exposure, nullptr);
  gParam->paramGetHandle(set, kCustomRms, &inst->customRms, nullptr);
  gParam->paramGetHandle(set, kRms, &inst->rms, nullptr);
  gParam->paramGetHandle(set, kAnimate, &inst->animate, nullptr);
  gParam->paramGetHandle(set, kSeed, &inst->seed, nullptr);
  gParam->paramGetHandle(set, kView, &inst->view, nullptr);
  inst->metal = createMetalScratch();

  OfxPropertySetHandle props = nullptr;
  gEffect->getPropertySet(effect, &props);
  gProp->propSetPointer(props, kOfxPropInstanceData, 0, inst);
  syncRmsParam(inst);
  return kOfxStatOK;
}

OfxStatus destroyInstance(OfxImageEffectHandle effect) {
  Instance *inst = getInstance(effect);
  if (inst) {
    destroyMetalScratch(inst->metal);
    delete inst;
  }
  return kOfxStatOK;
}

OfxStatus instanceChanged(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
  Instance *inst = getInstance(effect);
  if (!inst) return kOfxStatReplyDefault;
  char *type = nullptr, *name = nullptr;
  gProp->propGetString(inArgs, kOfxPropType, 0, &type);
  gProp->propGetString(inArgs, kOfxPropName, 0, &name);
  if (type && name && std::strcmp(type, kOfxTypeParameter) == 0 &&
      (std::strcmp(name, kStock) == 0 || std::strcmp(name, kCustomRms) == 0)) {
    syncRmsParam(inst);
    return kOfxStatOK;
  }
  return kOfxStatReplyDefault;
}

OfxStatus getClipPreferences(OfxImageEffectHandle effect, OfxPropertySetHandle outArgs) {
  Instance *inst = getInstance(effect);
  int animate = 1;
  if (inst) gParam->paramGetValue(inst->animate, &animate);
  gProp->propSetInt(outArgs, kOfxImageEffectFrameVarying, 0, animate ? 1 : 0);
  return kOfxStatOK;
}

OfxStatus isIdentity(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  Instance *inst = getInstance(effect);
  if (!inst) return kOfxStatReplyDefault;
  OfxTime time = 0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);
  const int outChoice = getInt(inst->outSpace, time, 0);
  const bool sameSpace = outChoice <= 0 || outChoice - 1 == getInt(inst->inSpace, time, kSpaceDWG);
  if (getDouble(inst->amount, time, 1.0) <= 0.0 && getInt(inst->view, time, 0) == 0 && sameSpace) {
    gProp->propSetString(outArgs, kOfxPropName, 0, kOfxImageEffectSimpleSourceClipName);
    return kOfxStatOK;
  }
  return kOfxStatReplyDefault;
}

struct ImageInfo {
  OfxPropertySetHandle handle = nullptr;
  void *data = nullptr;
  OfxRectI bounds{};
  int rowBytes = 0;
  int components = 4;
  bool isFloat = false;
};

bool fetchImage(OfxImageClipHandle clip, OfxTime time, ImageInfo &out) {
  if (gEffect->clipGetImage(clip, time, nullptr, &out.handle) != kOfxStatOK || !out.handle) return false;
  gProp->propGetPointer(out.handle, kOfxImagePropData, 0, &out.data);
  gProp->propGetIntN(out.handle, kOfxImagePropBounds, 4, &out.bounds.x1);
  gProp->propGetInt(out.handle, kOfxImagePropRowBytes, 0, &out.rowBytes);
  char *depth = nullptr, *comps = nullptr;
  gProp->propGetString(out.handle, kOfxImageEffectPropPixelDepth, 0, &depth);
  gProp->propGetString(out.handle, kOfxImageEffectPropComponents, 0, &comps);
  out.isFloat = depth && std::strcmp(depth, kOfxBitDepthFloat) == 0;
  out.components = comps && std::strcmp(comps, kOfxImageComponentRGB) == 0 ? 3 : 4;
  if (out.rowBytes == 0) {
    out.rowBytes = (out.bounds.x2 - out.bounds.x1) * out.components * static_cast<int>(sizeof(float));
  }
  return out.data != nullptr;
}

void postError(OfxImageEffectHandle effect, const std::string &message) {
  if (gMessage) {
    gMessage->message(effect, kOfxMessageError, nullptr, "Spektra RMS Grain: %s", message.c_str());
  }
}

OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
  Instance *inst = getInstance(effect);
  if (!inst) return kOfxStatFailed;
  OfxTime time = 0;
  OfxRectI window{};
  double renderScale[2] = {1.0, 1.0};
  int metalEnabled = 0;
  void *metalQueue = nullptr;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);
  gProp->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &window.x1);
  gProp->propGetDoubleN(inArgs, kOfxImageEffectPropRenderScale, 2, renderScale);
  if (gProp->propGetInt(inArgs, kOfxImageEffectPropMetalEnabled, 0, &metalEnabled) != kOfxStatOK) metalEnabled = 0;
  if (metalEnabled) gProp->propGetPointer(inArgs, kOfxImageEffectPropMetalCommandQueue, 0, &metalQueue);

  ImageInfo src, dst;
  if (!fetchImage(inst->dst, time, dst)) {
    if (dst.handle) gEffect->clipReleaseImage(dst.handle);
    return kOfxStatFailed;
  }
  if (!fetchImage(inst->src, time, src)) {
    if (src.handle) gEffect->clipReleaseImage(src.handle);
    gEffect->clipReleaseImage(dst.handle);
    return kOfxStatFailed;
  }

  OfxStatus status = kOfxStatOK;
  const OfxRectI region{std::max({window.x1, src.bounds.x1, dst.bounds.x1}),
                        std::max({window.y1, src.bounds.y1, dst.bounds.y1}),
                        std::min({window.x2, src.bounds.x2, dst.bounds.x2}),
                        std::min({window.y2, src.bounds.y2, dst.bounds.y2})};
  const int width = region.x2 - region.x1, height = region.y2 - region.y1;

  if (!src.isFloat || !dst.isFloat) {
    status = kOfxStatErrUnsupported;
  } else if (width > 0 && height > 0) {
    GrainSettings settings = readSettings(inst, time);
    settings.renderScale = static_cast<float>(renderScale[0]);
    settings.frameWidthPx = src.bounds.x2 - src.bounds.x1;

    if (metalEnabled && metalQueue) {
      MetalImageLayout sl, dl;
      sl.components = src.components;
      dl.components = dst.components;
      sl.strideFloats = static_cast<size_t>(src.rowBytes) / sizeof(float);
      dl.strideFloats = static_cast<size_t>(dst.rowBytes) / sizeof(float);
      sl.offsetFloats = static_cast<size_t>(region.y1 - src.bounds.y1) * sl.strideFloats +
                        static_cast<size_t>(region.x1 - src.bounds.x1) * src.components;
      dl.offsetFloats = static_cast<size_t>(region.y1 - dst.bounds.y1) * dl.strideFloats +
                        static_cast<size_t>(region.x1 - dst.bounds.x1) * dst.components;
      std::string error;
      if (!renderGrainMetal(settings, inst->metal, metalQueue, src.data, sl, dst.data, dl, width, height, false,
                            &error)) {
        postError(effect, error);
        status = kOfxStatFailed;
      }
    } else {
      auto at = [](const ImageInfo &img, int x, int y) {
        return static_cast<char *>(img.data) + static_cast<ptrdiff_t>(y - img.bounds.y1) * img.rowBytes +
               static_cast<ptrdiff_t>(x - img.bounds.x1) * img.components * static_cast<ptrdiff_t>(sizeof(float));
      };
      ImageRef in{at(src, region.x1, region.y1), src.rowBytes, src.components};
      MutableImageRef out{at(dst, region.x1, region.y1), dst.rowBytes, dst.components};
      renderGrain(settings, in, out, width, height);
    }
  }

  gEffect->clipReleaseImage(src.handle);
  gEffect->clipReleaseImage(dst.handle);
  return status;
}

OfxStatus mainEntry(const char *action, const void *handle, OfxPropertySetHandle inArgs,
                    OfxPropertySetHandle outArgs) {
  OfxImageEffectHandle effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void *>(handle));
  try {
    if (std::strcmp(action, kOfxActionLoad) == 0) return onLoad();
    if (std::strcmp(action, kOfxActionUnload) == 0) return kOfxStatOK;
    if (std::strcmp(action, kOfxActionDescribe) == 0) return describe(effect);
    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0) return describeInContext(effect);
    if (std::strcmp(action, kOfxActionCreateInstance) == 0) return createInstance(effect);
    if (std::strcmp(action, kOfxActionDestroyInstance) == 0) return destroyInstance(effect);
    if (std::strcmp(action, kOfxActionInstanceChanged) == 0) return instanceChanged(effect, inArgs);
    if (std::strcmp(action, kOfxImageEffectActionGetClipPreferences) == 0) return getClipPreferences(effect, outArgs);
    if (std::strcmp(action, kOfxImageEffectActionIsIdentity) == 0) return isIdentity(effect, inArgs, outArgs);
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0) return render(effect, inArgs);
  } catch (const std::exception &e) {
    postError(effect, e.what());
    return kOfxStatFailed;
  } catch (...) {
    return kOfxStatFailed;
  }
  return kOfxStatReplyDefault;
}

void setHost(OfxHost *host) { gHost = host; }

OfxPlugin gPlugin = {kOfxImageEffectPluginApi, 1, PLUGIN_ID, PLUGIN_VERSION_MAJOR, PLUGIN_VERSION_MINOR,
                     setHost, mainEntry};

}  // namespace

extern "C" {
OfxExport int OfxGetNumberOfPlugins(void) { return 1; }
OfxExport OfxPlugin *OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
}
