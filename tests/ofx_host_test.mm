// Minimal OpenFX host: loads the built plugin binary and drives it through
// load / describe / instance / param changes / CPU render / Metal render the
// way DaVinci Resolve does.
#import <Metal/Metal.h>

#include <dlfcn.h>

#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ofxCore.h"
#include "ofxGPURender.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

namespace {

int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

// properties
struct PropValue {
  std::vector<std::string> s;
  std::vector<double> d;
  std::vector<int> i;
  std::vector<void *> p;
};
struct PropSet {
  std::map<std::string, PropValue> props;
};
PropSet *P(OfxPropertySetHandle h) { return reinterpret_cast<PropSet *>(h); }
OfxPropertySetHandle H(PropSet *p) { return reinterpret_cast<OfxPropertySetHandle>(p); }

template <class T>
void setAt(std::vector<T> &v, int index, const T &value) {
  if (static_cast<int>(v.size()) <= index) v.resize(index + 1);
  v[index] = value;
}

OfxStatus propSetPointer(OfxPropertySetHandle h, const char *n, int i, void *v) { setAt(P(h)->props[n].p, i, v); return kOfxStatOK; }
OfxStatus propSetString(OfxPropertySetHandle h, const char *n, int i, const char *v) { setAt(P(h)->props[n].s, i, std::string(v)); return kOfxStatOK; }
OfxStatus propSetDouble(OfxPropertySetHandle h, const char *n, int i, double v) { setAt(P(h)->props[n].d, i, v); return kOfxStatOK; }
OfxStatus propSetInt(OfxPropertySetHandle h, const char *n, int i, int v) { setAt(P(h)->props[n].i, i, v); return kOfxStatOK; }
OfxStatus propSetPointerN(OfxPropertySetHandle h, const char *n, int c, void *const *v) { for (int i = 0; i < c; ++i) propSetPointer(h, n, i, v[i]); return kOfxStatOK; }
OfxStatus propSetStringN(OfxPropertySetHandle h, const char *n, int c, const char *const *v) { for (int i = 0; i < c; ++i) propSetString(h, n, i, v[i]); return kOfxStatOK; }
OfxStatus propSetDoubleN(OfxPropertySetHandle h, const char *n, int c, const double *v) { for (int i = 0; i < c; ++i) propSetDouble(h, n, i, v[i]); return kOfxStatOK; }
OfxStatus propSetIntN(OfxPropertySetHandle h, const char *n, int c, const int *v) { for (int i = 0; i < c; ++i) propSetInt(h, n, i, v[i]); return kOfxStatOK; }

template <class T>
OfxStatus getAt(PropSet *ps, const char *n, int i, std::vector<T> PropValue::*member, T *out) {
  auto it = ps->props.find(n);
  if (it == ps->props.end()) return kOfxStatErrUnknown;
  const std::vector<T> &v = it->second.*member;
  if (i < 0 || i >= static_cast<int>(v.size())) return kOfxStatErrBadIndex;
  *out = v[i];
  return kOfxStatOK;
}
OfxStatus propGetPointer(OfxPropertySetHandle h, const char *n, int i, void **v) { return getAt(P(h), n, i, &PropValue::p, v); }
OfxStatus propGetString(OfxPropertySetHandle h, const char *n, int i, char **v) {
  auto it = P(h)->props.find(n);
  if (it == P(h)->props.end() || i >= static_cast<int>(it->second.s.size())) return kOfxStatErrUnknown;
  *v = const_cast<char *>(it->second.s[i].c_str());
  return kOfxStatOK;
}
OfxStatus propGetDouble(OfxPropertySetHandle h, const char *n, int i, double *v) { return getAt(P(h), n, i, &PropValue::d, v); }
OfxStatus propGetInt(OfxPropertySetHandle h, const char *n, int i, int *v) { return getAt(P(h), n, i, &PropValue::i, v); }
OfxStatus propGetPointerN(OfxPropertySetHandle h, const char *n, int c, void **v) { for (int i = 0; i < c; ++i) if (propGetPointer(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus propGetStringN(OfxPropertySetHandle h, const char *n, int c, char **v) { for (int i = 0; i < c; ++i) if (propGetString(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus propGetDoubleN(OfxPropertySetHandle h, const char *n, int c, double *v) { for (int i = 0; i < c; ++i) if (propGetDouble(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus propGetIntN(OfxPropertySetHandle h, const char *n, int c, int *v) { for (int i = 0; i < c; ++i) if (propGetInt(h, n, i, &v[i])) return kOfxStatErrUnknown; return kOfxStatOK; }
OfxStatus propReset(OfxPropertySetHandle h, const char *n) { P(h)->props.erase(n); return kOfxStatOK; }
OfxStatus propGetDimension(OfxPropertySetHandle h, const char *n, int *c) {
  auto it = P(h)->props.find(n);
  if (it == P(h)->props.end()) return kOfxStatErrUnknown;
  const PropValue &v = it->second;
  *c = static_cast<int>(std::max({v.s.size(), v.d.size(), v.i.size(), v.p.size()}));
  return kOfxStatOK;
}

OfxPropertySuiteV1 gPropSuite = {propSetPointer, propSetString, propSetDouble, propSetInt, propSetPointerN,
                                 propSetStringN, propSetDoubleN, propSetIntN, propGetPointer, propGetString,
                                 propGetDouble, propGetInt, propGetPointerN, propGetStringN, propGetDoubleN,
                                 propGetIntN, propReset, propGetDimension};

// params
struct Param {
  std::string type, name;
  PropSet props;
  bool hasValue = false;
  double d = 0.0;
  int i = 0;
  bool isDouble() const { return type == kOfxParamTypeDouble; }
  void ensure() {
    if (hasValue) return;
    hasValue = true;
    auto it = props.props.find(kOfxParamPropDefault);
    if (it == props.props.end()) return;
    if (isDouble() && !it->second.d.empty()) d = it->second.d[0];
    if (!isDouble() && !it->second.i.empty()) i = it->second.i[0];
  }
};
struct ParamSet {
  std::map<std::string, Param *> params;
  std::vector<std::string> order;
  PropSet props;
};

OfxStatus paramDefine(OfxParamSetHandle s, const char *type, const char *name, OfxPropertySetHandle *props) {
  ParamSet *set = reinterpret_cast<ParamSet *>(s);
  if (set->params.count(name)) return kOfxStatErrExists;
  Param *p = new Param();
  p->type = type;
  p->name = name;
  set->params[name] = p;
  set->order.push_back(name);
  if (props) *props = H(&p->props);
  return kOfxStatOK;
}
OfxStatus paramGetHandle(OfxParamSetHandle s, const char *name, OfxParamHandle *h, OfxPropertySetHandle *props) {
  ParamSet *set = reinterpret_cast<ParamSet *>(s);
  auto it = set->params.find(name);
  if (it == set->params.end()) return kOfxStatErrUnknown;
  *h = reinterpret_cast<OfxParamHandle>(it->second);
  if (props) *props = H(&it->second->props);
  return kOfxStatOK;
}
OfxStatus paramSetGetPropertySet(OfxParamSetHandle s, OfxPropertySetHandle *props) {
  *props = H(&reinterpret_cast<ParamSet *>(s)->props);
  return kOfxStatOK;
}
OfxStatus paramGetPropertySet(OfxParamHandle h, OfxPropertySetHandle *props) {
  *props = H(&reinterpret_cast<Param *>(h)->props);
  return kOfxStatOK;
}
OfxStatus readValue(Param *p, va_list ap) {
  p->ensure();
  if (p->isDouble()) *va_arg(ap, double *) = p->d;
  else *va_arg(ap, int *) = p->i;
  return kOfxStatOK;
}
OfxStatus paramGetValue(OfxParamHandle h, ...) {
  va_list ap;
  va_start(ap, h);
  OfxStatus st = readValue(reinterpret_cast<Param *>(h), ap);
  va_end(ap);
  return st;
}
OfxStatus paramGetValueAtTime(OfxParamHandle h, OfxTime time, ...) {
  va_list ap;
  va_start(ap, time);
  OfxStatus st = readValue(reinterpret_cast<Param *>(h), ap);
  va_end(ap);
  return st;
}
OfxStatus paramSetValue(OfxParamHandle h, ...) {
  Param *p = reinterpret_cast<Param *>(h);
  p->ensure();
  va_list ap;
  va_start(ap, h);
  if (p->isDouble()) p->d = va_arg(ap, double);
  else p->i = va_arg(ap, int);
  va_end(ap);
  return kOfxStatOK;
}
OfxStatus notImplemented() { return kOfxStatErrUnsupported; }

OfxParameterSuiteV1 gParamSuite = {
    paramDefine, paramGetHandle, paramSetGetPropertySet, paramGetPropertySet, paramGetValue, paramGetValueAtTime,
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, OfxTime, ...)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, OfxTime, OfxTime, ...)>(notImplemented), paramSetValue,
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, OfxTime, ...)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, unsigned int *)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, unsigned int, OfxTime *)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, OfxTime, int, int *)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, OfxTime)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamHandle)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamHandle, OfxParamHandle, OfxTime, const OfxRangeD *)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamSetHandle, const char *)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxParamSetHandle)>(notImplemented)};

// effect
struct Clip {
  PropSet props;
  void *data = nullptr;  // float RGBA rows or id<MTLBuffer>
  int width = 0, height = 0, components = 4;
};
struct Effect {
  PropSet props;
  ParamSet params;
  std::map<std::string, Clip> clips;
  std::vector<PropSet *> liveImages;
};

OfxStatus getPropertySet(OfxImageEffectHandle e, OfxPropertySetHandle *p) { *p = H(&reinterpret_cast<Effect *>(e)->props); return kOfxStatOK; }
OfxStatus getParamSet(OfxImageEffectHandle e, OfxParamSetHandle *p) { *p = reinterpret_cast<OfxParamSetHandle>(&reinterpret_cast<Effect *>(e)->params); return kOfxStatOK; }
OfxStatus clipDefine(OfxImageEffectHandle e, const char *name, OfxPropertySetHandle *p) {
  *p = H(&reinterpret_cast<Effect *>(e)->clips[name].props);
  return kOfxStatOK;
}
OfxStatus clipGetHandle(OfxImageEffectHandle e, const char *name, OfxImageClipHandle *c, OfxPropertySetHandle *p) {
  Effect *fx = reinterpret_cast<Effect *>(e);
  auto it = fx->clips.find(name);
  if (it == fx->clips.end()) return kOfxStatErrUnknown;
  *c = reinterpret_cast<OfxImageClipHandle>(&it->second);
  if (p) *p = H(&it->second.props);
  return kOfxStatOK;
}
OfxStatus clipGetPropertySet(OfxImageClipHandle c, OfxPropertySetHandle *p) { *p = H(&reinterpret_cast<Clip *>(c)->props); return kOfxStatOK; }
int gLiveImages = 0;
OfxStatus clipGetImage(OfxImageClipHandle c, OfxTime, const OfxRectD *, OfxPropertySetHandle *img) {
  Clip *clip = reinterpret_cast<Clip *>(c);
  if (!clip->data) return kOfxStatFailed;
  PropSet *ps = new PropSet();
  propSetPointer(H(ps), kOfxImagePropData, 0, clip->data);
  const int bounds[4] = {0, 0, clip->width, clip->height};
  propSetIntN(H(ps), kOfxImagePropBounds, 4, bounds);
  propSetInt(H(ps), kOfxImagePropRowBytes, 0, clip->width * clip->components * 4);
  propSetString(H(ps), kOfxImageEffectPropPixelDepth, 0, kOfxBitDepthFloat);
  propSetString(H(ps), kOfxImageEffectPropComponents, 0,
                clip->components == 4 ? kOfxImageComponentRGBA : kOfxImageComponentRGB);
  *img = H(ps);
  ++gLiveImages;
  return kOfxStatOK;
}
OfxStatus clipReleaseImage(OfxPropertySetHandle img) {
  delete P(img);
  --gLiveImages;
  return kOfxStatOK;
}
int abortFn(OfxImageEffectHandle) { return 0; }

OfxImageEffectSuiteV1 gEffectSuite = {
    getPropertySet, getParamSet, clipDefine, clipGetHandle, clipGetPropertySet, clipGetImage, clipReleaseImage,
    reinterpret_cast<OfxStatus (*)(OfxImageClipHandle, OfxTime, OfxRectD *)>(notImplemented), abortFn,
    reinterpret_cast<OfxStatus (*)(OfxImageEffectHandle, size_t, OfxImageMemoryHandle *)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxImageMemoryHandle)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxImageMemoryHandle, void **)>(notImplemented),
    reinterpret_cast<OfxStatus (*)(OfxImageMemoryHandle)>(notImplemented)};

int gMessages = 0;
OfxStatus message(void *, const char *type, const char *, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  std::printf("  [plugin message %s] ", type);
  std::vprintf(format, ap);
  std::printf("\n");
  va_end(ap);
  ++gMessages;
  return kOfxStatOK;
}
OfxMessageSuiteV1 gMessageSuite = {message};

PropSet gHostProps;
const void *fetchSuite(OfxPropertySetHandle, const char *name, int version) {
  if (version != 1) return nullptr;
  if (!std::strcmp(name, kOfxPropertySuite)) return &gPropSuite;
  if (!std::strcmp(name, kOfxParameterSuite)) return &gParamSuite;
  if (!std::strcmp(name, kOfxImageEffectSuite)) return &gEffectSuite;
  if (!std::strcmp(name, kOfxMessageSuite)) return &gMessageSuite;
  return nullptr;
}
OfxHost gHost = {H(&gHostProps), fetchSuite};

Param *param(Effect &fx, const char *name) {
  auto it = fx.params.params.find(name);
  return it == fx.params.params.end() ? nullptr : it->second;
}

std::vector<float> makeImage(int W, int H, int comps) {
  std::vector<float> img(static_cast<size_t>(W) * H * comps);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      float *p = &img[(static_cast<size_t>(y) * W + x) * comps];
      p[0] = 0.1f + 0.8f * x / W;
      p[1] = 0.1f + 0.8f * y / H;
      p[2] = 0.45f;
      if (comps == 4) p[3] = 0.5f;
    }
  return img;
}

OfxStatus renderAt(OfxPlugin *plugin, Effect &fx, double time, int W, int Ht, bool metal, void *queue) {
  PropSet args;
  propSetDouble(H(&args), kOfxPropTime, 0, time);
  const int window[4] = {0, 0, W, Ht};
  propSetIntN(H(&args), kOfxImageEffectPropRenderWindow, 4, window);
  const double scale[2] = {1.0, 1.0};
  propSetDoubleN(H(&args), kOfxImageEffectPropRenderScale, 2, scale);
  propSetString(H(&args), kOfxImageEffectPropFieldToRender, 0, kOfxImageFieldNone);
  if (metal) {
    propSetInt(H(&args), kOfxImageEffectPropMetalEnabled, 0, 1);
    propSetPointer(H(&args), kOfxImageEffectPropMetalCommandQueue, 0, queue);
  }
  return plugin->mainEntry(kOfxImageEffectActionRender, &fx, H(&args), nullptr);
}

}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::printf("usage: ofx_host_test <plugin binary>\n");
      return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      std::printf("FAIL: dlopen: %s\n", dlerror());
      return 1;
    }
    auto getCount = reinterpret_cast<int (*)()>(dlsym(lib, "OfxGetNumberOfPlugins"));
    auto getPlugin = reinterpret_cast<OfxPlugin *(*)(int)>(dlsym(lib, "OfxGetPlugin"));
    CHECK(getCount && getPlugin, "entry points exported");
    if (!getCount || !getPlugin) return 1;
    CHECK(getCount() == 1, "one plugin");
    OfxPlugin *plugin = getPlugin(0);
    std::printf("plugin %s v%u.%u api %s\n", plugin->pluginIdentifier, plugin->pluginVersionMajor,
                plugin->pluginVersionMinor, plugin->pluginApi);
    CHECK(!std::strcmp(plugin->pluginApi, kOfxImageEffectPluginApi), "image effect API");

    plugin->setHost(&gHost);
    CHECK(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK, "load");

    Effect fx;
    CHECK(plugin->mainEntry(kOfxActionDescribe, &fx, nullptr, nullptr) == kOfxStatOK, "describe");
    char *label = nullptr, *metal = nullptr, *safety = nullptr;
    propGetString(H(&fx.props), kOfxPropLabel, 0, &label);
    propGetString(H(&fx.props), kOfxImageEffectPropMetalRenderSupported, 0, &metal);
    propGetString(H(&fx.props), kOfxImageEffectPluginRenderThreadSafety, 0, &safety);
    std::printf("label '%s', metal '%s', thread safety '%s'\n", label, metal, safety);
    CHECK(metal && !std::strcmp(metal, "true"), "metal advertised");

    PropSet ctx;
    propSetString(H(&ctx), kOfxImageEffectPropContext, 0, kOfxImageEffectContextFilter);
    CHECK(plugin->mainEntry(kOfxImageEffectActionDescribeInContext, &fx, H(&ctx), nullptr) == kOfxStatOK,
          "describe in context");
    std::printf("params:\n");
    for (const std::string &name : fx.params.order) {
      Param *p = fx.params.params[name];
      char *plabel = nullptr;
      propGetString(H(&p->props), kOfxPropLabel, 0, &plabel);
      int options = 0;
      propGetDimension(H(&p->props), kOfxParamPropChoiceOption, &options);
      std::printf("  %-16s %-34s %s%s\n", name.c_str(), plabel, p->type.c_str() + 12,
                  options ? (" (" + std::to_string(options) + " options)").c_str() : "");
    }
    Param *aspect = param(fx, "aspectRatio");
    CHECK(aspect, "aspect param exists");
    if (aspect) {
      double def = 0, mn = 0, mx = 0;
      propGetDouble(H(&aspect->props), kOfxParamPropDefault, 0, &def);
      propGetDouble(H(&aspect->props), kOfxParamPropMin, 0, &mn);
      propGetDouble(H(&aspect->props), kOfxParamPropMax, 0, &mx);
      CHECK(def == 1.0 && mn == 0.0 && mx == 2.0, "aspect range 0..2 default 1 (got %g %g..%g)", def, mn, mx);
    }
    CHECK(fx.clips.count(kOfxImageEffectSimpleSourceClipName) && fx.clips.count(kOfxImageEffectOutputClipName),
          "clips defined");

    CHECK(plugin->mainEntry(kOfxActionCreateInstance, &fx, nullptr, nullptr) == kOfxStatOK, "create instance");
    Param *rms = param(fx, "rmsGranularity");
    Param *stock = param(fx, "stock");
    rms->ensure();
    int enabled = -1;
    propGetInt(H(&rms->props), kOfxParamPropEnabled, 0, &enabled);
    std::printf("rms %.2f enabled %d\n", rms->d, enabled);
    CHECK(std::fabs(rms->d - 5.0) < 1e-6 && enabled == 0, "rms mirrors default stock and is locked");

    // Switch stock -> RMS follows.
    stock->ensure();
    stock->i = 3;  // Vision3 500T
    PropSet changed;
    propSetString(H(&changed), kOfxPropType, 0, kOfxTypeParameter);
    propSetString(H(&changed), kOfxPropName, 0, "stock");
    propSetString(H(&changed), kOfxPropChangeReason, 0, kOfxChangeUserEdited);
    propSetDouble(H(&changed), kOfxPropTime, 0, 0.0);
    plugin->mainEntry(kOfxActionInstanceChanged, &fx, H(&changed), nullptr);
    std::printf("after stock change: rms %.2f\n", rms->d);
    CHECK(std::fabs(rms->d - 6.5) < 1e-6, "rms follows stock");

    PropSet prefs;
    CHECK(plugin->mainEntry(kOfxImageEffectActionGetClipPreferences, &fx, nullptr, H(&prefs)) == kOfxStatOK,
          "clip prefs");
    int varying = -1;
    propGetInt(H(&prefs), kOfxImageEffectFrameVarying, 0, &varying);
    CHECK(varying == 1, "frame varying while animated");

    // CPU render.
    const int W = 1920, Hh = 1080;
    std::vector<float> in = makeImage(W, Hh, 4), out(in.size(), -1.0f), out2(in.size()), out3(in.size());
    Clip &srcClip = fx.clips[kOfxImageEffectSimpleSourceClipName];
    Clip &dstClip = fx.clips[kOfxImageEffectOutputClipName];
    srcClip.data = in.data();
    srcClip.width = dstClip.width = W;
    srcClip.height = dstClip.height = Hh;
    dstClip.data = out.data();
    CHECK(renderAt(plugin, fx, 12.0, W, Hh, false, nullptr) == kOfxStatOK, "CPU render");
    dstClip.data = out2.data();
    renderAt(plugin, fx, 12.0, W, Hh, false, nullptr);
    dstClip.data = out3.data();
    renderAt(plugin, fx, 13.0, W, Hh, false, nullptr);
    double diff = 0.0, mean = 0.0, sameFrame = 0.0, nextFrame = 0.0;
    bool alphaOk = true, finite = true;
    for (size_t i = 0; i < in.size(); ++i) {
      if (i % 4 == 3) {
        alphaOk = alphaOk && out[i] == in[i];
        continue;
      }
      finite = finite && std::isfinite(out[i]);
      diff += (out[i] - in[i]) * (out[i] - in[i]);
      mean += out[i] - in[i];
      sameFrame = std::max(sameFrame, static_cast<double>(std::fabs(out[i] - out2[i])));
      nextFrame += std::fabs(out[i] - out3[i]);
    }
    const double n = in.size() * 0.75;
    std::printf("CPU: grain rms %.4f, mean shift %+.5f, repeat diff %.1e, next-frame diff %.4f\n",
                std::sqrt(diff / n), mean / n, sameFrame, nextFrame / n);
    CHECK(finite, "finite output");
    CHECK(alphaOk, "alpha preserved");
    CHECK(std::sqrt(diff / n) > 1e-3, "grain visible");
    CHECK(std::fabs(mean / n) < 2e-3, "grain mean-neutral");
    CHECK(sameFrame == 0.0, "deterministic per frame");
    CHECK(nextFrame / n > 1e-3, "animates over time");

    // Metal render through the plugin; must match the CPU result.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> srcBuf = [device newBufferWithBytes:in.data() length:in.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [device newBufferWithLength:in.size() * 4 options:MTLResourceStorageModeShared];
    srcClip.data = (__bridge void *)srcBuf;
    dstClip.data = (__bridge void *)dstBuf;
    CHECK(renderAt(plugin, fx, 12.0, W, Hh, true, (__bridge void *)queue) == kOfxStatOK, "Metal render");
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];
    const float *gpu = static_cast<const float *>(dstBuf.contents);
    double maxDiff = 0.0;
    for (size_t i = 0; i < in.size(); ++i) maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(gpu[i] - out[i])));
    std::printf("Metal vs CPU through plugin: max diff %.2e\n", maxDiff);
    CHECK(maxDiff < 1e-4, "Metal matches CPU");

    // Stretched grain + grain-only view on Metal, then RGB on CPU.
    param(fx, "aspectRatio")->d = 1.8;
    param(fx, "view")->ensure();
    param(fx, "view")->i = 1;
    CHECK(renderAt(plugin, fx, 20.0, W, Hh, true, (__bridge void *)queue) == kOfxStatOK, "Metal stretched render");
    param(fx, "view")->i = 0;
    std::vector<float> rgbIn = makeImage(W, Hh, 3), rgbOut(rgbIn.size());
    srcClip.data = rgbIn.data();
    dstClip.data = rgbOut.data();
    srcClip.components = dstClip.components = 3;
    CHECK(renderAt(plugin, fx, 20.0, W, Hh, false, nullptr) == kOfxStatOK, "CPU RGB render");
    double rgbDiff = 0.0;
    for (size_t i = 0; i < rgbIn.size(); ++i) rgbDiff += std::fabs(rgbOut[i] - rgbIn[i]);
    CHECK(rgbDiff / rgbIn.size() > 1e-4, "RGB render has grain");
    srcClip.components = dstClip.components = 4;

    // Amount 0 -> identity.
    param(fx, "amount")->ensure();
    param(fx, "amount")->d = 0.0;
    PropSet idIn, idOut;
    propSetDouble(H(&idIn), kOfxPropTime, 0, 1.0);
    CHECK(plugin->mainEntry(kOfxImageEffectActionIsIdentity, &fx, H(&idIn), H(&idOut)) == kOfxStatOK,
          "identity at amount 0");

    CHECK(gLiveImages == 0, "all images released (%d live)", gLiveImages);
    CHECK(gMessages == 0, "no error messages");
    CHECK(plugin->mainEntry(kOfxActionDestroyInstance, &fx, nullptr, nullptr) == kOfxStatOK, "destroy");
    CHECK(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatOK, "unload");

    std::printf(failures ? "%d FAILURES\n" : "OFX host test passed\n", failures);
    return failures ? 1 : 0;
  }
}
