// Metal backend. Mirrors renderGrain() in GrainEngine.cpp stage for stage;
// the per-pixel math lives in GrainShared.h, compiled into both.
#include "GrainMetal.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include "GrainInternal.h"

namespace spektragrain {
namespace {

const char *const kMetalSource =
#include "MetalSource.inc"
    ;

struct Pipelines {
  id<MTLComputePipelineState> grain, blurH, blurV, mix, resampleH, resampleV, final;
};

bool buildPipelines(id<MTLDevice> device, Pipelines &out, std::string *error) {
  MTLCompileOptions *options = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  NSError *err = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                options:options
                                                  error:&err];
  if (!library) {
    if (error) *error = std::string("Metal compile failed: ") + (err ? err.localizedDescription.UTF8String : "?");
    return false;
  }
  auto make = [&](const char *name) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) return nil;
    NSError *perr = nil;
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:fn error:&perr];
    if (!state && error) *error = std::string("Metal pipeline failed: ") + name;
    return state;
  };
  out.grain = make("sg_grain");
  out.blurH = make("sg_blurH");
  out.blurV = make("sg_blurV");
  out.mix = make("sg_mix");
  out.resampleH = make("sg_resampleH");
  out.resampleV = make("sg_resampleV");
  out.final = make("sg_final");
  return out.grain && out.blurH && out.blurV && out.mix && out.resampleH && out.resampleV && out.final;
}

const Pipelines *pipelinesFor(id<MTLDevice> device, std::string *error) {
  static std::mutex mutex;
  static std::map<uint64_t, Pipelines> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const uint64_t key = device.registryID;
  auto it = cache.find(key);
  if (it != cache.end()) {
    return &it->second;
  }
  Pipelines p;
  if (!buildPipelines(device, p, error)) {
    return nullptr;
  }
  return &cache.emplace(key, p).first->second;
}

void dispatch2D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> state, NSUInteger w, NSUInteger h,
                NSUInteger d = 1) {
  const NSUInteger tw = state.threadExecutionWidth;
  const NSUInteger th = std::max<NSUInteger>(1, state.maxTotalThreadsPerThreadgroup / tw / 2);
  [enc setComputePipelineState:state];
  [enc dispatchThreads:MTLSizeMake(w, h, d) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

id<MTLBuffer> tapsBuffer(id<MTLDevice> device, const ResampleTaps &t, id<MTLBuffer> *weights) {
  std::vector<int> ints;
  ints.reserve(t.start.size() * 3);
  ints.insert(ints.end(), t.start.begin(), t.start.end());
  ints.insert(ints.end(), t.count.begin(), t.count.end());
  ints.insert(ints.end(), t.offset.begin(), t.offset.end());
  *weights = [device newBufferWithBytes:t.weight.data()
                                 length:std::max<size_t>(4, t.weight.size() * sizeof(float))
                                options:MTLResourceStorageModeShared];
  return [device newBufferWithBytes:ints.data() length:ints.size() * sizeof(int) options:MTLResourceStorageModeShared];
}

}  // namespace

struct MetalScratch {
  id<MTLDevice> device = nil;
  id<MTLBuffer> a = nil, b = nil;
  size_t capacityBytes = 0;
  id<MTLCommandQueue> lastQueue = nil;
  id<MTLCommandBuffer> lastCommands = nil;
};

MetalScratch *createMetalScratch() { return new MetalScratch(); }

void destroyMetalScratch(MetalScratch *scratch) {
  if (scratch && scratch->lastCommands) {
    [scratch->lastCommands waitUntilCompleted];
  }
  delete scratch;
}

bool renderGrainMetal(const GrainSettings &settings, MetalScratch *scratch, void *commandQueue, void *srcBuffer,
                      const MetalImageLayout &srcLayout, void *dstBuffer, const MetalImageLayout &dstLayout,
                      int width, int height, bool waitUntilDone, std::string *error) {
  @autoreleasepool {
    if (!scratch || !commandQueue || !srcBuffer || !dstBuffer || width <= 0 || height <= 0) {
      if (error) *error = "invalid Metal render arguments";
      return false;
    }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcBuffer;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstBuffer;
    id<MTLDevice> device = queue.device;

    const Pipelines *pl = pipelinesFor(device, error);
    if (!pl) {
      return false;
    }

    FramePlan plan = makeFramePlan(settings, width, height);
    SGParams &k = plan.k;
    k.srcOffset = static_cast<sg_u32>(srcLayout.offsetFloats);
    k.dstOffset = static_cast<sg_u32>(dstLayout.offsetFloats);
    k.srcStride = static_cast<sg_u32>(srcLayout.strideFloats);
    k.dstStride = static_cast<sg_u32>(dstLayout.strideFloats);
    k.srcComponents = static_cast<sg_u32>(srcLayout.components);
    k.dstComponents = static_cast<sg_u32>(dstLayout.components);
    const StockLuts &L = *plan.luts;
    const size_t gw = k.gw, gh = k.gh, W = k.width, H = k.height;

    // Scratch reuse: work on another queue may still be reading it.
    if (scratch->lastCommands && scratch->lastQueue != queue) {
      [scratch->lastCommands waitUntilCompleted];
    }
    const size_t need = 3 * std::max({gw * gh, W * gh, W * H}) * sizeof(float);
    if (scratch->device != device || scratch->capacityBytes < need) {
      if (scratch->lastCommands) {
        [scratch->lastCommands waitUntilCompleted];
      }
      scratch->device = device;
      scratch->a = [device newBufferWithLength:need options:MTLResourceStorageModePrivate];
      scratch->b = [device newBufferWithLength:need options:MTLResourceStorageModePrivate];
      scratch->capacityBytes = (scratch->a && scratch->b) ? need : 0;
      if (!scratch->capacityBytes) {
        if (error) *error = "out of GPU memory for grain scratch buffers";
        return false;
      }
    }
    id<MTLBuffer> A = scratch->a, B = scratch->b;

    id<MTLBuffer> layerLut = [device newBufferWithBytes:L.layer.data()
                                                 length:L.layer.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> slopeLut = [device newBufferWithBytes:L.slope.data()
                                                 length:L.slope.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];

    // Blur weights [3][taps], centred, zero padded.
    uint32_t maxRadius = std::max({k.blurRadius[0], k.blurRadius[1], k.blurRadius[2]});
    const uint32_t taps = 2 * maxRadius + 1;
    std::vector<float> weights(3 * taps, 0.0f);
    for (int c = 0; c < 3; ++c) {
      if (k.blurRadius[c] == 0) continue;
      const std::vector<float> kern = gaussianKernel(plan.blurSigma[c]);
      const int r = static_cast<int>(kern.size() / 2);
      for (int j = -r; j <= r; ++j) weights[c * taps + maxRadius + j] = kern[j + r];
    }
    id<MTLBuffer> blurWeights = [device newBufferWithBytes:weights.data()
                                                    length:weights.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commands = [queue commandBuffer];
    commands.label = @"SpektraGrain";
    id<MTLComputeCommandEncoder> enc = [commands computeCommandEncoder];
    [enc setBytes:&k length:sizeof(SGParams) atIndex:0];

    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBuffer:layerLut offset:0 atIndex:2];
    [enc setBuffer:A offset:0 atIndex:3];
    dispatch2D(enc, pl->grain, gw, gh);

    if (maxRadius > 0) {
      [enc setBuffer:A offset:0 atIndex:1];
      [enc setBuffer:B offset:0 atIndex:2];
      [enc setBuffer:blurWeights offset:0 atIndex:3];
      [enc setBytes:&taps length:sizeof(taps) atIndex:4];
      dispatch2D(enc, pl->blurH, gw, gh, 3);
      [enc setBuffer:B offset:0 atIndex:1];
      [enc setBuffer:A offset:0 atIndex:2];
      dispatch2D(enc, pl->blurV, gw, gh, 3);
    }
    if (std::fabs(k.satMix - 1.0f) > 1e-4f) {
      [enc setBuffer:A offset:0 atIndex:1];
      dispatch2D(enc, pl->mix, gw, gh);
    }
    if (!plan.identity) {
      id<MTLBuffer> wx = nil, wy = nil;
      id<MTLBuffer> tx = tapsBuffer(device, resampleTaps(static_cast<int>(gw), static_cast<int>(W)), &wx);
      id<MTLBuffer> ty = tapsBuffer(device, resampleTaps(static_cast<int>(gh), static_cast<int>(H)), &wy);
      [enc setBuffer:A offset:0 atIndex:1];
      [enc setBuffer:B offset:0 atIndex:2];
      [enc setBuffer:tx offset:0 atIndex:3];
      [enc setBuffer:wx offset:0 atIndex:4];
      dispatch2D(enc, pl->resampleH, W, gh, 3);
      [enc setBuffer:B offset:0 atIndex:1];
      [enc setBuffer:A offset:0 atIndex:2];
      [enc setBuffer:ty offset:0 atIndex:3];
      [enc setBuffer:wy offset:0 atIndex:4];
      dispatch2D(enc, pl->resampleV, W, H, 3);
    }

    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBuffer:dst offset:0 atIndex:2];
    [enc setBuffer:A offset:0 atIndex:3];
    [enc setBuffer:slopeLut offset:0 atIndex:4];
    dispatch2D(enc, pl->final, W, H);
    [enc endEncoding];
    [commands commit];

    scratch->lastQueue = queue;
    scratch->lastCommands = commands;
    if (waitUntilDone) {
      [commands waitUntilCompleted];
      if (commands.status == MTLCommandBufferStatusError) {
        if (error) *error = std::string("Metal command buffer failed: ") +
                            (commands.error ? commands.error.localizedDescription.UTF8String : "?");
        return false;
      }
    }
    return true;
  }
}

}  // namespace spektragrain
