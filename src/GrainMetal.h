// Metal backend: renders grain on the host's command queue with host-owned
// MTLBuffers (OFX Metal render, e.g. DaVinci Resolve on macOS).
#pragma once

#include <cstddef>
#include <string>

#include "GrainEngine.h"

namespace spektragrain {

struct MetalScratch;  // per-instance GPU scratch buffers
MetalScratch *createMetalScratch();
void destroyMetalScratch(MetalScratch *scratch);

struct MetalImageLayout {
  size_t offsetFloats = 0;  // to the first pixel of the render region
  size_t strideFloats = 0;  // per row
  int components = 4;
};

// commandQueue is an id<MTLCommandQueue>, src/dst are id<MTLBuffer> holding
// float pixels. Work is committed to the queue; set waitUntilDone to block.
bool renderGrainMetal(const GrainSettings &settings, MetalScratch *scratch, void *commandQueue, void *srcBuffer,
                      const MetalImageLayout &srcLayout, void *dstBuffer, const MetalImageLayout &dstLayout,
                      int width, int height, bool waitUntilDone, std::string *error);

}  // namespace spektragrain
