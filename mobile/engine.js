// Spektra Mobile engine: a JavaScript port of the spektrafilm pipeline
// (https://github.com/andreavolpato/spektrafilm, GPL-3.0) for phones.
//
//   camera RGB -> film exposure (spectral upsampling, baked per film)
//   -> halation and in-emulsion scatter
//   -> develop: characteristic curves, DIR couplers, sublayer grain
//   -> enlarger: film dyes x lamp x dichroic filters x paper sensitivity
//   -> print develop -> scan (viewing illuminant, CIE observer, gamut map)
//   -> glare, scanner unsharp mask, sRGB
//
// Slide films skip the print and are scanned directly. The spectral steps
// that do not depend on the image are baked by mobile/bake.py from upstream;
// everything spatial or random runs here, tile by tile.
(function (root) {
  "use strict";

  // Upstream defaults (spektrafilm/runtime/params_schema.py).
  const GRAIN = {
    particleAreaUm2: 0.2,
    particleScale: [1.6, 1.6, 3.2],
    particleScaleLayers: [2.0, 1.0, 0.5],
    densityMin: [0.03, 0.03, 0.03],
    uniformity: [0.97, 0.99, 0.97],
    blur: 0.65,
    blurDyeCloudsUm: 1.0,
  };
  const COUPLER_DIFFUSION_UM = 20.0;
  const COUPLER_TAIL_UM = 200.0;
  const COUPLER_TAIL_WEIGHT = 0.06;
  const GLARE = { percent: 0.03, roughness: 0.7, blur: 0.5 };
  const UNSHARP = { sigma: 0.7, amount: 0.7 };
  const EXP_FIT = [[0.1633, 0.536], [0.6496, 1.5236], [0.187, 2.7684]]; // exponential PSF as 3 Gaussians
  const ENLARGER_N = 25;
  const MIDGRAY = 0.184;
  const TILE_ROWS = 160;

  // ---------------------------------------------------------------- utils

  function b64ToU16(b64) {
    let bin;
    if (typeof atob === "function") bin = atob(b64);
    else bin = Buffer.from(b64, "base64").toString("binary");
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Uint16Array(bytes.buffer);
  }

  // Seedable PRNG (sfc32) + standard normal.
  function makeRng(seed) {
    let a = 0x9e3779b9 ^ seed, b = 0x243f6a88 + seed * 31, c = 0xb7e15162 ^ (seed * 7919), d = 1;
    function next() {
      a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
      let t = (a + b) | 0;
      a = b ^ (b >>> 9);
      b = (c + (c << 3)) | 0;
      c = (c << 21) | (c >>> 11);
      d = (d + 1) | 0;
      t = (t + d) | 0;
      c = (c + t) | 0;
      return (t >>> 0) / 4294967296;
    }
    for (let i = 0; i < 12; i++) next();
    // Two table-driven standard normals per 32-bit draw.
    let spare = -1, held = 0;
    function normal() {
      if (spare >= 0) { const v = NORMAL_TABLE[spare]; spare = -1; return v; }
      a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
      let t = (a + b) | 0;
      a = b ^ (b >>> 9); b = (c + (c << 3)) | 0; c = (c << 21) | (c >>> 11); d = (d + 1) | 0;
      t = (t + d) | 0; c = (c + t) | 0;
      held = t >>> 0;
      spare = held & 0xffff;
      return NORMAL_TABLE[held >>> 16];
    }
    // Poisson: exact CDF inversion for small means; above that a
    // Cornish-Fisher corrected Gaussian that keeps Poisson's skew.
    function poisson(lam) {
      if (!(lam > 0)) return 0;
      if (lam >= POISSON_EXACT_BELOW) {
        const z = normal();
        const k = Math.round(lam + Math.sqrt(lam) * z + (z * z - 1) / 6);
        return k > 0 ? k : 0;
      }
      const u = next();
      let pk = Math.exp(-lam), cdf = pk, k = 0;
      while (u > cdf && k < 64) { k++; pk *= lam / k; cdf += pk; }
      return k;
    }
    return { next, normal, poisson };
  }
  const POISSON_EXACT_BELOW = 12;

  // Inverse normal CDF at 65536 mid-quantiles (Acklam's approximation).
  const NORMAL_TABLE = (function () {
    const A = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const B = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const C = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    function inv(p) {
      if (p < 0.02425) {
        const q = Math.sqrt(-2 * Math.log(p));
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
      }
      if (p > 1 - 0.02425) return -inv(1 - p);
      const q = p - 0.5, r = q * q;
      return ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q) / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1);
    }
    const t = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) t[i] = inv((i + 0.5) / 65536);
    return t;
  })();

  // np.interp on a uniform grid.
  function interpU(x, x0, step, ys) {
    const n = ys.length;
    let t = (x - x0) / step;
    if (!(t > 0)) return ys[0];
    if (t >= n - 1) return ys[n - 1];
    const i = t | 0;
    const f = t - i;
    return ys[i] + (ys[i + 1] - ys[i]) * f;
  }

  // np.interp on a sorted (non-uniform) axis, right-biased like fast_interp.
  function interpXY(x, xs, ys) {
    const n = xs.length;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n; // first index with xs[i] > x
    while (lo < hi) { const m = (lo + hi) >> 1; if (xs[m] > x) hi = m; else lo = m + 1; }
    const i = lo - 1;
    const dx = xs[i + 1] - xs[i];
    return dx !== 0 ? ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / dx : ys[i];
  }

  // A dense uniform table of interpXY, for the hot loops.
  function denseTable(xs, ys, size) {
    let lo = Infinity, hi = -Infinity;
    for (const v of xs) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const step = (hi - lo) / (size - 1);
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) out[i] = interpXY(lo + i * step, xs, ys);
    return { x0: lo, step, ys: out };
  }

  function srgbDecode(v) { return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }
  function srgbEncode(v) { return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055; }
  const DECODE8 = new Float32Array(256);
  for (let i = 0; i < 256; i++) DECODE8[i] = srgbDecode(i / 255);
  const ENCODE_N = 4096;
  const ENCODE_LUT = new Uint8Array(ENCODE_N + 1);
  for (let i = 0; i <= ENCODE_N; i++) ENCODE_LUT[i] = Math.round(255 * srgbEncode(i / ENCODE_N));
  function encode8(v) {
    if (!(v > 0)) return 0;
    if (v >= 1) return 255;
    // finer near black where the curve is steep
    if (v < 0.01) return Math.round(255 * srgbEncode(v));
    return ENCODE_LUT[Math.round(v * ENCODE_N)];
  }

  // ---------------------------------------------------------------- blur

  function gaussKernel(sigma, truncate) {
    const r = Math.floor(truncate * sigma + 0.5);
    const k = new Float32Array(2 * r + 1);
    let s = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) * (i / sigma)); k[i + r] = v; s += v; }
    for (let i = 0; i < k.length; i++) k[i] /= s;
    return k;
  }

  function reflect(i, n) {
    if (i >= 0 && i < n) return i;
    const period = 2 * n;
    i %= period; if (i < 0) i += period;
    return i >= n ? period - 1 - i : i;
  }

  // Separable FIR, scipy-style reflect borders.
  function firBlur(src, w, h, sigma, dst, tmp) {
    const k = gaussKernel(sigma, 3);
    const r = (k.length - 1) >> 1;
    if (r === 0) { if (dst !== src) dst.set(src); return dst; }
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        let s = 0;
        if (x >= r && x < w - r) for (let j = -r; j <= r; j++) s += k[j + r] * src[row + x + j];
        else for (let j = -r; j <= r; j++) s += k[j + r] * src[row + reflect(x + j, w)];
        tmp[row + x] = s;
      }
    }
    dst.fill(0);
    for (let j = -r; j <= r; j++) {
      const kj = k[j + r];
      for (let y = 0; y < h; y++) {
        const yy = reflect(y + j, h) * w, row = y * w;
        for (let x = 0; x < w; x++) dst[row + x] += kj * tmp[yy + x];
      }
    }
    return dst;
  }

  // Three box passes approximate a Gaussian for large sigma, O(1) per pixel.
  function boxSizes(sigma, n) {
    const wIdeal = Math.sqrt((12 * sigma * sigma) / n + 1);
    let wl = Math.floor(wIdeal); if (wl % 2 === 0) wl--;
    const wu = wl + 2;
    const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
    const sizes = [];
    for (let i = 0; i < n; i++) sizes.push(i < m ? wl : wu);
    return sizes;
  }
  function boxH(src, dst, w, h, r) {
    const inv = 1 / (2 * r + 1);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let acc = 0;
      for (let j = -r; j <= r; j++) acc += src[row + reflect(j, w)];
      for (let x = 0; x < w; x++) {
        dst[row + x] = acc * inv;
        acc += src[row + reflect(x + r + 1, w)] - src[row + reflect(x - r, w)];
      }
    }
  }
  function boxV(src, dst, w, h, r) {
    const inv = 1 / (2 * r + 1);
    const acc = new Float64Array(w);
    for (let j = -r; j <= r; j++) { const yy = reflect(j, h) * w; for (let x = 0; x < w; x++) acc[x] += src[yy + x]; }
    for (let y = 0; y < h; y++) {
      const row = y * w, add = reflect(y + r + 1, h) * w, sub = reflect(y - r, h) * w;
      for (let x = 0; x < w; x++) { dst[row + x] = acc[x] * inv; acc[x] += src[add + x] - src[sub + x]; }
    }
  }
  function boxBlur(src, w, h, sigma, dst, tmp) {
    const sizes = boxSizes(sigma, 3);
    let a = src;
    for (let i = 0; i < 3; i++) {
      const r = (sizes[i] - 1) >> 1;
      boxH(a, tmp, w, h, r);
      boxV(tmp, dst, w, h, r);
      a = dst;
    }
    return dst;
  }

  function blur(src, w, h, sigma, dst, tmp) {
    if (!(sigma > 0)) { if (dst !== src) dst.set(src); return dst; }
    return sigma < 3 ? firBlur(src, w, h, sigma, dst, tmp) : boxBlur(src, w, h, sigma, dst, tmp);
  }
  function blurReach(sigma) { return sigma > 0 ? Math.ceil(3 * sigma) + 2 : 0; }

  // ---------------------------------------------------------------- stocks

  function prepareFilm(f) {
    const n = f.curves[0].length;
    const film = {
      raw: f, n, positive: f.positive,
      le0: f.logE0, step: f.logEStep,
      curves: f.curves.map((c) => Float32Array.from(c)),
      curves0: f.curves0.map((c) => Float32Array.from(c)),
      rawCurves: f.rawCurves.map((c) => Float32Array.from(c)),
      couplers: f.couplers,
      chroma: Float32Array.from(f.chroma),
      dmax: [0, 0, 0],
      layerTables: [],
      layerMax: [],
    };
    for (let c = 0; c < 3; c++) {
      let m = -Infinity;
      for (const v of film.curves[c]) if (v > m) m = v;
      film.dmax[c] = m;
      // density -> sublayer density, as interp_density_cmy_layers
      const xs = film.positive ? Array.from(film.curves[c], (v) => -v) : Array.from(film.curves[c]);
      const tables = [], maxes = [];
      for (let l = 0; l < 3; l++) {
        const ys = f.layers[c][l];
        tables.push(denseTable(xs, ys, 2048));
        let mx = 0; for (const v of ys) if (v > mx) mx = v;
        maxes.push(mx);
      }
      film.layerTables.push(tables);
      film.layerMax.push(maxes);
    }
    if (f.scan) film.scan = prepareScan(f.scan);
    return film;
  }

  function prepareScan(s) {
    const q = b64ToU16(s.data);
    const out = new Float32Array(q.length);
    const lo = DATA.logLo, span = DATA.logHi - DATA.logLo;
    for (let i = 0; i < q.length; i++) out[i] = Math.pow(10, lo + (q[i] / 65535) * span);
    return { lo: s.lo, hi: s.hi, n: DATA.scanN, table: out };
  }

  function preparePrint(p) {
    return {
      raw: p, le0: p.logE0, step: p.logEStep,
      curves: p.curves.map((c) => Float32Array.from(c)),
      sens: p.sensitivity,
      scan: prepareScan(p.scan),
    };
  }

  let DATA = null;
  const filmCache = new Map(), printCache = new Map();
  function init(data) { DATA = data; filmCache.clear(); printCache.clear(); }
  function getFilm(id) {
    if (!filmCache.has(id)) filmCache.set(id, prepareFilm(DATA.films.find((f) => f.id === id) || DATA.films[0]));
    return filmCache.get(id);
  }
  function getPrint(id) {
    if (!printCache.has(id)) printCache.set(id, preparePrint(DATA.prints.find((p) => p.id === id) || DATA.prints[0]));
    return printCache.get(id);
  }

  // Trilinear lookup into an n^3 x 3 table over [lo, hi] per axis.
  function lut3(t, a, b, c, out) {
    const n = t.n, n1 = n - 1;
    let fa = ((a - t.lo[0]) / (t.hi[0] - t.lo[0])) * n1;
    let fb = ((b - t.lo[1]) / (t.hi[1] - t.lo[1])) * n1;
    let fc = ((c - t.lo[2]) / (t.hi[2] - t.lo[2])) * n1;
    fa = fa < 0 ? 0 : fa > n1 ? n1 : fa;
    fb = fb < 0 ? 0 : fb > n1 ? n1 : fb;
    fc = fc < 0 ? 0 : fc > n1 ? n1 : fc;
    let ia = fa | 0, ib = fb | 0, ic = fc | 0;
    if (ia > n - 2) ia = n - 2; if (ib > n - 2) ib = n - 2; if (ic > n - 2) ic = n - 2;
    const da = fa - ia, db = fb - ib, dc = fc - ic;
    const T = t.table, sA = n * n * 3, sB = n * 3;
    const base = ia * sA + ib * sB + ic * 3;
    for (let k = 0; k < 3; k++) {
      const p = base + k;
      const c00 = T[p] + (T[p + sA] - T[p]) * da;
      const c10 = T[p + sB] + (T[p + sA + sB] - T[p + sB]) * da;
      const c01 = T[p + 3] + (T[p + sA + 3] - T[p + 3]) * da;
      const c11 = T[p + sB + 3] + (T[p + sA + sB + 3] - T[p + sB + 3]) * da;
      const c0 = c00 + (c10 - c00) * db, c1 = c01 + (c11 - c01) * db;
      out[k] = c0 + (c1 - c0) * dc;
    }
    return out;
  }

  // camera linear sRGB -> film raw exposure (1 = mid gray), bilinear on chromaticity.
  function filmRaw(film, r, g, b, out) {
    const s = r + g + b;
    if (!(s > 1e-12)) { out[0] = out[1] = out[2] = 0; return out; }
    const n = DATA.chromaN, n1 = n - 1, T = film.chroma;
    let fu = (r / s) * n1, fv = (g / s) * n1;
    fu = fu < 0 ? 0 : fu > n1 ? n1 : fu;
    fv = fv < 0 ? 0 : fv > n1 ? n1 : fv;
    let iu = fu | 0, iv = fv | 0;
    if (iu > n - 2) iu = n - 2; if (iv > n - 2) iv = n - 2;
    const du = fu - iu, dv = fv - iv;
    const p00 = (iv * n + iu) * 3, p10 = p00 + 3, p01 = p00 + n * 3, p11 = p01 + 3;
    for (let k = 0; k < 3; k++) {
      const a = T[p00 + k] + (T[p10 + k] - T[p00 + k]) * du;
      const c = T[p01 + k] + (T[p11 + k] - T[p01 + k]) * du;
      out[k] = (a + (c - a) * dv) * s;
    }
    return out;
  }

  // ---------------------------------------------------------------- enlarger

  // Film CMY density -> print log exposure, tabulated for this film, paper,
  // filtration and negative exposure (printing.py, _film_cmy_to_print_log_raw).
  function buildEnlarger(film, paper, s) {
    const f = film.raw;
    const nf = ((DATA.neutralFilters[paper.raw.id] || {})["TH-KG3"] || {})[f.id] || [0, 65, 55];
    const cc = [nf[0], nf[1] + s.magentaShift, nf[2] + s.yellowShift];
    const L = DATA.enlargerLamp, F = DATA.dichroics, W = L.length;
    const illum = new Float64Array(W);
    for (let i = 0; i < W; i++) {
      let t = 1;
      for (let k = 0; k < 3; k++) t *= 1 - (1 - F[k][i]) * (1 - Math.pow(10, -cc[k] / 100));
      illum[i] = L[i] * t;
    }
    const cd = f.channelDensity, base = f.baseDensity, valid = f.spectralValid, sens = paper.sens;
    function rawPrint(c, m, y, out) {
      let r0 = 0, r1 = 0, r2 = 0;
      for (let i = 0; i < W; i++) {
        if (!valid[i]) continue;
        const light = Math.pow(10, -(c * cd[0][i] + m * cd[1][i] + y * cd[2][i] + base[i])) * illum[i];
        r0 += light * sens[0][i]; r1 += light * sens[1][i]; r2 += light * sens[2][i];
      }
      out[0] = r0; out[1] = r1; out[2] = r2;
      return out;
    }
    // Mid-gray normalisation with exposure compensation (geometric mean).
    const ev = s.printExposureCompensation ? s.exposureEv : 0;
    const raw = filmRaw(film, MIDGRAY * Math.pow(2, ev), MIDGRAY * Math.pow(2, ev), MIDGRAY * Math.pow(2, ev), [0, 0, 0]);
    const mid = [0, 1, 2].map((c) => interpU(Math.log10(raw[c] + 1e-10), film.le0, film.step, film.rawCurves[c]));
    const pm = rawPrint(mid[0], mid[1], mid[2], [0, 0, 0]);
    const logFactor = -(Math.log10(Math.max(pm[0], 1e-10)) + Math.log10(Math.max(pm[1], 1e-10)) + Math.log10(Math.max(pm[2], 1e-10))) / 3;
    const n = ENLARGER_N;
    const lo = [-0.15, -0.15, -0.15], hi = film.dmax.map((d) => d + 0.25);
    const table = new Float32Array(n * n * n * 3);
    const tmp = [0, 0, 0];
    const logPE = Math.log10(Math.max(s.printExposure, 1e-6));
    let p = 0;
    for (let a = 0; a < n; a++) {
      const c = lo[0] + ((hi[0] - lo[0]) * a) / (n - 1);
      for (let b = 0; b < n; b++) {
        const m = lo[1] + ((hi[1] - lo[1]) * b) / (n - 1);
        for (let d = 0; d < n; d++) {
          const y = lo[2] + ((hi[2] - lo[2]) * d) / (n - 1);
          rawPrint(c, m, y, tmp);
          for (let k = 0; k < 3; k++) table[p++] = Math.log10(Math.max(tmp[k], 0) * Math.pow(10, logFactor) + 1e-10) + logPE;
        }
      }
    }
    return { lo, hi, n, table };
  }

  let enlargerKey = "", enlargerTable = null;
  function cachedEnlarger(film, paper, s) {
    const key = [film.raw.id, paper.raw.id, s.magentaShift, s.yellowShift, s.printExposure,
                 s.printExposureCompensation ? s.exposureEv : "-"].join("|");
    if (key !== enlargerKey) { enlargerTable = buildEnlarger(film, paper, s); enlargerKey = key; }
    return enlargerTable;
  }

  // ---------------------------------------------------------------- pipeline

  function centerWeightedEv(lin, w, h) {
    // autoexposure.py, center_weighted: Gaussian (sigma 0.2 of the long edge) weighted mean Y
    const L = Math.max(w, h);
    let sum = 0, wsum = 0;
    for (let y = 0; y < h; y++) {
      const ny = (y / h - 0.5) * (h / L);
      for (let x = 0; x < w; x++) {
        const nx = (x / w - 0.5) * (w / L);
        const m = Math.exp(-(nx * nx + ny * ny) / (2 * 0.04));
        const i = y * w + x;
        sum += m * (0.2126 * lin[0][i] + 0.7152 * lin[1][i] + 0.0722 * lin[2][i]);
        wsum += m;
      }
    }
    const e = sum / wsum / MIDGRAY;
    return e > 0 ? -Math.log2(e) : 0;
  }

  function planes(n) { return [new Float32Array(n), new Float32Array(n), new Float32Array(n)]; }

  // Linear planar RGB for a block of the source, box-downsampled by `scale`.
  function readLinear(src, sw, x0, y0, w, h, scale) {
    const out = planes(w * h);
    if (scale === 1) {
      for (let y = 0; y < h; y++) {
        let p = ((y0 + y) * sw + x0) * 4, q = y * w;
        for (let x = 0; x < w; x++, p += 4, q++) {
          out[0][q] = DECODE8[src[p]]; out[1][q] = DECODE8[src[p + 1]]; out[2][q] = DECODE8[src[p + 2]];
        }
      }
      return out;
    }
    const sh = (src.length / 4 / sw) | 0;
    for (let y = 0; y < h; y++) {
      const ya = Math.floor(y * scale), yb = Math.min(sh, Math.max(ya + 1, Math.floor((y + 1) * scale)));
      for (let x = 0; x < w; x++) {
        const xa = Math.floor(x * scale), xb = Math.min(sw, Math.max(xa + 1, Math.floor((x + 1) * scale)));
        let r = 0, g = 0, b = 0, n = 0;
        for (let yy = ya; yy < yb; yy++) for (let xx = xa; xx < xb; xx++) {
          const p = (yy * sw + xx) * 4;
          r += DECODE8[src[p]]; g += DECODE8[src[p + 1]]; b += DECODE8[src[p + 2]]; n++;
        }
        const q = y * w + x;
        out[0][q] = r / n; out[1][q] = g / n; out[2][q] = b / n;
      }
    }
    return out;
  }

  function couplerSilver(film, density, c) {
    return film.positive ? film.dmax[c] - density : density;
  }

  // Exponential tail of the DIR-coupler diffusion, computed once on a small
  // copy of the frame (its kernel spans hundreds of microns).
  function couplerTailField(ctx, src, sw, sh) {
    const { film, s } = ctx;
    const scale = Math.max(1, Math.max(sw, sh) / 384);
    const w = Math.max(1, Math.round(sw / scale)), h = Math.max(1, Math.round(sh / scale));
    const lin = readLinear(src, sw, 0, 0, w, h, scale);
    const pixelUm = ctx.pixelUm * scale;
    const n = w * h, M = film.couplers;
    const silver = planes(n), raw = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      filmRaw(film, lin[0][i] * ctx.gain, lin[1][i] * ctx.gain, lin[2][i] * ctx.gain, raw);
      for (let c = 0; c < 3; c++) {
        const d = interpU(Math.log10(Math.max(raw[c], 0) + 1e-10), film.le0, film.step, film.curves[c]);
        silver[c][i] = couplerSilver(film, d, c);
      }
    }
    const field = planes(n), tmp = new Float32Array(n), acc = new Float32Array(n);
    for (let m = 0; m < 3; m++) {
      const corr = new Float32Array(n);
      for (let i = 0; i < n; i++) corr[i] = (silver[0][i] * M[0][m] + silver[1][i] * M[1][m] + silver[2][i] * M[2][m]) * s.couplers;
      for (const [amp, ratio] of EXP_FIT) {
        blur(corr, w, h, (ratio * COUPLER_TAIL_UM) / pixelUm, acc, tmp);
        for (let i = 0; i < n; i++) field[m][i] += amp * acc[i];
      }
    }
    return { w, h, scale, field };
  }

  function sampleField(F, c, fx, fy) {
    const { w, h } = F;
    fx = fx < 0 ? 0 : fx > w - 1 ? w - 1 : fx;
    fy = fy < 0 ? 0 : fy > h - 1 ? h - 1 : fy;
    const x0 = Math.min(fx | 0, w - 1), y0 = Math.min(fy | 0, h - 1);
    const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
    const tx = fx - x0, ty = fy - y0, P = F.field[c];
    const a = P[y0 * w + x0], b = P[y0 * w + x1], cc = P[y1 * w + x0], d = P[y1 * w + x1];
    const top = a + (b - a) * tx, bot = cc + (d - cc) * tx;
    return top + (bot - top) * ty;
  }

  // Halation and in-emulsion scatter on linear film exposure (diffusion.py, apply_halation_um).
  function applyHalation(ctx, raw, w, h) {
    const H = ctx.film.raw.halation, s = ctx.s, px = ctx.pixelUm;
    const n = w * h, tmp = new Float32Array(n), a = new Float32Array(n), acc = new Float32Array(n);
    for (let c = 0; c < 3; c++) {
      const E = raw[c];
      if (s.scatter > 0) {
        const wt = H.scatterTailWeight[c];
        blur(E, w, h, Math.max(H.scatterCoreUm[c] / px, 1e-6), a, tmp);
        acc.fill(0);
        for (let i = 0; i < n; i++) acc[i] = (1 - wt) * a[i];
        for (const [amp, ratio] of EXP_FIT) {
          blur(E, w, h, Math.max((ratio * H.scatterTailUm[c]) / px, 1e-6), a, tmp);
          for (let i = 0; i < n; i++) acc[i] += wt * amp * a[i];
        }
        for (let i = 0; i < n; i++) E[i] = (1 - s.scatter) * E[i] + s.scatter * acc[i];
      }
      const aTot = H.strength[c] * s.halation;
      if (aTot > 0) {
        const N = H.bounces, rho = H.decay;
        let dsum = 0; for (let k = 1; k <= N; k++) dsum += Math.pow(rho, k - 1);
        acc.fill(0);
        for (let k = 1; k <= N; k++) {
          blur(E, w, h, Math.max((H.sigmaUm[c] * Math.sqrt(k)) / px, 1e-6), a, tmp);
          const wk = Math.pow(rho, k - 1) / dsum;
          for (let i = 0; i < n; i++) acc[i] += wk * a[i];
        }
        const norm = 1 / (1 + aTot);
        for (let i = 0; i < n; i++) E[i] = (E[i] + aTot * acc[i]) * norm;
      }
    }
  }

  function halationReach(ctx) {
    const H = ctx.film.raw.halation, s = ctx.s, px = ctx.pixelUm;
    let r = 0;
    for (let c = 0; c < 3; c++) {
      if (s.scatter > 0) r = Math.max(r, blurReach(H.scatterCoreUm[c] / px), blurReach((2.7684 * H.scatterTailUm[c]) / px));
      if (H.strength[c] * s.halation > 0) r = Math.max(r, blurReach((H.sigmaUm[c] * Math.sqrt(H.bounces)) / px));
    }
    return r;
  }

  // Sublayer grain (grain.py, apply_grain_to_density_layers). Binomial thinning
  // of a Poisson count is Poisson, so each sublayer is one Poisson draw.
  function applyGrain(ctx, dens, w, h, rng) {
    const { film, s } = ctx;
    const n = w * h, px2 = ctx.pixelUm * ctx.pixelUm;
    const area = GRAIN.particleAreaUm2 * s.grainSize;
    const out = planes(n);
    for (let c = 0; c < 3; c++) {
      const maxes = film.layerMax[c];
      const total = maxes[0] + maxes[1] + maxes[2];
      const L = [];
      for (let l = 0; l < 3; l++) {
        const frac = total > 0 ? maxes[l] / total : 0;
        const dminL = frac * GRAIN.densityMin[c];
        const dmaxL = maxes[l] + dminL;
        const nPart = (px2 * frac) / (area * GRAIN.particleScale[c] * GRAIN.particleScaleLayers[l]);
        const od = nPart > 0 ? dmaxL / nPart : 0;
        L.push({ t: film.layerTables[c][l], dminL, dmaxL, nPart, od, sigma: GRAIN.blurDyeCloudsUm * Math.sqrt(od) });
      }
      const u = GRAIN.uniformity[c], D = dens[c], O = out[c], sign = film.positive ? -1 : 1;
      // dye-cloud blur only matters for very small pixels (sigma = sqrt(od) px)
      const perLayer = L.map((q) => (q.sigma >= 0.35 ? new Float32Array(n) : null));
      for (let i = 0; i < n; i++) {
        const xq = sign * D[i];
        let sum = 0;
        for (let l = 0; l < 3; l++) {
          const q = L[l];
          if (!(q.nPart > 0)) continue;
          const t = q.t;
          const dl = interpU(xq, t.x0, t.step, t.ys) + q.dminL;
          let p = dl / q.dmaxL;
          p = p < 1e-6 ? 1e-6 : p > 1 - 1e-6 ? 1 - 1e-6 : p;
          const sat = 1 - p * u * (1 - 1e-6);
          const g = rng.poisson((q.nPart * p) / sat) * q.od * sat;
          if (perLayer[l]) perLayer[l][i] = g; else sum += g;
        }
        O[i] = sum;
      }
      const tmp = new Float32Array(n), b = new Float32Array(n);
      for (let l = 0; l < 3; l++) {
        if (!perLayer[l]) continue;
        blur(perLayer[l], w, h, L[l].sigma, b, tmp);
        for (let i = 0; i < n; i++) O[i] += b[i];
      }
      for (let i = 0; i < n; i++) O[i] -= GRAIN.densityMin[c];
      if (GRAIN.blur > 0) { blur(O, w, h, GRAIN.blur, b, tmp); O.set(b); }
      if (s.grainAmount !== 1) {
        // blend toward the grain-free (but equally blurred) density
        blur(D, w, h, GRAIN.blur, b, tmp);
        for (let i = 0; i < n; i++) O[i] = b[i] + s.grainAmount * (O[i] - b[i]);
      }
    }
    return out;
  }

  // Process one block (with its overlap) of the frame; returns linear output planes.
  function processBlock(ctx, lin, w, h, bx, by, rng) {
    const { film, s } = ctx;
    const n = w * h;
    // 1. film exposure
    const raw = planes(n), t3 = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      filmRaw(film, lin[0][i] * ctx.gain, lin[1][i] * ctx.gain, lin[2][i] * ctx.gain, t3);
      raw[0][i] = t3[0]; raw[1][i] = t3[1]; raw[2][i] = t3[2];
    }
    if (s.spatial) applyHalation(ctx, raw, w, h);
    // 2. develop
    const logE = raw;
    for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) logE[c][i] = Math.log10(Math.max(logE[c][i], 0) + 1e-10);
    const dens = planes(n);
    for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) dens[c][i] = interpU(logE[c][i], film.le0, film.step, film.curves[c]);
    if (s.couplers > 0) {
      const M = film.couplers, tmp = new Float32Array(n), local = new Float32Array(n), blurred = new Float32Array(n);
      const sigma = s.spatial ? COUPLER_DIFFUSION_UM / ctx.pixelUm : 0;
      const tailW = s.spatial ? COUPLER_TAIL_WEIGHT : 0;
      const corr = planes(n);
      for (let m = 0; m < 3; m++) {
        for (let i = 0; i < n; i++) {
          local[i] = (couplerSilver(film, dens[0][i], 0) * M[0][m] + couplerSilver(film, dens[1][i], 1) * M[1][m] +
                      couplerSilver(film, dens[2][i], 2) * M[2][m]) * s.couplers;
        }
        blur(local, w, h, sigma, blurred, tmp);
        corr[m].set(blurred);
      }
      for (let m = 0; m < 3; m++) {
        const C = corr[m];
        for (let y = 0, i = 0; y < h; y++) {
          for (let x = 0; x < w; x++, i++) {
            let k = (1 - tailW) * C[i];
            if (tailW > 0) k += tailW * sampleField(ctx.tail, m, (bx + x + 0.5) / ctx.tail.scale - 0.5, (by + y + 0.5) / ctx.tail.scale - 0.5);
            C[i] = logE[m][i] - k;
          }
        }
      }
      for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) dens[c][i] = interpU(corr[c][i], film.le0, film.step, film.curves0[c]);
    }
    const grained = s.grain ? applyGrain(ctx, dens, w, h, rng) : dens;
    // 3. print (or scan the slide)
    const out = planes(n), v = [0, 0, 0];
    if (film.positive) {
      for (let i = 0; i < n; i++) {
        lut3(film.scan, grained[0][i], grained[1][i], grained[2][i], v);
        out[0][i] = v[0]; out[1][i] = v[1]; out[2][i] = v[2];
      }
    } else {
      const P = ctx.paper, E = ctx.enlarger;
      for (let i = 0; i < n; i++) {
        lut3(E, grained[0][i], grained[1][i], grained[2][i], v);
        const pc = interpU(v[0], P.le0, P.step, P.curves[0]);
        const pm = interpU(v[1], P.le0, P.step, P.curves[1]);
        const py = interpU(v[2], P.le0, P.step, P.curves[2]);
        lut3(P.scan, pc, pm, py, v);
        out[0][i] = v[0]; out[1][i] = v[1]; out[2][i] = v[2];
      }
    }
    // 4. scanner: glare (stochastic veiling flare), unsharp mask
    if (s.glare) {
      const m = GLARE.percent, sd = GLARE.roughness * m;
      const sig2 = Math.log(1 + (sd * sd) / (m * m)), mu = Math.log(m) - sig2 / 2, sg = Math.sqrt(sig2);
      const gl = new Float32Array(n), b = new Float32Array(n), tmp = new Float32Array(n);
      for (let i = 0; i < n; i++) gl[i] = Math.exp(mu + sg * rng.normal());
      blur(gl, w, h, s.spatial ? GLARE.blur : 0, b, tmp);
      for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) out[c][i] += b[i] / 100;
    }
    if (s.spatial && s.sharpen > 0) {
      const b = new Float32Array(n), tmp = new Float32Array(n);
      for (let c = 0; c < 3; c++) {
        blur(out[c], w, h, UNSHARP.sigma, b, tmp);
        const O = out[c], a = UNSHARP.amount * s.sharpen;
        for (let i = 0; i < n; i++) O[i] += a * (O[i] - b[i]);
      }
    }
    return out;
  }

  const DEFAULTS = {
    film: "kodak_portra_400",
    print: null,              // null = the film's own target paper
    exposureEv: 0,
    autoExposure: true,
    printExposure: 1,
    printExposureCompensation: true,
    yellowShift: 0,
    magentaShift: 0,
    formatMm: 35,
    grain: true,
    grainSize: 1,             // x upstream particle area (0.2 um^2)
    grainAmount: 1,
    halation: 1,
    scatter: 1,
    couplers: 1,
    glare: true,
    sharpen: 1,
    spatial: true,
    seed: 1,
  };

  // Whole-frame decisions, made once per render and shared by every tile:
  // metering, the enlarger table and the coupler tail field. The plan is
  // plain data so it can be posted to workers.
  function plan(src, sw, sh, settings) {
    const s = Object.assign({}, DEFAULTS, settings);
    const film = getFilm(s.film);
    const paper = film.positive ? null : getPrint(s.print || film.raw.targetPrint);
    const ctx = { s, film, paper, pixelUm: (s.formatMm * 1000) / Math.max(sw, sh) };
    let auto = 0;
    if (s.autoExposure) {
      const sc = Math.max(1, Math.max(sw, sh) / 256);
      const w = Math.max(1, Math.round(sw / sc)), h = Math.max(1, Math.round(sh / sc));
      auto = centerWeightedEv(readLinear(src, sw, 0, 0, w, h, sc), w, h);
    }
    ctx.gain = Math.pow(2, auto + s.exposureEv);
    const p = {
      s, sw, sh, autoEv: auto, gain: ctx.gain, pixelUm: ctx.pixelUm,
      filmName: film.raw.name, paperId: paper ? paper.raw.id : null, paperName: paper ? paper.raw.name : null,
      enlarger: paper ? cachedEnlarger(film, paper, s) : null,
      tail: s.couplers > 0 && s.spatial ? couplerTailField(ctx, src, sw, sh) : null,
    };
    ctx.tail = p.tail;
    p.reach = s.spatial
      ? halationReach(ctx) + blurReach(COUPLER_DIFFUSION_UM / ctx.pixelUm) + blurReach(GRAIN.blur) +
        blurReach(GLARE.blur) + blurReach(UNSHARP.sigma)
      : 0;
    return p;
  }

  // Render the rectangle [x0, x1) x [y0, y1) of the frame. `src` holds the
  // source rows [srcY0, srcY0 + rows) at full frame width; it must cover the
  // rectangle plus plan.reach rows on each side (clamped to the frame).
  function renderRect(p, src, srcY0, x0, y0, x1, y1, onProgress) {
    const s = p.s, sw = p.sw, sh = p.sh;
    const film = getFilm(s.film);
    const ctx = { s, film, paper: p.paperId ? getPrint(p.paperId) : null, pixelUm: p.pixelUm, gain: p.gain, enlarger: p.enlarger, tail: p.tail };
    const reach = p.reach, w = x1 - x0;
    const out = new Uint8ClampedArray(w * (y1 - y0) * 4);
    const xa = Math.max(0, x0 - reach), xb = Math.min(sw, x1 + reach), bw = xb - xa;
    for (let ty = y0; ty < y1; ty += TILE_ROWS) {
      const ty1 = Math.min(y1, ty + TILE_ROWS);
      const ya = Math.max(0, ty - reach), yb = Math.min(sh, ty1 + reach);
      const lin = readLinear(src, sw, xa, ya - srcY0, bw, yb - ya, 1);
      // seed by absolute tile position, so any split across workers renders the same frame
      const rng = makeRng((s.seed * 1000003 + Math.floor(ty / TILE_ROWS) * 7919 + Math.floor(x0 / 64) * 104729) | 0);
      const res = processBlock(ctx, lin, bw, yb - ya, xa, ya, rng);
      for (let y = ty; y < ty1; y++) {
        let q = (y - ya) * bw + (x0 - xa), o = ((y - y0) * w) * 4;
        for (let x = 0; x < w; x++, q++, o += 4) {
          out[o] = encode8(res[0][q]); out[o + 1] = encode8(res[1][q]); out[o + 2] = encode8(res[2][q]);
          out[o + 3] = 255;
        }
      }
      if (onProgress) onProgress((ty1 - y0) / (y1 - y0));
    }
    return out;
  }

  // Single-threaded convenience: plan + render the whole frame.
  function render(src, sw, sh, settings, onProgress) {
    const p = plan(src, sw, sh, settings);
    const pixels = renderRect(p, src, 0, 0, 0, sw, sh, onProgress);
    return { pixels, autoEv: p.autoEv, film: p.filmName, paper: p.paperName };
  }

  const api = { init, plan, renderRect, render, DEFAULTS, _internal: { getFilm, getPrint, buildEnlarger, filmRaw, lut3, interpU } };
  root.SpektraEngine = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
