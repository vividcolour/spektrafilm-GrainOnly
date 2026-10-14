// Colour space table and matrix maths (CPU only). Same list as SaturnOFX.
// Everything goes through linear DaVinci Wide Gamut.
#pragma once

#include <cmath>

#include "GrainShared.h"

namespace spektragrain {

struct ColourSpace {
  const char *label;
  double xy[8];  // Rx Ry Gx Gy Bx By Wx Wy
  sg_u32 tf;
};

#define SG_D65 0.3127, 0.3290
#define SG_ACESW 0.32168, 0.33767

static const ColourSpace kColourSpaces[] = {
  {"DaVinci Wide Gamut / Intermediate", {0.8, 0.313, 0.1682, 0.9877, 0.079, -0.1155, SG_D65}, SG_TF_DAVINCI_INT},
  {"DaVinci Wide Gamut / Linear", {0.8, 0.313, 0.1682, 0.9877, 0.079, -0.1155, SG_D65}, SG_TF_LINEAR},
  {"ACEScct (AP1)", {0.713, 0.293, 0.165, 0.83, 0.128, 0.044, SG_ACESW}, SG_TF_ACESCCT},
  {"ACEScg (AP1 Linear)", {0.713, 0.293, 0.165, 0.83, 0.128, 0.044, SG_ACESW}, SG_TF_LINEAR},
  {"ACES2065-1 (AP0 Linear)", {0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, SG_ACESW}, SG_TF_LINEAR},
  {"ARRI LogC3 (EI800) / AWG3", {0.684, 0.313, 0.221, 0.848, 0.0861, -0.102, SG_D65}, SG_TF_LOGC3},
  {"ARRI LogC4 / AWG4", {0.7347, 0.2653, 0.1424, 0.8576, 0.0991, -0.0308, SG_D65}, SG_TF_LOGC4},
  {"Sony S-Log3 / S-Gamut3.Cine", {0.766, 0.275, 0.225, 0.8, 0.089, -0.087, SG_D65}, SG_TF_SLOG3},
  {"Sony S-Log3 / S-Gamut3", {0.73, 0.28, 0.14, 0.855, 0.1, -0.05, SG_D65}, SG_TF_SLOG3},
  {"Panasonic V-Log / V-Gamut", {0.73, 0.28, 0.165, 0.84, 0.1, -0.03, SG_D65}, SG_TF_VLOG},
  {"RED Log3G10 / REDWideGamutRGB",
   {0.780308, 0.304253, 0.121595, 1.493994, 0.095612, -0.084589, SG_D65},
   SG_TF_LOG3G10},
  {"Canon Log 3 / Cinema Gamut", {0.74, 0.27, 0.17, 1.14, 0.08, -0.1, SG_D65}, SG_TF_CLOG3},
  {"Fujifilm F-Log2 / F-Gamut", {0.708, 0.292, 0.17, 0.797, 0.131, 0.046, SG_D65}, SG_TF_FLOG2},
  {"Apple Log / Rec.2020", {0.708, 0.292, 0.17, 0.797, 0.131, 0.046, SG_D65}, SG_TF_APPLELOG},
  {"Blackmagic Film Gen5 / BMWG",
   {0.7177215, 0.3171181, 0.228041, 0.861569, 0.1005841, -0.0820452, 0.312717, 0.3290312},
   SG_TF_BMDGEN5},
  {"DJI D-Log / D-Gamut", {0.71, 0.31, 0.21, 0.88, 0.09, -0.08, SG_D65}, SG_TF_DLOG},
  {"Rec.709 / Gamma 2.4", {0.64, 0.33, 0.3, 0.6, 0.15, 0.06, SG_D65}, SG_TF_GAMMA24},
  {"Rec.709 / Linear", {0.64, 0.33, 0.3, 0.6, 0.15, 0.06, SG_D65}, SG_TF_LINEAR},
  {"Rec.2020 / Gamma 2.4", {0.708, 0.292, 0.17, 0.797, 0.131, 0.046, SG_D65}, SG_TF_GAMMA24},
  {"Rec.2020 / Linear", {0.708, 0.292, 0.17, 0.797, 0.131, 0.046, SG_D65}, SG_TF_LINEAR},
  {"sRGB", {0.64, 0.33, 0.3, 0.6, 0.15, 0.06, SG_D65}, SG_TF_SRGB},
  {"Display P3", {0.68, 0.32, 0.265, 0.69, 0.15, 0.06, SG_D65}, SG_TF_SRGB},
};
static const int kColourSpaceCount = static_cast<int>(sizeof(kColourSpaces) / sizeof(kColourSpaces[0]));
static const int kSpaceDWG = 0;         // working space (and the default Input)
static const int kSpaceDWGLinear = 1;
static const int kOutputMatchInput = -1;

// 3x3 matrix helpers, row-major
inline void cs_mul(const double *a, const double *b, double *o) {
  double t[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) t[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  for (int i = 0; i < 9; ++i) o[i] = t[i];
}

inline void cs_inv(const double *m, double *o) {
  const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
  const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const double det = a * A + b * B + c * C;
  const double t[9] = {A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d),
                       C, -(a * h - b * g), a * e - b * d};
  for (int k = 0; k < 9; ++k) o[k] = t[k] / det;
}

inline void cs_vec(const double *m, const double *v, double *o) {
  double t[3];
  for (int r = 0; r < 3; ++r) t[r] = m[r * 3] * v[0] + m[r * 3 + 1] * v[1] + m[r * 3 + 2] * v[2];
  o[0] = t[0];
  o[1] = t[1];
  o[2] = t[2];
}

// RGB -> XYZ normalised primary matrix.
inline void cs_npm(const double *xy, double *M) {
  double P[9];
  for (int k = 0; k < 3; ++k) {
    const double x = xy[k * 2], y = xy[k * 2 + 1];
    P[0 * 3 + k] = x / y;
    P[1 * 3 + k] = 1.0;
    P[2 * 3 + k] = (1.0 - x - y) / y;
  }
  const double W[3] = {xy[6] / xy[7], 1.0, (1.0 - xy[6] - xy[7]) / xy[7]};
  double Pi[9], S[3];
  cs_inv(P, Pi);
  cs_vec(Pi, W, S);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) M[r * 3 + c] = P[r * 3 + c] * S[c];
}

// Bradford chromatic adaptation from white ws to white wd (xy).
inline void cs_bradford(const double *ws, const double *wd, double *M) {
  static const double B[9] = {0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296};
  double Bi[9];
  cs_inv(B, Bi);
  const double Ws[3] = {ws[0] / ws[1], 1.0, (1.0 - ws[0] - ws[1]) / ws[1]};
  const double Wd[3] = {wd[0] / wd[1], 1.0, (1.0 - wd[0] - wd[1]) / wd[1]};
  double s[3], d[3];
  cs_vec(B, Ws, s);
  cs_vec(B, Wd, d);
  const double D[9] = {d[0] / s[0], 0, 0, 0, d[1] / s[1], 0, 0, 0, d[2] / s[2]};
  double T[9];
  cs_mul(D, B, T);
  cs_mul(Bi, T, M);
}

// Linear RGB of space `src` -> linear RGB of space `dst`.
inline void cs_rgbToRgb(int src, int dst, double *M) {
  const double *a = kColourSpaces[src].xy;
  const double *b = kColourSpaces[dst].xy;
  double Ma[9], Mb[9], Mbi[9], C[9], T[9];
  cs_npm(a, Ma);
  cs_npm(b, Mb);
  cs_inv(Mb, Mbi);
  const bool sameWhite = std::fabs(a[6] - b[6]) < 1e-5 && std::fabs(a[7] - b[7]) < 1e-5;
  if (sameWhite) {
    cs_mul(Mbi, Ma, M);
  } else {
    cs_bradford(a + 6, b + 6, C);
    cs_mul(C, Ma, T);
    cs_mul(Mbi, T, M);
  }
}

// outSpace < 0 means Match Input.
inline void setupColourSpaces(SGParams &k, int inSpace, int outSpace) {
  const bool match = outSpace < 0 || outSpace == inSpace;
  if (outSpace < 0) outSpace = inSpace;
  double Min[9], Mout[9];
  cs_rgbToRgb(inSpace, kSpaceDWG, Min);
  cs_rgbToRgb(kSpaceDWG, outSpace, Mout);
  for (int i = 0; i < 9; ++i) {
    k.inMat[i] = static_cast<float>(Min[i]);
    k.outMat[i] = static_cast<float>(Mout[i]);
  }
  k.inTF = kColourSpaces[inSpace].tf;
  k.outTF = kColourSpaces[outSpace].tf;
  k.matchInput = match ? 1u : 0u;
}

inline float decode(sg_u32 tf, float v) { return sg_decode(tf, v); }
inline float encode(sg_u32 tf, float l) { return sg_encode(tf, l); }

}  // namespace spektragrain
