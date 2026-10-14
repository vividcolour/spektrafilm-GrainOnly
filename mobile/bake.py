#!/usr/bin/env python3
"""Bake the data for Spektra Mobile from upstream spektrafilm.

Runs spektrafilm's own pipeline (https://github.com/andreavolpato/spektrafilm)
to export everything the phone app needs:

  - per film: density curves, sublayer curves, the curves before DIR couplers,
    the coupler matrix, halation preset, spectral dye densities, and a 2D
    chromaticity table for camera RGB -> film exposure (spectral upsampling,
    hanatos2025). Positive films also get a 3D table for the film scan.
  - per print paper: spectral sensitivity, characteristic curves and a 3D
    table for the print scan (viewing illuminant, CIE observer, output gamut
    compression).
  - the enlarger lamp, dichroic filters and the neutral filter database.

Everything spatial or random (grain, couplers diffusion, halation, glare,
unsharp mask) and the enlarger exposure itself runs in the app.

Usage (inside an environment with spektrafilm installed):
  python mobile/bake.py mobile/spektra_data.js
"""
import base64
import json
import sys

import numpy as np

from spektrafilm.model.color_filters import custom_dichroic_filters
from spektrafilm.model.couplers import compute_density_curves_before_dir_couplers, compute_dir_couplers_matrix
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.api import digest_params, init_params
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.utils.io import read_neutral_print_filters

FILMS = [
    "kodak_portra_400", "kodak_portra_160", "kodak_portra_800", "kodak_portra_800_push1",
    "kodak_portra_800_push2", "kodak_ektar_100", "kodak_gold_200", "kodak_ultramax_400",
    "kodak_vision3_50d", "kodak_vision3_250d", "kodak_vision3_200t", "kodak_vision3_500t",
    "kodak_verita_200d", "fujifilm_pro_400h", "fujifilm_c200", "fujifilm_xtra_400",
    "kodak_ektachrome_100", "kodak_kodachrome_64", "fujifilm_provia_100f", "fujifilm_velvia_100",
]
PRINTS = [
    "kodak_portra_endura", "kodak_endura_premier", "kodak_ektacolor_edge", "kodak_supra_endura",
    "kodak_ultra_endura", "fujifilm_crystal_archive_typeii", "kodak_2383", "kodak_2393",
]
CHROMA_N = 33   # 2D chromaticity grid for rgb -> film raw
SCAN_N = 25     # 3D density grid for the scan tables
LOG_LO, LOG_HI = -5.0, 0.5  # range of the encoded log10 linear RGB


def r(a, digits=5):
    return np.round(np.nan_to_num(np.asarray(a, dtype=np.float64)), digits).tolist()


def pack_u16(values, lo, hi):
    q = np.clip((np.asarray(values) - lo) / (hi - lo), 0.0, 1.0)
    q = np.round(q * 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def pipeline(film, paper, scan_film=False):
    p = init_params(film, paper)
    p.debug.lut_mode = True
    p.io.input_color_space = "sRGB"
    p.io.input_cctf_decoding = False
    p.io.output_color_space = "sRGB"
    p.io.output_cctf_encoding = False
    p.io.scan_film = scan_film
    p = digest_params(p)
    return SimulationPipeline(p), p


def chroma_table(pipe):
    # raw(k * rgb) = k * raw(rgb), so sample rgb on the r + g + b = 1 plane.
    n = CHROMA_N
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="xy")
    # outside the triangle, clamp onto its hypotenuse
    s = u + v
    over = s > 1
    u2, v2 = np.where(over, u / s, u), np.where(over, v / s, v)
    rgb = np.stack([u2, v2, 1 - u2 - v2], axis=-1)
    rgb = np.maximum(rgb, 0)
    log_raw = pipe.process(rgb, inject="rgb_pre", collect="log_e_film")
    return 10 ** log_raw  # [v][u][3]


def scan_table(pipe, lo, hi, inject):
    n = SCAN_N
    axes = [np.linspace(lo[c], hi[c], n) for c in range(3)]
    c, m, y = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    cmy = np.stack([c, m, y], axis=-1).reshape(n * n, n, 3)
    rgb = pipe.process(cmy, inject=inject, collect="rgb_out")
    return np.log10(np.fmax(rgb, 10 ** LOG_LO)).reshape(n, n, n, 3)


def film_entry(stock):
    prof = load_profile(stock)
    d, info = prof.data, prof.info
    paper = info.target_print or "kodak_portra_endura"
    positive = info.type == "positive"
    pipe, params = pipeline(stock, paper, scan_film=positive)
    dc = np.asarray(d.density_curves)
    dcn = dc - np.nanmin(dc, axis=0)
    le = np.asarray(d.log_exposure)
    couplers = params.film_render.dir_couplers
    matrix = compute_dir_couplers_matrix(couplers) * couplers.amount
    curves0 = compute_density_curves_before_dir_couplers(dcn, le, matrix, positive=positive)
    hal = params.film_render.halation
    cd = np.asarray(d.channel_density, dtype=np.float64)
    base = np.asarray(d.base_density, dtype=np.float64)
    entry = {
        "id": stock,
        "name": info.name,
        "positive": positive,
        "use": info.use,
        "targetPrint": paper,
        "logE0": float(le[0]),
        "logEStep": float((le[-1] - le[0]) / (len(le) - 1)),
        "curves": [r(dcn[:, c]) for c in range(3)],
        "curves0": [r(curves0[:, c]) for c in range(3)],
        "layers": [[r(d.density_curves_layers[:, l, c]) for l in range(3)] for c in range(3)],
        "rawCurves": [r(dc[:, c]) for c in range(3)],
        "couplers": r(matrix, 6),
        "halation": {
            "strength": list(hal.halation_strength),
            "sigmaUm": list(hal.halation_first_sigma_um),
            "bounces": hal.halation_n_bounces,
            "decay": hal.halation_bounce_decay,
            "scatterCoreUm": list(hal.scatter_core_um),
            "scatterTailUm": list(hal.scatter_tail_um),
            "scatterTailWeight": list(hal.scatter_tail_weight),
        },
        "channelDensity": [r(cd[:, c]) for c in range(3)],
        "baseDensity": r(base),
        "spectralValid": (~np.isnan(cd).any(axis=1) & ~np.isnan(base)).astype(int).tolist(),
        "chroma": r(chroma_table(pipe).reshape(-1), 5),
    }
    if positive:
        lo = -np.asarray(params.film_render.grain.density_min)
        hi = np.nanmax(dc, axis=0)
        entry["scan"] = {"lo": r(lo), "hi": r(hi), "data": pack_u16(scan_table(pipe, lo, hi, "cmy_film"), LOG_LO, LOG_HI)}
    print(f"film {stock}", file=sys.stderr)
    return entry


def print_entry(stock):
    prof = load_profile(stock)
    d, info = prof.data, prof.info
    pipe, _ = pipeline("kodak_portra_400", stock)
    le = np.asarray(d.log_exposure)
    ramp = np.repeat(le[None, :, None], 3, axis=2)
    curves = pipe.process(ramp, inject="log_e_print", collect="cmy_print")[0]
    dc = np.asarray(d.density_curves)
    lo, hi = np.nanmin(dc, axis=0), np.nanmax(dc, axis=0)
    sens = np.nan_to_num(10 ** np.asarray(d.log_sensitivity, dtype=np.float64))
    print(f"print {stock}", file=sys.stderr)
    return {
        "id": stock,
        "name": info.name,
        "logE0": float(le[0]),
        "logEStep": float((le[-1] - le[0]) / (len(le) - 1)),
        "curves": [r(curves[:, c]) for c in range(3)],
        "sensitivity": [r(sens[:, c], 7) for c in range(3)],
        "scan": {"lo": r(lo), "hi": r(hi), "data": pack_u16(scan_table(pipe, lo, hi, "cmy_print"), LOG_LO, LOG_HI)},
    }


def main():
    out = sys.argv[1]
    data = {
        "source": "spektrafilm (Andrea Volpato), https://github.com/andreavolpato/spektrafilm",
        "scanN": SCAN_N,
        "chromaN": CHROMA_N,
        "logLo": LOG_LO,
        "logHi": LOG_HI,
        "enlargerLamp": r(standard_illuminant("TH-KG3"), 6),
        "dichroics": [r(custom_dichroic_filters.filters[:, c], 6) for c in range(3)],
        "neutralFilters": read_neutral_print_filters(),
        "films": [film_entry(s) for s in FILMS],
        "prints": [print_entry(s) for s in PRINTS],
    }
    with open(out, "w") as f:
        f.write("// Generated by mobile/bake.py from spektrafilm - do not edit.\n")
        f.write("// Film and paper data: spektrafilm profiles by Andrea Volpato, CC BY-SA 4.0 (resampled and derived).\n")
        f.write("const SPEKTRA_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
