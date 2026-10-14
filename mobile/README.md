# Spektra Pocket

A phone-sized port of [spektrafilm](https://github.com/andreavolpato/spektrafilm)
by Andrea Volpato. It runs entirely in the browser: open a photo, pick a stock,
and it goes through spektrafilm's pipeline:

1. **Expose**: camera RGB to film-layer exposure through the stock's spectral
   sensitivities (spektrafilm's hanatos2025 spectral upsampling, baked per film)
2. **Halation and scatter** with the stock's antihalation preset
3. **Develop**: characteristic curves, DIR couplers (with their diffusion), and
   spektrafilm's sublayer particle grain
4. **Print**: film dye densities × enlarger lamp × dichroic filters × paper
   sensitivity, using spektrafilm's neutral filter database, then the paper's
   curves. Slide films skip this step.
5. **Scan**: viewing illuminant, CIE observer, output gamut compression, glare
   and scanner unsharp mask

Open `dist/index.html` on your phone (any modern browser), or host the folder
anywhere static. Photos never leave the device.

## How it relates to upstream

`bake.py` imports upstream spektrafilm and runs its own pipeline to export the
parts that don't depend on the image: the camera-to-film table, the scan
tables, curves, coupler matrices and spectra. `engine.js` reimplements the rest
in JavaScript. With grain and spatial effects off, its output matches
upstream's `simulate()` to within 1–2 of 255 code values on Portra 400, Vision3
500T, Velvia 100, C200 and Gold 200 at 0 and +1.5 EV. Grain standard deviation
on flat patches matches upstream's within the 8-bit rounding.

What's simplified: the exponential tail of the coupler diffusion is computed
on a 384 px copy of the frame, large blurs use three box passes, Poisson
sampling switches to a skew-corrected Gaussian above a mean of 12, and the
enlarger and scan run through 25³ tables. Diffusion filters, lens blur, print
curve morphing, preflash and the highlight boost aren't exposed.

## Rebuilding

```
pip install -e /path/to/spektrafilm      # upstream, for bake.py
python3 mobile/bake.py mobile/spektra_data.js
python3 mobile/build.py                  # writes dist/index.html
```

## License

Code is GPL-3.0, like spektrafilm. `spektra_data.js` is derived from
spektrafilm's profiles and shares their CC BY-SA 4.0 license
(`../SPEKTRAFILM_LICENSE.txt`).
