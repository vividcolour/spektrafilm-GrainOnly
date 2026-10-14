# spektrafilm GrainOnly

A fork of [spektrafilm](https://github.com/andreavolpato/spektrafilm) by
Andrea Volpato, cut down to just the grain and turned into an OFX plugin for
DaVinci Resolve (macOS), **Spektra RMS Grain**. It's meant for a grain node
after your look, without running the whole film simulation.

The grain is a C++/Metal port of spektrafilm's sublayer particle model
(`spektrafilm/model/grain.py`). It's generated per dye layer from each stock's
measured curves and scaled to the stock's published RMS granularity, so it
gets finer and coarser with exposure like real negative instead of sitting on
top as flat noise. Everything else from the original project (the Python
package, GUI, LUT creator, print stocks) has been removed here; head to the
upstream repo for the full simulation.

Runs on the GPU (Metal) with a CPU fallback. Apple Silicon and Intel,
macOS 11 or later.

## Install

Grab the `.pkg` from [Releases](https://github.com/vividcolour/spektrafilm-GrainOnly/releases),
quit Resolve and run it. It installs to `/Library/OFX/Plugins`, and the node
shows up under OpenFX > VividColour > Spektra RMS Grain.

It isn't signed, so if macOS blocks it, right-click the pkg and choose Open.

## Controls

- **Film Stock**: 20 camera stocks from spektrafilm (Vision3, Portra, Ektar,
  Gold, Ektachrome, Kodachrome, Fuji and so on)
- **Film Format**: Super 8 up to IMAX. Smaller formats mean bigger grain
- **Custom RMS**: override the stock's RMS value
- **Exposure Offset**: where the image sits on the stock's curve
- **Grain Amount, Size, Softness, Saturation**
- **Grain Aspect Ratio**: 1.0 is round, above 1 stretches horizontally,
  below 1 vertically
- **Animate / Seed**
- **Colour Space**: Input and Output. Default is DaVinci Wide Gamut /
  Intermediate with Output on Match Input, which leaves the image untouched
  apart from the grain
- **View**: final image, or grain on its own over mid grey

## Building

```
./build.sh
```

Needs the Xcode command line tools and Python 3. It runs the tests, builds a
universal bundle and puts a `.pkg` and `.dmg` in `dist/`. Set `SIGN_ID` and
`INSTALLER_SIGN_ID` if you want to sign with a Developer ID.

`src/StockData.h` is generated from the spektrafilm profiles in
`tools/profiles` by `tools/gen_stocks.py`.

## Credits and license

spektrafilm by Andrea Volpato, https://github.com/andreavolpato/spektrafilm.
If the grain is useful to you, consider supporting the original project.

Code is GPL-3.0, like spektrafilm (`LICENSE`). The film profiles in
`tools/profiles` are unmodified copies licensed CC BY-SA 4.0
(`SPEKTRAFILM_LICENSE.txt`); `src/StockData.h` is resampled from them and
shares that license. OpenFX headers are BSD-3-Clause. Details in `legal/`.
