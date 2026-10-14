#!/usr/bin/env bash
# Builds the Spektra RMS Grain OFX bundle (universal arm64 + x86_64), runs the
# tests, and packages an installer .pkg inside a .dmg.
#
#   ./build.sh                 build + test + package (ad-hoc signed)
#   SIGN_ID="Developer ID Application: ..." ./build.sh   sign the bundle with a real identity
#   INSTALLER_SIGN_ID="Developer ID Installer: ..." ./build.sh   also sign the .pkg
#   SKIP_TESTS=1 ./build.sh
set -euo pipefail
export COPYFILE_DISABLE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="SpektraRMSGrain"
LABEL="Spektra RMS Grain"
BUNDLE_ID="com.vividcolour.SpektraRMSGrain"
VERSION="1.1.0"
BUILD="$ROOT/build"
GEN="$BUILD/gen"
BUNDLE="$BUILD/stage/$NAME.ofx.bundle"
DIST="$ROOT/dist"
MACOS_MIN="11.0"

mkdir -p "$GEN" "$BUILD/test" "$DIST"

echo "==> Generating sources"
if [[ -d "$ROOT/tools/profiles" ]]; then
  python3 -I "$ROOT/tools/gen_stocks.py" "$ROOT/tools/profiles" "$ROOT/src/StockData.h"
fi
{ printf 'R"SGMETAL('; cat "$ROOT/src/GrainShared.h" "$ROOT/src/GrainKernels.metal"; printf ')SGMETAL"\n'; } > "$GEN/MetalSource.inc"

CXXFLAGS=(-std=c++17 -O3 -Wall -Wextra -fvisibility=hidden -fvisibility-inlines-hidden
          -mmacosx-version-min="$MACOS_MIN" -I"$ROOT/src" -I"$ROOT/external/openfx" -I"$GEN")

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "==> Tests"
  clang++ "${CXXFLAGS[@]}" -fobjc-arc -o "$BUILD/test/engine_test" "$ROOT/tests/engine_test.cpp" "$ROOT/src/GrainEngine.cpp"
  "$BUILD/test/engine_test" "$BUILD/test" | tail -3
  clang++ "${CXXFLAGS[@]}" -fobjc-arc -o "$BUILD/test/metal_test" "$ROOT/tests/metal_test.mm" \
    "$ROOT/src/GrainMetal.mm" "$ROOT/src/GrainEngine.cpp" -framework Metal -framework Foundation
  "$BUILD/test/metal_test"
fi

echo "==> Building universal plugin"
rm -rf "$BUILD/stage"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
OBJ="$BUILD/obj"
rm -rf "$OBJ" && mkdir -p "$OBJ"
for arch in arm64 x86_64; do
  clang++ "${CXXFLAGS[@]}" -arch "$arch" -c "$ROOT/src/GrainEngine.cpp" -o "$OBJ/GrainEngine.$arch.o"
  clang++ "${CXXFLAGS[@]}" -arch "$arch" -c "$ROOT/src/SpektraGrainPlugin.cpp" -o "$OBJ/Plugin.$arch.o"
  clang++ "${CXXFLAGS[@]}" -arch "$arch" -fobjc-arc -c "$ROOT/src/GrainMetal.mm" -o "$OBJ/GrainMetal.$arch.o"
  clang++ -arch "$arch" -mmacosx-version-min="$MACOS_MIN" -bundle -o "$OBJ/$NAME.$arch.ofx" \
    "$OBJ/GrainEngine.$arch.o" "$OBJ/Plugin.$arch.o" "$OBJ/GrainMetal.$arch.o" \
    -framework Metal -framework Foundation -Wl,-exported_symbol,_OfxGetPlugin -Wl,-exported_symbol,_OfxGetNumberOfPlugins
done
lipo -create "$OBJ/$NAME.arm64.ofx" "$OBJ/$NAME.x86_64.ofx" -output "$BUNDLE/Contents/MacOS/$NAME.ofx"
sed -e "s/@NAME@/$NAME/g" -e "s/@BUNDLE_ID@/$BUNDLE_ID/g" -e "s/@VERSION@/$VERSION/g" -e "s/@LABEL@/$LABEL/g" \
  "$ROOT/resources/Info.plist.in" > "$BUNDLE/Contents/Info.plist"
mkdir -p "$BUNDLE/Contents/Resources/Legal"
cp "$ROOT/LICENSE" "$BUNDLE/Contents/Resources/Legal/LICENSE.txt"
cp "$ROOT/legal/"*.txt "$BUNDLE/Contents/Resources/Legal/"
if [[ -f "$ROOT/resources/icon.png" ]]; then
  cp "$ROOT/resources/icon.png" "$BUNDLE/Contents/Resources/$BUNDLE_ID.png"
fi

echo "==> Signing bundle"
xattr -cr "$BUNDLE"
if [[ -n "${SIGN_ID:-}" ]]; then
  codesign --force --options runtime --timestamp --sign "$SIGN_ID" "$BUNDLE"
else
  codesign --force --sign - "$BUNDLE"
fi
codesign --verify --strict "$BUNDLE"

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "==> OFX host test"
  clang++ "${CXXFLAGS[@]}" -fobjc-arc -o "$BUILD/test/ofx_host_test" "$ROOT/tests/ofx_host_test.mm" \
    -framework Metal -framework Foundation
  "$BUILD/test/ofx_host_test" "$BUNDLE/Contents/MacOS/$NAME.ofx"
fi

echo "==> Packaging"
PKGROOT="$BUILD/pkgroot"
rm -rf "$PKGROOT" && mkdir -p "$PKGROOT"
ditto --norsrc --noextattr --noacl "$BUNDLE" "$PKGROOT/$NAME.ofx.bundle"
pkgbuild --analyze --root "$PKGROOT" "$BUILD/components.plist" >/dev/null
# Never let Installer "relocate" into some other copy of the bundle on disk.
plutil -replace 0.BundleIsRelocatable -bool NO "$BUILD/components.plist"
plutil -replace 0.BundleOverwriteAction -string upgrade "$BUILD/components.plist"
pkgbuild --root "$PKGROOT" --component-plist "$BUILD/components.plist" \
  --identifier "$BUNDLE_ID.pkg" --version "$VERSION" --install-location /Library/OFX/Plugins \
  "$BUILD/$NAME-component.pkg" >/dev/null

sed -e "s/@NAME@/$NAME/g" -e "s/@BUNDLE_ID@/$BUNDLE_ID/g" -e "s/@VERSION@/$VERSION/g" -e "s/@LABEL@/$LABEL/g" \
  "$ROOT/resources/distribution.xml.in" > "$BUILD/distribution.xml"
mkdir -p "$BUILD/pkgres"
cp "$ROOT/resources/welcome.html" "$BUILD/pkgres/"
cp "$ROOT/LICENSE" "$BUILD/pkgres/LICENSE.txt"
PRODUCT_ARGS=(--distribution "$BUILD/distribution.xml" --package-path "$BUILD" --resources "$BUILD/pkgres")
if [[ -n "${INSTALLER_SIGN_ID:-}" ]]; then
  PRODUCT_ARGS+=(--sign "$INSTALLER_SIGN_ID" --timestamp)
fi
PKG="$DIST/$NAME-$VERSION.pkg"
productbuild "${PRODUCT_ARGS[@]}" "$PKG" >/dev/null

DMGSTAGE="$BUILD/dmg"
rm -rf "$DMGSTAGE" && mkdir -p "$DMGSTAGE"
cp "$PKG" "$DMGSTAGE/Install $LABEL.pkg"
cp "$ROOT/resources/READ ME.txt" "$DMGSTAGE/"
cp "$ROOT/resources/Uninstall.command" "$DMGSTAGE/"
chmod +x "$DMGSTAGE/Uninstall.command"
DMG="$DIST/$NAME-$VERSION.dmg"
rm -f "$DMG"
hdiutil create -volname "$LABEL" -srcfolder "$DMGSTAGE" -ov -format UDZO "$DMG" >/dev/null

echo "==> Done"
ls -lh "$PKG" "$DMG"
