#!/bin/bash
# Removes the Spektra RMS Grain OFX plugin. Asks for your password.
TARGET="/Library/OFX/Plugins/SpektraRMSGrain.ofx.bundle"
if [[ ! -e "$TARGET" ]]; then
  echo "Spektra RMS Grain is not installed."
  exit 0
fi
echo "This will remove $TARGET"
read -r -p "Continue? [y/N] " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
  sudo rm -rf "$TARGET" && sudo pkgutil --forget com.vividcolour.SpektraRMSGrain.pkg >/dev/null 2>&1
  echo "Removed. Restart DaVinci Resolve."
fi
