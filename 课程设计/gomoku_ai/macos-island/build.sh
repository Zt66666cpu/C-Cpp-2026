#!/bin/zsh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${ZSH_VERSION:-}" ]; then
  exec /bin/zsh "$DIR/build.sh" "$@"
fi

INSTALL=0
for arg in "$@"; do
  if [[ "$arg" == "--install" ]]; then
    INSTALL=1
  fi
done

APP="$DIR/GomokuIsland.app"
rm -rf "$APP"

if [[ -z "${SDKROOT:-}" ]] \
  && [[ "$(xcode-select -p 2>/dev/null)" == "/Library/Developer/CommandLineTools" ]] \
  && [[ ! -e "/Library/Developer/CommandLineTools/usr/lib/swift/host/libSwiftUIMacros.dylib" ]] \
  && [[ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk" ]]; then
  export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk"
  echo "CLT-only toolchain detected (no libSwiftUIMacros.dylib); using SDKROOT=$SDKROOT"
fi

echo "Building GomokuIsland via SwiftPM (release)..."
(cd "$DIR" && swift build -c release --scratch-path "$DIR/.build")

BIN="$DIR/.build/release/GomokuIsland"
if [[ ! -f "$BIN" ]]; then
  echo "Error: compiled binary not found at $BIN" >&2
  exit 1
fi

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/ui"
cp "$BIN" "$APP/Contents/MacOS/GomokuIsland"
cp "$DIR/Info.plist" "$APP/Contents/Info.plist"
cp -R "$DIR/ui/." "$APP/Contents/Resources/ui/"

for bundle in "$DIR"/.build/release/*.bundle(N); do
  cp -R "$bundle" "$APP/Contents/Resources/"
done

codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
echo "Built successfully: $APP"

if [[ "$INSTALL" -eq 1 ]]; then
  rm -rf "/Applications/GomokuIsland.app"
  cp -R "$APP" "/Applications/GomokuIsland.app"
  codesign --force --deep --sign - "/Applications/GomokuIsland.app" >/dev/null 2>&1 || true
  echo "Installed: /Applications/GomokuIsland.app"
fi
