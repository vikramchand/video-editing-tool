#!/usr/bin/env bash
#
# Builds Video Understanding.app.
#
#   ./macapp/build.sh                 build into macapp/build/
#   ./macapp/build.sh --install       also copy it into /Applications
#   ./macapp/build.sh --install --open
#   ./macapp/build.sh --native-only   skip the universal binary
#
# Requires macOS with the Xcode Command Line Tools (`xcode-select --install`).
# Nothing else: no Xcode project, no package manager, no signing identity.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD_DIR="$HERE/build"
APP_NAME="Video Understanding"
APP="$BUILD_DIR/$APP_NAME.app"
BINARY_NAME="VideoUnderstanding"
DEPLOYMENT_TARGET="12.0"

INSTALL=0
OPEN_AFTER=0
NATIVE_ONLY=0
for argument in "$@"; do
    case "$argument" in
        --install) INSTALL=1 ;;
        --open) OPEN_AFTER=1 ;;
        --native-only) NATIVE_ONLY=1 ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

say() { printf '\033[1m▸\033[0m %s\n' "$1"; }
die() { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

# --- Preconditions -----------------------------------------------------------

[ "$(uname -s)" = "Darwin" ] || die "this builds a Mac app, so it needs macOS."
command -v xcrun >/dev/null 2>&1 || die "Xcode Command Line Tools are missing. Run: xcode-select --install"
xcrun --find swiftc >/dev/null 2>&1 || die "swiftc is missing. Run: xcode-select --install"
[ -d "$ROOT/video_understanding" ] || die "run this from a checkout of the project."

SWIFTC="$(xcrun --find swiftc)"
SWIFT="$(xcrun --find swift)"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || VERSION="0.0.0"
if git -C "$ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
    REVISION="$(git -C "$ROOT" rev-parse --short HEAD)"
    git -C "$ROOT" diff --quiet 2>/dev/null || REVISION="$REVISION-dirty"
else
    REVISION="nogit"
fi
BUILD_ID="$VERSION+$REVISION.$(date -u +%Y%m%d%H%M%S)"

say "Building $APP_NAME $VERSION ($REVISION)"

# --- Bundle skeleton ---------------------------------------------------------

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- Stage the Python source -------------------------------------------------
#
# The package ships inside the bundle and is pip-installed into a private
# virtualenv on first launch. Everything the wheel needs, and nothing else.

say "Staging the Python package"
STAGE="$APP/Contents/Resources/app-source"
mkdir -p "$STAGE"
if command -v rsync >/dev/null 2>&1; then
    rsync -a \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        "$ROOT/video_understanding" "$STAGE/"
else
    cp -R "$ROOT/video_understanding" "$STAGE/"
    find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find "$STAGE" \( -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true
fi
cp "$ROOT/pyproject.toml" "$STAGE/pyproject.toml"
cp "$ROOT/README.md" "$STAGE/README.md"
printf '%s\n' "$BUILD_ID" > "$STAGE/BUILD_ID"

# --- Compile -----------------------------------------------------------------

SOURCES=("$HERE"/Sources/*.swift)
[ -f "${SOURCES[0]}" ] || die "no Swift sources found in $HERE/Sources"

compile_slice() {
    local arch="$1" output="$2"
    "$SWIFTC" \
        -O \
        -swift-version 5 \
        -target "${arch}-apple-macos${DEPLOYMENT_TARGET}" \
        -framework AppKit -framework WebKit \
        -o "$output" \
        "${SOURCES[@]}"
}

NATIVE_ARCH="$(uname -m)"
say "Compiling for $NATIVE_ARCH"
mkdir -p "$BUILD_DIR/obj"
compile_slice "$NATIVE_ARCH" "$BUILD_DIR/obj/$BINARY_NAME.$NATIVE_ARCH"

OTHER_ARCH=""
if [ "$NATIVE_ONLY" -eq 0 ]; then
    case "$NATIVE_ARCH" in
        arm64) OTHER_ARCH="x86_64" ;;
        x86_64) OTHER_ARCH="arm64" ;;
    esac
fi

if [ -n "$OTHER_ARCH" ]; then
    say "Compiling for $OTHER_ARCH (universal binary)"
    if compile_slice "$OTHER_ARCH" "$BUILD_DIR/obj/$BINARY_NAME.$OTHER_ARCH" 2>"$BUILD_DIR/obj/$OTHER_ARCH.log"; then
        lipo -create \
            "$BUILD_DIR/obj/$BINARY_NAME.$NATIVE_ARCH" \
            "$BUILD_DIR/obj/$BINARY_NAME.$OTHER_ARCH" \
            -output "$APP/Contents/MacOS/$BINARY_NAME"
    else
        say "  $OTHER_ARCH slice failed — shipping a $NATIVE_ARCH-only build"
        say "  (details: $BUILD_DIR/obj/$OTHER_ARCH.log)"
        cp "$BUILD_DIR/obj/$BINARY_NAME.$NATIVE_ARCH" "$APP/Contents/MacOS/$BINARY_NAME"
    fi
else
    cp "$BUILD_DIR/obj/$BINARY_NAME.$NATIVE_ARCH" "$APP/Contents/MacOS/$BINARY_NAME"
fi
chmod +x "$APP/Contents/MacOS/$BINARY_NAME"

# --- Icon --------------------------------------------------------------------

say "Drawing the icon"
ICONSET="$BUILD_DIR/AppIcon.iconset"
rm -rf "$ICONSET"
if "$SWIFT" "$HERE/scripts/MakeIcon.swift" "$ICONSET" >/dev/null 2>&1 \
    && iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns" 2>/dev/null; then
    :
else
    say "  icon generation failed — continuing with the default icon"
fi

# --- Info.plist --------------------------------------------------------------

sed -e "s/__VERSION__/$VERSION/" -e "s/__BUILD__/$BUILD_ID/" \
    "$HERE/Resources/Info.plist" > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

# --- Sign --------------------------------------------------------------------
#
# An ad-hoc signature is enough for an app you build and run yourself, and it
# keeps macOS from re-prompting about the binary after every rebuild.

say "Signing (ad-hoc)"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || say "  codesign failed — the app will still run"

# --- Install -----------------------------------------------------------------

if [ "$INSTALL" -eq 1 ]; then
    say "Installing to /Applications"
    rm -rf "/Applications/$APP_NAME.app"
    ditto "$APP" "/Applications/$APP_NAME.app"
    FINAL="/Applications/$APP_NAME.app"
else
    FINAL="$APP"
fi

say "Done: $FINAL"
echo
echo "  Double-click it, or run: open \"$FINAL\""
echo "  First launch installs a private Python runtime (a few minutes, needs the network)."
echo

if [ "$OPEN_AFTER" -eq 1 ]; then
    open "$FINAL"
fi
