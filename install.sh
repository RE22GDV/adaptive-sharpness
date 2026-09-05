#!/usr/bin/env bash
# Install everything this project needs on Raspberry Pi OS (bookworm) or Debian.
#
#   ./install.sh            core + camera support + dev tools
#   ./install.sh --no-camera   skip libgphoto2 (no PTP camera support)
#   ./install.sh --core        core library only
#
# Raspberry Pi OS marks its Python installation as externally managed (PEP 668),
# so pip needs --break-system-packages for a user install. That is expected here
# and is why this script exists rather than a bare `pip install -r`.
set -euo pipefail

WITH_CAMERA=1
WITH_DEV=1

for arg in "$@"; do
    case "$arg" in
        --no-camera) WITH_CAMERA=0 ;;
        --core)      WITH_CAMERA=0; WITH_DEV=0 ;;
        -h|--help)   sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2; }

# --- Python version -------------------------------------------------------
say "Checking Python"
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    sys.exit(
        f"Python 3.11+ is required (configuration loading uses tomllib); "
        f"found {sys.version.split()[0]}"
    )
print(f"Python {sys.version.split()[0]} OK")
PY

PIP_FLAGS=(--user)
if python3 -c "import sys; sys.exit(0)" 2>/dev/null && \
   [ -f /usr/lib/python3*/EXTERNALLY-MANAGED ] 2>/dev/null; then
    PIP_FLAGS+=(--break-system-packages)
fi

# --- System packages ------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
    say "Installing system packages"
    SYS_PKGS=(python3-numpy python3-opencv v4l-utils)
    if [ "$WITH_CAMERA" = 1 ]; then
        SYS_PKGS+=(gphoto2 libgphoto2-dev python3-gphoto2)
    fi
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${SYS_PKGS[@]}"
else
    warn "apt-get not found; install NumPy, OpenCV and (optionally) libgphoto2 yourself"
fi

# --- Python packages ------------------------------------------------------
say "Installing Python packages"
REQ=requirements.txt
[ "$WITH_DEV" = 1 ] && REQ=requirements-dev.txt
python3 -m pip install "${PIP_FLAGS[@]}" -q -r "$REQ"

if [ "$WITH_CAMERA" = 1 ] && ! python3 -c "import gphoto2" 2>/dev/null; then
    warn "python3-gphoto2 is not importable; PTP live view will be unavailable"
fi

# --- Verify ---------------------------------------------------------------
say "Verifying the installation"
python3 tools/check_env.py

say "Checking the capture path"
python3 tools/probe_camera.py --no-liveview || \
    warn "probe reported a problem; see docs/GH6_SETUP.md"

cat <<'DONE'

==> Done.

Next steps:

  python3 -m pytest tests/ -q                              run the test suite
  python3 tools/probe_camera.py                            check the camera
  python3 demo/focus_ui.py --backend gphoto2               live UI
  python3 demo/focus_ui.py --backend synthetic --headless  no camera needed

If the camera is detected but cannot be claimed, a desktop session is probably
holding it:

  pkill -f gvfsd-gphoto2

DONE
