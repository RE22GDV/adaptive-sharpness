"""Hardware discovery for the capture stage.

Answers, with measurements rather than assumptions, three questions:

1. Does the attached camera expose a UVC / V4L2 video node?
2. Does it expose a PTP live-view stream through libgphoto2?
3. What frame rate and latency does the working path actually deliver?

Run it on the machine the camera is attached to::

    python3 tools/probe_camera.py
    python3 tools/probe_camera.py --frames 60 --json report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger("probe")

# Video nodes belonging to the Raspberry Pi's own ISP / codec blocks.  They are
# not capture devices and must not be mistaken for a camera.
INTERNAL_DRIVERS = {"pispbe", "rpi-hevc-dec", "bcm2835-codec", "bcm2835-isp", "rpivid"}


@dataclass
class UsbDevice:
    bus: str
    device: str
    vendor_id: str
    product_id: str
    description: str

    @property
    def is_camera_vendor(self) -> bool:
        # Panasonic, Canon, Nikon, Sony, Fujifilm.
        return self.vendor_id.lower() in {"04da", "04a9", "04b0", "054c", "04cb"}


@dataclass
class VideoNode:
    path: str
    driver: str = ""
    card: str = ""
    formats: list[str] = field(default_factory=list)
    is_internal: bool = True
    error: str = ""


@dataclass
class ProbeReport:
    usb_devices: list[UsbDevice] = field(default_factory=list)
    camera_usb_devices: list[UsbDevice] = field(default_factory=list)
    video_nodes: list[VideoNode] = field(default_factory=list)
    external_video_nodes: list[VideoNode] = field(default_factory=list)
    tools: dict[str, str] = field(default_factory=dict)
    gphoto_detected: list[str] = field(default_factory=list)
    gphoto_abilities: dict[str, str] = field(default_factory=dict)
    gphoto_preview_supported: bool = False
    liveview: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    """Run a command, returning ``(returncode, combined_output)``."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"LC_ALL": "C", "LANG": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: timed out after {timeout}s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def probe_usb(report: ProbeReport) -> None:
    code, out = _run(["lsusb"])
    if code != 0:
        report.notes.append(f"lsusb unavailable: {out.strip()}")
        return
    pattern = re.compile(
        r"Bus (\d+) Device (\d+): ID ([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\s*(.*)"
    )
    for line in out.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        dev = UsbDevice(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5).strip())
        report.usb_devices.append(dev)
        if dev.is_camera_vendor:
            report.camera_usb_devices.append(dev)

    for dev in report.camera_usb_devices:
        code, out = _run(["lsusb", "-d", f"{dev.vendor_id}:{dev.product_id}", "-v"])
        if code != 0:
            continue
        classes = set(re.findall(r"bInterfaceClass\s+(\d+)", out))
        if "14" in classes:
            report.notes.append(
                f"{dev.description}: exposes a UVC (class 14) interface -> V4L2 path possible"
            )
        else:
            names = ", ".join(sorted(classes)) or "none"
            report.notes.append(
                f"{dev.description}: USB interface classes = {names}; "
                "no UVC (14) interface, so no direct V4L2 video node"
            )


def probe_video_nodes(report: ProbeReport) -> None:
    nodes = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
    has_v4l2 = shutil.which("v4l2-ctl") is not None
    for path in nodes:
        node = VideoNode(path=str(path))
        if not has_v4l2:
            node.error = "v4l2-ctl not installed"
            report.video_nodes.append(node)
            continue
        code, out = _run(["v4l2-ctl", "-d", str(path), "--info"], timeout=8)
        if code != 0:
            node.error = out.strip()[:120]
            report.video_nodes.append(node)
            continue
        driver = re.search(r"Driver name\s*:\s*(\S+)", out)
        card = re.search(r"Card type\s*:\s*(.+)", out)
        node.driver = driver.group(1) if driver else ""
        node.card = card.group(1).strip() if card else ""
        node.is_internal = node.driver in INTERNAL_DRIVERS
        if not node.is_internal:
            code, fmt = _run(["v4l2-ctl", "-d", str(path), "--list-formats-ext"], timeout=8)
            if code == 0:
                node.formats = [
                    line.strip() for line in fmt.splitlines() if "Size:" in line or "]:" in line
                ][:40]
            report.external_video_nodes.append(node)
        report.video_nodes.append(node)


def probe_tools(report: ProbeReport) -> None:
    for tool in ("v4l2-ctl", "gphoto2", "ffmpeg", "gst-launch-1.0"):
        report.tools[tool] = shutil.which(tool) or "MISSING"


def probe_gphoto(report: ProbeReport) -> None:
    if shutil.which("gphoto2") is None:
        report.notes.append("gphoto2 CLI not installed; skipping PTP probe")
        return
    code, out = _run(["gphoto2", "--auto-detect"], timeout=25)
    if code == 0:
        for line in out.splitlines():
            if "usb:" in line:
                report.gphoto_detected.append(line.strip())
    if not report.gphoto_detected:
        report.notes.append("gphoto2 did not detect any camera")
        return

    code, out = _run(["gphoto2", "--abilities"], timeout=25)
    if code == 0:
        capture_block = False
        choices: list[str] = []
        for line in out.splitlines():
            if ":" in line and not line.startswith(" " * 30):
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key:
                    report.gphoto_abilities[key] = value
                capture_block = key.startswith("Capture choices")
                if capture_block and value:
                    choices.append(value)
            elif capture_block:
                choices.append(line.strip())
        report.gphoto_abilities["Capture choices"] = ", ".join(c for c in choices if c)
        report.gphoto_preview_supported = "preview" in " ".join(choices).lower()


def measure_liveview(report: ProbeReport, frames: int, warmup: int) -> None:
    """Measure the sustained frame rate of the PTP live-view stream."""
    if not report.gphoto_preview_supported:
        return
    try:
        import gphoto2 as gp
    except ImportError:
        report.notes.append("python-gphoto2 not installed; cannot measure live-view rate")
        return
    try:
        import cv2
        import numpy as np
    except ImportError:
        report.notes.append("opencv/numpy missing; cannot decode preview frames")
        return

    camera = gp.Camera()
    try:
        camera.init()
    except gp.GPhoto2Error as exc:
        report.liveview = {"error": f"camera.init() failed: {exc}"}
        report.notes.append(
            "Could not claim the camera. A desktop session may be holding it: "
            "stop gvfs with `pkill -f gvfsd-gphoto2` and retry."
        )
        return

    grab_times: list[float] = []
    decode_times: list[float] = []
    shape = None
    sizes: list[int] = []
    error = ""
    try:
        for i in range(frames + warmup):
            t0 = time.perf_counter()
            try:
                capture = camera.capture_preview()
                data = memoryview(capture.get_data_and_size())
            except gp.GPhoto2Error as exc:
                error = f"capture_preview failed on frame {i}: {exc}"
                break
            t1 = time.perf_counter()
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            t2 = time.perf_counter()
            if img is None:
                error = f"frame {i}: JPEG decode failed"
                break
            if i < warmup:
                continue
            grab_times.append(t1 - t0)
            decode_times.append(t2 - t1)
            sizes.append(len(data))
            shape = img.shape
    finally:
        try:
            camera.exit()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass

    if not grab_times:
        report.liveview = {"error": error or "no frames captured"}
        return

    total = [g + d for g, d in zip(grab_times, decode_times)]
    report.liveview = {
        "frames": len(total),
        "resolution": f"{shape[1]}x{shape[0]}" if shape else "?",
        "jpeg_bytes_mean": int(statistics.mean(sizes)),
        "grab_ms_mean": round(statistics.mean(grab_times) * 1e3, 2),
        "grab_ms_p95": round(sorted(grab_times)[int(0.95 * (len(grab_times) - 1))] * 1e3, 2),
        "decode_ms_mean": round(statistics.mean(decode_times) * 1e3, 2),
        "total_ms_mean": round(statistics.mean(total) * 1e3, 2),
        "fps_mean": round(1.0 / statistics.mean(total), 2),
        "fps_min": round(1.0 / max(total), 2),
        "error": error,
    }


def decide(report: ProbeReport) -> None:
    if report.external_video_nodes:
        node = report.external_video_nodes[0]
        report.recommendation = (
            f"v4l2: use {node.path} ({node.card}). Direct UVC capture is available."
        )
        return
    fps = report.liveview.get("fps_mean")
    if report.gphoto_preview_supported and fps:
        report.recommendation = (
            f"gphoto2: PTP live view works at ~{fps} fps "
            f"({report.liveview.get('resolution')}). No UVC node is present, so this "
            "is the only direct USB path for this camera in its current USB mode."
        )
        return
    if report.gphoto_detected:
        report.recommendation = (
            "gphoto2: camera detected over PTP but live view could not be measured. "
            "Check the notes."
        )
        return
    report.recommendation = (
        "none: no usable capture path found. Use the file/synthetic backend, or add an "
        "external USB HDMI capture device."
    )


def format_report(report: ProbeReport) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 72)
    add("CAPTURE PATH PROBE")
    add("=" * 72)

    add("\n[USB devices]")
    for dev in report.usb_devices:
        mark = " <-- camera vendor" if dev.is_camera_vendor else ""
        add(f"  {dev.vendor_id}:{dev.product_id}  {dev.description}{mark}")
    if not report.usb_devices:
        add("  (none)")

    add("\n[V4L2 video nodes]")
    if not report.video_nodes:
        add("  (none)")
    internal = [n for n in report.video_nodes if n.is_internal]
    for node in report.external_video_nodes:
        add(f"  {node.path}: {node.card} (driver {node.driver})  <-- CAPTURE DEVICE")
        for line in node.formats[:12]:
            add(f"      {line}")
    if internal:
        drivers = sorted({n.driver or "?" for n in internal})
        add(
            f"  {len(internal)} internal node(s) belonging to {', '.join(drivers)} "
            "- these are the SoC codec/ISP blocks, not cameras"
        )

    add("\n[Tools]")
    for name, path in report.tools.items():
        add(f"  {name:16s} {path}")

    add("\n[PTP / gphoto2]")
    if report.gphoto_detected:
        for line in report.gphoto_detected:
            add(f"  detected: {line}")
        for key in ("Abilities for camera", "Capture choices", "Configuration support"):
            if key in report.gphoto_abilities:
                add(f"  {key}: {report.gphoto_abilities[key]}")
        add(f"  preview (live view) supported: {report.gphoto_preview_supported}")
    else:
        add("  no camera detected over PTP")

    if report.liveview:
        add("\n[Live-view measurement]")
        for key, value in report.liveview.items():
            if value != "":
                add(f"  {key:18s} {value}")

    if report.notes:
        add("\n[Notes]")
        for note in report.notes:
            add(f"  - {note}")

    add("\n[Recommendation]")
    add(f"  {report.recommendation}")
    add("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30, help="frames to time")
    parser.add_argument("--warmup", type=int, default=3, help="frames to discard first")
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--no-liveview", action="store_true", help="skip the timing run")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    report = ProbeReport()
    probe_tools(report)
    probe_usb(report)
    probe_video_nodes(report)
    probe_gphoto(report)
    if not args.no_liveview:
        measure_liveview(report, args.frames, args.warmup)
    decide(report)

    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
