# Panasonic Lumix GH6 over USB-C: what works and what does not

## Summary

The GH6 in its tethering USB mode presents a **PTP-only** interface. There is no
UVC video node, so `/dev/video*` capture is not possible; live view comes
through libgphoto2 instead, at **640x360 and 25 fps**.

This was determined by inspection of the device, not assumed.

## What the device actually reports

With the camera switched on, connected by USB-C, and set to the tethering USB
mode:

```console
$ lsusb
Bus 004 Device 004: ID 04da:2382 Panasonic (Matsushita) DC-GH6
```

```console
$ lsusb -d 04da:2382 -v | grep -E "bcdUSB|bNumConfigurations|bConfigurationValue|bInterfaceClass|bInterfaceSubClass|bInterfaceProtocol"
  bcdUSB               3.20
  bNumConfigurations      2
    bConfigurationValue     1
      bInterfaceClass         6 Imaging
      bInterfaceSubClass      1 Still Image Capture
      bInterfaceProtocol      1 Picture Transfer Protocol (PIMA 15470)
    bConfigurationValue     2
      bInterfaceClass         6 Imaging
      bInterfaceSubClass      1 Still Image Capture
      bInterfaceProtocol      1 Picture Transfer Protocol (PIMA 15470)
```

Both USB configurations are PTP. **Neither exposes interface class 14 (Video),**
which is what a UVC webcam would present. Consequently no video node appears:

```console
$ ls /dev/video*
/dev/video19  /dev/video20  ...  /dev/video35
```

All 17 of those belong to the Raspberry Pi 5's own hardware:

```console
$ v4l2-ctl -d /dev/video20 --info | grep "Driver name"
	Driver name      : pispbe
$ v4l2-ctl -d /dev/video19 --info | grep "Driver name"
	Driver name      : rpi-hevc-dec
```

`pispbe` is the PiSP back end (the image signal processor) and `rpi-hevc-dec` is
the video decoder. They are not capture devices, and mistaking them for one is
the classic way to "successfully open" a device that can never deliver a frame.
`tools/probe_camera.py` classifies them by driver name for exactly this reason.

## The working path: PTP live view

libgphoto2 detects the camera (it matches the shared Panasonic driver entry, so
it reports the model as `DC-GH5`) and reports that preview capture is available:

```console
$ gphoto2 --auto-detect
Model                          Port
----------------------------------------
Panasonic DC-GH5               usb:004,004

$ gphoto2 --abilities
Abilities for camera             : Panasonic DC-GH5
USB support                      : yes
Capture choices                  :
                                 : Image
                                 : Preview
Configuration support            : yes
```

Measured throughput of that path, 40 frames after warm-up:

| Quantity | Value |
| --- | --- |
| Resolution | 640x360 |
| JPEG size | ~28 kB per frame |
| Grab time | 38.6 ms mean, 38.7 ms p95 |
| Decode time | 1.4 ms |
| Sustained rate | 25.0 fps |

The grab time is almost entirely waiting on the camera - it is pacing the stream
at 25 fps - so it is not something the software can improve.

## Setting up

### On the camera

1. Charge or power the camera; live view over USB draws steadily.
2. Set the USB mode to the **tethering** option (`PC(Tether)`), not mass storage
   and not the USB power option. On the GH6 this is in the setup menu under the
   USB entry.
3. Turn the camera on and leave it on - it must not enter sleep. Disable the
   auto power-off / sleep timer for a long session.
4. Put the lens in manual focus if an external actuator drives the focus ring.

The exact menu wording varies between firmware versions; check the model's
manual for the current path. Confirm the mode is right by running
`tools/probe_camera.py` - if the camera is in mass-storage mode it appears as a
block device rather than under `gphoto2 --auto-detect`.

### On the Raspberry Pi

```bash
sudo apt install gphoto2 libgphoto2-dev python3-gphoto2 v4l-utils
```

Verify end to end:

```bash
python3 tools/probe_camera.py --frames 40
```

A working setup ends with:

```
[Recommendation]
  gphoto2: PTP live view works at ~25.0 fps (640x360). No UVC node is present,
  so this is the only direct USB path for this camera in its current USB mode.
```

## Troubleshooting

**`could not claim the camera over PTP`.** Another process holds it. On a
Raspberry Pi OS desktop the usual culprit is the gvfs PTP volume monitor, which
grabs any camera the moment it is plugged in:

```bash
pkill -f gvfsd-gphoto2
pkill -f gvfs-gphoto2-volume-monitor
```

To stop it permanently, mask the service or remove `gvfs-backends`.

**The camera is not detected at all.** Check `lsusb` for `04da:`. If it is
absent, the camera is off, asleep, or the cable is charge-only - a USB-C cable
without data lines is a common cause. If it is present but `gphoto2
--auto-detect` finds nothing, the USB mode is wrong.

**Detected, but `--capture-preview` fails.** The lens may be retracted or the
camera may be in a mode where live view is unavailable. Check that live view
works on the camera's own screen first.

**Frames stop after a while.** The camera went to sleep. Disable the power-save
timer.

**`gphoto2` output is unreadable.** Its messages are localised; prefix commands
with `LC_ALL=C` for English output. The Python tools already do this.

## If a higher resolution is required

The 640x360 live-view stream is the camera's own; libgphoto2 cannot enlarge it.
If the application needs more resolution or a lower latency than 40 ms, the
options are:

1. **HDMI capture.** The GH6 has a full-size HDMI output that can deliver clean
   1080p or higher. Feeding that through a USB HDMI capture stick makes it a UVC
   device, so `capture.backend = "v4l2"` then works and `tools/probe_camera.py`
   will list the node. This is additional hardware and is not currently
   installed on this machine.
2. **A different USB mode**, if a future firmware adds a UVC webcam mode. The
   probe tool checks for interface class 14 on every camera-vendor device, so it
   will report such a mode automatically if one appears.

Both paths are already supported by the capture abstraction; only the backend
selection changes.
