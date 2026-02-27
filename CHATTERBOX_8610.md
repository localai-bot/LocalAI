# Issue #8610: Chatterbox Backend on MacOS

## Problem

Users on MacOS M2 (ARM64) get the error:
```
Error installing model "chatterbox": not a valid backend: run file not found "/backends/cpu-chatterbox/run.sh"
```

## Investigation Findings

1. The meta backend "chatterbox" has proper metal capability defined in `backend/index.yaml`:
   - `metal: "metal-chatterbox"`
   - `default: "cpu-chatterbox"`

2. On MacOS ARM64, the capability detection correctly returns "metal" via `pkg/system/capabilities.go`

3. The issue is that the backend images are not being published to the container registry:
   - The Darwin backend build workflow has been failing (runs #1492, #1493)
   - Without the images, the fallback to "default" (cpu-chatterbox) also fails

## Workaround

Users can work around this by setting the environment variable:
```bash
export LOCALAI_FORCE_META_BACKEND_CAPABILITY=metal
local-ai backends install chatterbox
```

This bypasses the meta backend resolution and directly uses metal-chatterbox.

## Related Issues

- Issue #8217: Similar issue for qwen-tts that was previously fixed

## Status

- Meta backend definition: ✅ Correct
- Backend image availability: ❌ Not available (build failure)
- Darwin build workflow: ❌ Failing
