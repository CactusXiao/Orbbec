# Ego Streaming Reproduction Notes

This project reproduces the original Unity project settings and the client/server streaming capture path from the original project.

## Source Project

- Original project: `E:\Project\Unity\egocollectpj`
- Target project: `C:\A_Project\embody\unity\egopj`
- Rule: the source project is read-only. Do not write files under `E:\Project\Unity\egocollectpj`.

## Reproduced Scope

The retained core feature is PICO client/server streaming capture.

The client entry point is:

- `Assets/Scripts/EgoStreamingClientController.cs`

The PC-side server is:

- `outside/stream_server/server.py`

The original `Assets/Scripts` folder is now copied as a whole. The non-streaming scripts are present because the original `EgoCapture.unity` scene references them, although the active streaming component remains `EgoStreamingClientController`.

## Migrated Files

Unity client/scripts:

- `Assets/Scripts/`
- `Assets/Scripts.meta`

Android HEVC encoder bridge:

- `Assets/Plugins/`
- `Assets/Plugins.meta`

PC streaming server:

- `outside/stream_server/server.py`
- `outside/stream_server/decode_h265_to_jpg.py`
- `outside/stream_server/setup_adb_reverse.ps1`
- `outside/stream_server/README.md`
- `outside/stream_server/use.txt`

Project glue:

- `ProjectSettings/` is copied from the original project.
- `Packages/manifest.json` and `Packages/packages-lock.json` are copied from the original project.
- `Assets/Resources/`, `Assets/XR/`, and `Assets/XRI/` are copied from the original project.
- `Assets/Scenes/` is copied from the original project.
- `user.keystore` is copied from the original project to match the Android signing settings.
- `.gitignore` starts from the original project rules and adds generated streaming decode/Python cache/keystore ignores.

## Dependency Notes

`EgoStreamingClientController.cs` depends on:

- UnityEngine, UnityEngine.UI, UnityEngine.XR
- `System.Net.Sockets` for the TCP transport
- `Unity.XR.PXR` and `Unity.XR.PICO.TOBSupport` from the PICO SDK
- Android `MediaCodec` through `AndroidJavaObject("com.fudanfvl.picoego.PicoHevcEncoder")`

The target project intentionally uses the local PICO SDK path that belongs to this workspace, not the original drive path. This avoids depending on `E:\Project\Unity\...`, because that drive must be returned intact.

```json
"com.unity.xr.picoxr": "file:C:/A_Project/embody/unity/PICO-Unity-Integration-SDK-release_3.0.5/PICO-Unity-Integration-SDK-release_3.0.5"
```

That current SDK path exists on this machine. If another machine reproduces this project, update the package path or install the same PICO Unity Integration SDK at the same path.

The unsafe C# option is required because the direct-buffer HEVC path uses `AndroidJNI.NewDirectByteBuffer` with a native pointer.

## Scene Setup

Open `Assets/Scenes/EgoCapture.unity`.

Expected scene objects:

- `Vst` at scene root.
- `Vst` has the original capture scripts attached.
- `EgoStreamingClientController` is enabled on `Vst`.
- The other non-streaming capture behaviours remain disabled, matching the original scene.
- `Main Camera` and directional light match the original scene.

Current streaming component values match the active streaming component in the source `EgoCapture.unity` scene:

- `serverHost`: `127.0.0.1`
- `serverPort`: `50051`
- `targetFps`: `60`
- `hevcBitrate`: `12000000`
- `hevcIFrameIntervalSeconds`: `1`
- `hevcUseSourceResolution`: enabled
- `hevcInputMode`: `AutoDirectBuffer`

For a lighter throughput test, use:

- `targetFps`: `30`
- `hevcUseSourceResolution`: disabled
- `hevcWidth`: `1280`
- `hevcHeight`: `960`

## Run The PC Server

From the target project root:

```powershell
powershell -ExecutionPolicy Bypass -File outside/stream_server/setup_adb_reverse.ps1 -Port 50051
python outside/stream_server/server.py --host 127.0.0.1 --port 50051 --output-root C:\A_Project\embody\unity\egopj\outside\stream_server\sessions
```

The client connects to `127.0.0.1:50051`. With ADB reverse active, the headset-side loopback connection is forwarded to the PC server.

In the server console:

```text
start test_001
status
stop
quit
```

## Output Files

Each streaming session is written under `outside/stream_server/sessions/<session_name>/` and should contain:

- `video.h265`
- `metadata.csv`
- `timestamps.csv`
- `camera.json`
- `network_log.jsonl`
- `session.json`

These generated session folders are ignored by Git.

## Decode Check

If FFmpeg is installed:

```powershell
ffplay -f hevc outside/stream_server/sessions/<session_name>/video.h265
```

Or use the helper:

```powershell
python outside/stream_server/decode_h265_to_jpg.py --session-dir outside/stream_server/sessions/<session_name>
```

The helper can use `imageio_ffmpeg` when system FFmpeg is not on `PATH`. Use an existing Python/Conda environment with OpenCV, NumPy, and `imageio_ffmpeg` installed for the PC-side tools.

## Verification Checklist

1. Unity opens `C:\A_Project\embody\unity\egopj` without package resolution errors.
2. C# scripts compile without errors.
3. `Assets/Scripts/EgoStreamingClientController.cs` resolves PICO SDK types.
4. Android build includes `Assets/Plugins/Android/com/fudanfvl/picoego/PicoHevcEncoder.java`.
5. `EgoCapture.unity` is in `EditorBuildSettings.asset` and contains enabled `EgoStreamingClientController`.
6. On device, the app connects to the PC server after `adb reverse` is active.
7. The server accepts `start`, receives HEVC/metadata packets, and writes a complete session folder after `stop`.

## Verification Performed

On 2026-06-25, Codex ran:

```powershell
C:\App_install\Unity\Editor\6000.3.15f1\Editor\Unity.exe -batchmode -nographics -quit -projectPath C:\A_Project\embody\unity\egopj -logFile C:\A_Project\embody\unity\egopj\Logs\codex-unity-verify.log
```

Result:

- Unity package resolution completed and loaded the local PICO package.
- `Assembly-CSharp.dll` compiled successfully.
- No `error CS` or script compiler error was present in the verification log.

Codex also ran:

```powershell
C:\Users\bot\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile outside\stream_server\server.py outside\stream_server\decode_h265_to_jpg.py
C:\Users\bot\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe outside\stream_server\server.py --help
C:\Users\bot\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe outside\stream_server\decode_h265_to_jpg.py --help
```

Result:

- PC-side Python scripts passed syntax compilation.
- Server and decode helper command-line entry points are runnable.

Codex also ran a local protocol smoke test with a fake TCP client that used the same packet framing constants as the Unity client. The server accepted `HELLO`, `CAMERA_JSON`, metadata/timestamp headers and rows, one `HEVC_SAMPLE`, and `SESSION_END`, then produced a valid `session.json` with one HEVC sample and one metadata/timestamp row.

The Android Java encoder bridge was checked with Unity's bundled JDK and Android SDK:

```powershell
C:\App_install\Unity\Editor\6000.3.15f1\Editor\Data\PlaybackEngines\AndroidPlayer\OpenJDK\bin\javac.exe -cp C:\App_install\Unity\Editor\6000.3.15f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platforms\android-36\android.jar -d Temp\codex-java-verify Assets\Plugins\Android\com\fudanfvl\picoego\PicoHevcEncoder.java
```

Result:

- `PicoHevcEncoder.java` compiled successfully against Android API 36.
- The compiler emitted only deprecation notes, not errors.

## Settings Parity Correction

After an initial minimal migration, Codex rechecked the project against the original and found that several Unity project-level settings were still from the new template. The correction pass copied these original directories/files byte-for-byte into the target project:

- `ProjectSettings/`
- `Packages/` was copied from the original, then the PICO SDK dependency was intentionally retargeted from `E:/Project/Unity/...` to `C:/A_Project/embody/unity/...`.
- `Assets/Resources/`
- `Assets/XR/`
- `Assets/XRI/`
- `Assets/Scenes/`
- `Assets/Scripts/`
- `Assets/Plugins/`
- `Assets/InputSystem_Actions.inputactions`
- `Assets/InputSystem_Actions.inputactions.meta`
- `user.keystore`

The parity check reported:

```text
ProjectSettings: missing=0 extra=0 changed=0
Packages: intentionally different only for the PICO SDK local file path
Resources: missing=0 extra=0 changed=0
XR: missing=0 extra=0 changed=0
XRI: missing=0 extra=0 changed=0
Scenes: missing=0 extra=0 changed=0
Scripts: missing=0 extra=0 changed=0
Plugins: missing=0 extra=0 changed=0
```

After the parity correction, Codex reran Unity:

```powershell
C:\App_install\Unity\Editor\6000.3.15f1\Editor\Unity.exe -batchmode -nographics -quit -projectPath C:\A_Project\embody\unity\egopj -logFile C:\A_Project\embody\unity\egopj\Logs\codex-unity-verify-settings-parity.log
```

Result:

- Unity refreshed the copied original assets and settings.
- The log shows the active PICO SDK path under `C:/A_Project/embody/unity/PICO-Unity-Integration-SDK-release_3.0.5/...`.
- `Assembly-CSharp.dll` compiled successfully with `Tundra build success`.
- The log contains PICO SDK warnings only; no `error CS`, script compiler failure, or Unity process failure was found.

After confirming the original external drive must be returned intact, Codex retargeted the PICO SDK dependency to the workspace-local SDK path and reran Unity:

```powershell
C:\App_install\Unity\Editor\6000.3.15f1\Editor\Unity.exe -batchmode -nographics -quit -projectPath C:\A_Project\embody\unity\egopj -logFile C:\A_Project\embody\unity\egopj\Logs\codex-unity-verify-current-pico.log
```

Result:

- `Packages/manifest.json`, `Packages/packages-lock.json`, and `Library/PackageManager/projectResolution.json` resolve `com.unity.xr.picoxr` to `C:/A_Project/embody/unity/PICO-Unity-Integration-SDK-release_3.0.5/...`.
- `Assembly-CSharp.dll` compiled successfully with `Tundra build success`.
- The log contains PICO SDK warnings only; no `error CS`, script compiler failure, or Unity process failure was found.

Known intentionally different files after this correction:

- `.gitignore`: based on the original `.gitignore`, with additional ignores for generated decoded frames, generated MP4 files, Python cache files, virtual environments, and keystores.
- `Packages/manifest.json` and `Packages/packages-lock.json`: same as the original except `com.unity.xr.picoxr` points to the current workspace SDK path instead of `E:/Project/Unity/...`.
- `REPRODUCTION.md`: target-only reproduction log.
- `outside/stream_server/`: copied server tooling plus target-local documentation updates.

## Current Known Limits

- End-to-end device validation requires a PICO headset, USB ADB access, and Unity Editor/Android build support.
- If direct-buffer input fails on a device, `AutoDirectBuffer` falls back to Java byte-array input and records the path in session metadata.
