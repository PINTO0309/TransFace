# TransFace LiteRT Check

This is a verification-only Electron + LiteRT.js + TypeScript + React app for running the three-part TransFace-L `.tflite` feature extraction model from the browser renderer.

## Requirements

- Node.js and Corepack must be available.
- The package manager is `pnpm@11.5.2`.
- Dependency versions are pinned in `package.json` and `pnpm-lock.yaml`.
- `pnpm-workspace.yaml` sets `minimumReleaseAge: 10080`, so packages released less than seven days ago are not selected.

## Model Files

Place the following three files under `web-litert-check/public/models/`.

```text
web-litert-check/public/models/glint360k_model_TransFace_L_0001_float32.tflite
web-litert-check/public/models/glint360k_model_TransFace_L_0002_float32.tflite
web-litert-check/public/models/glint360k_model_TransFace_L_0003_float32.tflite
```

The `.tflite` files are intentionally ignored by git because they are large. Only `public/models/.gitkeep` is tracked.

## First-Time Setup

Run these commands from the repository root.

```bash
cd web-litert-check
COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack pnpm install --frozen-lockfile
```

You can run the same command even when `node_modules/` already exists to verify that the installed dependencies match the lockfile.

## Development

```bash
cd web-litert-check
COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack pnpm dev
```

`pnpm dev` starts the Vite dev server at `127.0.0.1:5173`, then launches Electron after the dev server is ready.

## Build

```bash
cd web-litert-check
COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack pnpm build
```

This command runs:

- TypeScript type checking for the renderer
- Vite renderer build
- TypeScript build for Electron main/preload

Build artifacts are written to:

```text
web-litert-check/dist-renderer/
web-litert-check/dist-electron/
```

## Start the Built App

Run `pnpm build` first, then start the built app.

```bash
cd web-litert-check
COREPACK_ENABLE_DOWNLOAD_PROMPT=0 corepack pnpm start
```

## What the App Shows

- Input mode: dummy / image
- Stage 1/2 execution preference: WebGPU / WASM
- WebGPU adapter information
- Compile time for each stage
- Run time for each stage
- Input/output tensor shapes
- Input/output tensor norm / min / max / leading values
- Final output length / L2 norm / leading values
- Logs for explicit exceptions, `window.onerror`, and `unhandledrejection`
- Warnings for suspicious results, such as an extremely short run time or all-zero output from non-zero input

  <img width="1122" height="973" alt="image" src="https://github.com/user-attachments/assets/ca0f0d02-cae5-4ec2-8aa0-448fac0d9c19" />

## Execution Layout

The model order is fixed.

```text
0001 -> 0002 -> 0003
```

When WebGPU is selected, Stage 1/2 are attempted with the WebGPU delegate, and Stage 3 is executed in a separate non-JSPI WASM runtime. Some environments short-circuit Stage 3 when it is run in the same LiteRT runtime after Stage 1/2, so the app materializes the Stage 1/2 outputs as TypedArrays, unloads LiteRT, reloads the WASM runtime, and then runs Stage 3.

When WASM is selected, Stage 1/2 and Stage 3 are also separated across LiteRT runtime instances.

## Troubleshooting

If `ERR_PNPM_NO_IMPORTER_MANIFEST_FOUND` appears:

- Confirm that `web-litert-check/package.json` exists.
- Confirm that the current working directory is `web-litert-check`.

If the models are shown as Missing:

- Confirm that the `.tflite` files are placed under `web-litert-check/public/models/`.
- Confirm that the filenames match exactly.

If WebGPU is not used, or acceleration is shown as Partial:

- Even when Electron initializes WebGPU successfully, the LiteRT.js delegate may not be able to execute the entire model on the GPU.
- Check `Input Buffers`, `Output Buffers`, and `Accelerated` in the stage table.

If Stage 3 shows a `0.x ms` run time and returns all-zero or otherwise invalid output:

- The inference is likely not valid.
- The current implementation avoids this by running Stage 3 in a separate WASM runtime.
