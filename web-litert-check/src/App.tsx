import {
  type Accelerator,
  type CompiledModel,
  type TensorDetails,
  type TypedArray,
  Tensor,
  TensorBufferType,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  setWebGpuDevice,
  unloadLiteRt,
} from '@litertjs/core';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';

import './styles.css';

const TRANSFACE_L_MODEL_FILES = [
  'glint360k_model_TransFace_L_0001_float32.tflite',
  'glint360k_model_TransFace_L_0002_float32.tflite',
  'glint360k_model_TransFace_L_0003_float32.tflite',
] as const;

const TRANSFACE_S_MODEL_FILES = ['glint360k_model_TransFace_S_float32.tflite'] as const;

const MODEL_SETS = {
  'transface-l': {
    label: 'TransFace-L',
    files: TRANSFACE_L_MODEL_FILES,
    runtime: 'split',
  },
  'transface-s': {
    label: 'TransFace-S',
    files: TRANSFACE_S_MODEL_FILES,
    runtime: 'single',
  },
} as const;

const WEBGPU_COMPILE_DISABLED_MODELS = new Set<string>([
  'glint360k_model_TransFace_L_0003_float32.tflite',
]);

const LITERT_WASM_ASSET_ROOT = 'transface-wasm://assets/';

type InputMode = 'dummy' | 'image';
type Backend = 'webgpu' | 'wasm';
type ModelSetId = keyof typeof MODEL_SETS;
type StagePreference = 'webgpu-stage12' | 'wasm-stage12';
type TensorValues = TypedArray;

interface StageTiming {
  model: string;
  stageBackend: Backend;
  compileNote: string;
  warnings: string[];
  modelBytes: number;
  compileMs: number;
  runMs: number;
  inputShapes: number[][];
  outputShapes: number[][];
  inputNames: string[];
  outputNames: string[];
  inputBufferTypes: string[][];
  outputBufferTypes: string[][];
  inputStats: TensorStats[];
  outputStats: TensorStats[];
  fullyAccelerated: boolean;
}

interface TensorStats {
  length: number;
  norm: number;
  min: number;
  max: number;
  preview: number[];
}

interface MaterializedTensor {
  values: TensorValues;
  shape: number[];
}

interface RunResult {
  backend: Backend;
  fallbackUsed: boolean;
  totalMs: number;
  stages: StageTiming[];
  outputLength: number;
  outputNorm: number;
  outputPreview: number[];
  warnings: string[];
  webGpuInitInfo?: WebGpuInitInfo;
}

interface WebGpuInitInfo {
  adapterInfo: Record<string, unknown>;
  features: string[];
  limits: Record<string, number>;
}

interface LogEntry {
  time: string;
  level: 'info' | 'warn' | 'error';
  message: string;
}

const modelBytesCache = new Map<string, Promise<Uint8Array>>();

function nowLabel(): string {
  return new Date().toLocaleTimeString('ja-JP', { hour12: false });
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}${error.stack ? `\n${error.stack}` : ''}`;
  }
  return String(error);
}

function product(values: readonly number[]): number {
  return values.reduce((acc, value) => acc * value, 1);
}

function tensorShape(details: TensorDetails): number[] {
  return Array.from(details.shape, (dim) => Number(dim));
}

function tensorBufferTypeName(value: number): string {
  switch (value) {
    case TensorBufferType.HOST_MEMORY:
      return 'HOST_MEMORY';
    case TensorBufferType.WEB_GPU_BUFFER:
      return 'WEB_GPU_BUFFER';
    case TensorBufferType.WEB_GPU_BUFFER_FP16:
      return 'WEB_GPU_BUFFER_FP16';
    case TensorBufferType.WEB_GPU_BUFFER_PACKED:
      return 'WEB_GPU_BUFFER_PACKED';
    default:
      return `UNKNOWN_${value}`;
  }
}

function tensorBufferTypes(details: TensorDetails): string[] {
  return Array.from(details.supportedBufferTypes, tensorBufferTypeName);
}

function isImageShape(shape: readonly number[]): boolean {
  return (
    shape.length === 4 &&
    ((shape[1] === 3 && shape[2] === 112 && shape[3] === 112) ||
      (shape[1] === 112 && shape[2] === 112 && shape[3] === 3))
  );
}

function makeDummyInput(shape: readonly number[]): Float32Array {
  return new Float32Array(product(shape));
}

function tensorElementCount(tensor: Tensor): number {
  return product(Array.from(tensor.type.layout.dimensions, (dim) => Number(dim)));
}

function deleteTensors(tensors: readonly Tensor[]): void {
  for (const tensor of tensors) {
    if (!tensor.deleted) {
      tensor.delete();
    }
  }
}

async function fetchModelBytes(filename: string): Promise<Uint8Array> {
  const cached = modelBytesCache.get(filename);
  if (cached) {
    return cached;
  }

  const promise = (async () => {
    const response = await fetch(`models/${filename}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filename}: ${response.status} ${response.statusText}`);
    }

    const bytes = new Uint8Array(await response.arrayBuffer());
    const identifier = new TextDecoder('ascii').decode(bytes.slice(4, 8));
    if (identifier !== 'TFL3') {
      throw new Error(
        `Invalid TFLite identifier for ${filename}: expected TFL3 at bytes 4-7, got ${JSON.stringify(identifier)}; byteLength=${bytes.byteLength}`,
      );
    }
    return bytes;
  })();

  modelBytesCache.set(filename, promise);
  return promise;
}

async function imageFileToInput(file: File, shape: readonly number[]): Promise<Float32Array> {
  if (!isImageShape(shape)) {
    throw new Error(`Image input requires [1,3,112,112] or [1,112,112,3], got [${shape.join(', ')}]`);
  }

  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(112, 112);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    bitmap.close();
    throw new Error('Failed to create canvas context for image preprocessing.');
  }

  context.drawImage(bitmap, 0, 0, 112, 112);
  bitmap.close();

  const pixels = context.getImageData(0, 0, 112, 112).data;
  const input = new Float32Array(product(shape));
  const nchw = shape[1] === 3;

  for (let y = 0; y < 112; y += 1) {
    for (let x = 0; x < 112; x += 1) {
      const pixelOffset = (y * 112 + x) * 4;
      const normalized = [
        (pixels[pixelOffset] / 255 - 0.5) / 0.5,
        (pixels[pixelOffset + 1] / 255 - 0.5) / 0.5,
        (pixels[pixelOffset + 2] / 255 - 0.5) / 0.5,
      ];

      if (nchw) {
        const planeOffset = y * 112 + x;
        input[planeOffset] = normalized[0];
        input[112 * 112 + planeOffset] = normalized[1];
        input[2 * 112 * 112 + planeOffset] = normalized[2];
      } else {
        const offset = (y * 112 + x) * 3;
        input[offset] = normalized[0];
        input[offset + 1] = normalized[1];
        input[offset + 2] = normalized[2];
      }
    }
  }

  return input;
}

function l2Norm(values: Float32Array | Int32Array | Uint8Array): number {
  let sum = 0;
  for (const value of values) {
    sum += Number(value) * Number(value);
  }
  return Math.sqrt(sum);
}

function typedArrayStats(values: Float32Array | Int32Array | Uint8Array): TensorStats {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    const numericValue = Number(value);
    min = Math.min(min, numericValue);
    max = Math.max(max, numericValue);
  }
  return {
    length: values.length,
    norm: l2Norm(values),
    min,
    max,
    preview: Array.from(values.slice(0, 6), (value) => Number(value)),
  };
}

async function tensorStats(tensor: Tensor): Promise<TensorStats> {
  const hostTensor = tensor.accelerator === 'wasm' ? tensor : await tensor.copyTo('wasm');
  try {
    return typedArrayStats(hostTensor.toTypedArray());
  } finally {
    if (hostTensor !== tensor && !hostTensor.deleted) {
      hostTensor.delete();
    }
  }
}

function copyTensorValues(values: TensorValues): TensorValues {
  if (values instanceof Float32Array) {
    return new Float32Array(values);
  }
  if (values instanceof Int32Array) {
    return new Int32Array(values);
  }
  return new Uint8Array(values);
}

async function materializeTensors(tensors: readonly Tensor[]): Promise<MaterializedTensor[]> {
  return Promise.all(
    tensors.map(async (tensor) => {
      const hostTensor = tensor.accelerator === 'wasm' ? tensor : await tensor.copyTo('wasm');
      try {
        return {
          values: copyTensorValues(hostTensor.toTypedArray()),
          shape: Array.from(hostTensor.type.layout.dimensions, (dim) => Number(dim)),
        };
      } finally {
        if (hostTensor !== tensor && !hostTensor.deleted) {
          hostTensor.delete();
        }
      }
    }),
  );
}

function readAdapterInfo(adapter: GPUAdapter): Record<string, unknown> {
  const maybeInfo = (adapter as GPUAdapter & { info?: unknown }).info;
  if (maybeInfo && typeof maybeInfo === 'object') {
    const info = maybeInfo as {
      vendor?: unknown;
      architecture?: unknown;
      device?: unknown;
      description?: unknown;
    };
    return Object.fromEntries(
      Object.entries({
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      }).filter(([, value]) => value !== undefined && value !== ''),
    );
  }
  return {};
}

async function requestWebGpuDevice(): Promise<{ device: GPUDevice; info: WebGpuInitInfo }> {
  if (!navigator.gpu) {
    throw new Error('navigator.gpu is not available in this Electron renderer.');
  }

  const adapterDescriptors: GPURequestAdapterOptions[] = [
    {},
    { powerPreference: 'high-performance' },
    { powerPreference: 'low-power' },
  ];
  let adapter: GPUAdapter | null = null;
  let adapterDescriptorUsed = 'default';

  for (const descriptor of adapterDescriptors) {
    adapter = await navigator.gpu.requestAdapter(descriptor);
    if (adapter) {
      adapterDescriptorUsed = descriptor.powerPreference ?? 'default';
      break;
    }
  }

  if (!adapter) {
    throw new Error('navigator.gpu.requestAdapter() returned null for default, high-performance, and low-power adapters.');
  }

  const desiredFeatures = ['shader-f16', 'subgroups'] as GPUFeatureName[];
  const requiredFeatures = desiredFeatures.filter((feature) => adapter.features.has(feature));
  const requiredLimits = {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
  };

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  });

  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.reason, info.message);
  });

  return {
    device,
    info: {
      adapterInfo: {
        ...readAdapterInfo(adapter),
        requestedAdapter: adapterDescriptorUsed,
      },
      features: requiredFeatures,
      limits: requiredLimits,
    },
  };
}

async function initializeWebGpuForLiteRt(): Promise<WebGpuInitInfo> {
  const { device, info } = await requestWebGpuDevice();
  setWebGpuDevice(device);
  return info;
}

async function loadLiteRtForBackend(backend: Backend): Promise<void> {
  unloadLiteRt();
  await loadLiteRt(LITERT_WASM_ASSET_ROOT, backend === 'webgpu' ? { threads: false, jspi: true } : { threads: false });
}

async function compileModelForStage(
  filename: string,
  requestedBackend: Backend,
  compileNote = '',
): Promise<{ model: CompiledModel; timing: StageTiming }> {
  const modelBytes = await fetchModelBytes(filename);
  const compileStart = performance.now();
  const webGpuCompileDisabled = requestedBackend === 'webgpu' && WEBGPU_COMPILE_DISABLED_MODELS.has(filename);
  const stageBackend: Backend = webGpuCompileDisabled ? 'wasm' : requestedBackend;
  const accelerator: Accelerator | Accelerator[] = stageBackend === 'webgpu' ? ['webgpu', 'wasm'] : 'wasm';
  let model: CompiledModel;
  try {
    model = await loadAndCompile(modelBytes, {
      accelerator,
      gpuOptions: { precision: 'fp32' },
    });
  } catch (error) {
    throw new Error(`${filename} compile failed on ${stageBackend}: ${formatError(error)}`);
  }
  const inputDetails = model.getInputDetails();
  const outputDetails = model.getOutputDetails();
  if (inputDetails.length === 0 || outputDetails.length === 0) {
    model.delete();
    throw new Error(`${filename} must have at least 1 input and 1 output.`);
  }

  return {
    model,
    timing: {
      model: filename,
      stageBackend,
      compileNote:
        compileNote ||
        (webGpuCompileDisabled ? 'WebGPU compile disabled for this stage to avoid LiteRT.js delegate abort.' : ''),
      warnings: [],
      modelBytes: modelBytes.byteLength,
      compileMs: performance.now() - compileStart,
      runMs: 0,
      inputShapes: inputDetails.map(tensorShape),
      outputShapes: outputDetails.map(tensorShape),
      inputNames: inputDetails.map((details) => details.name),
      outputNames: outputDetails.map((details) => details.name),
      inputBufferTypes: inputDetails.map(tensorBufferTypes),
      outputBufferTypes: outputDetails.map(tensorBufferTypes),
      inputStats: [],
      outputStats: [],
      fullyAccelerated: model.isFullyAccelerated,
    },
  };
}

async function compileModels(
  modelFiles: readonly string[],
  backend: Backend,
): Promise<{ models: CompiledModel[]; timings: StageTiming[] }> {
  const models: CompiledModel[] = [];
  const timings: StageTiming[] = [];

  for (const filename of modelFiles) {
    const { model, timing } = await compileModelForStage(filename, backend);
    models.push(model);
    timings.push(timing);
  }

  return { models, timings };
}

async function runStage(
  model: CompiledModel,
  timing: StageTiming,
  inputs: Tensor[],
  warnings: string[],
): Promise<Tensor[]> {
  if (inputs.length === 0) {
    throw new Error(`Missing input tensor for ${timing.model}.`);
  }
  const expectedInputCount = timing.inputShapes.length;
  if (inputs.length !== expectedInputCount) {
    throw new Error(`${timing.model} expected ${expectedInputCount} input tensor(s), got ${inputs.length} from previous stage.`);
  }
  timing.inputStats = await Promise.all(inputs.map(tensorStats));

  let outputs: Tensor[];
  const runStart = performance.now();
  try {
    outputs = (await model.run(inputs)) as Tensor[];
  } catch (error) {
    throw new Error(`${timing.model} run failed on ${timing.stageBackend}: ${formatError(error)}`);
  } finally {
    timing.runMs = performance.now() - runStart;
  }

  if (outputs.length === 0) {
    throw new Error(`${timing.model} returned no output tensors.`);
  }
  timing.outputStats = await Promise.all(outputs.map(tensorStats));
  const inputHasSignal = timing.inputStats.some((stat) => stat.norm > 1e-6);
  const outputAllZero = timing.outputStats.every((stat) => stat.norm <= 1e-9 && stat.min === 0 && stat.max === 0);
  if (inputHasSignal && outputAllZero) {
    const warning = `${timing.model} returned all-zero output despite non-zero input.`;
    timing.warnings.push(warning);
    warnings.push(warning);
  }
  if (timing.modelBytes > 100 * 1024 * 1024 && timing.runMs < 1) {
    const warning = `${timing.model} run time is ${formatMs(timing.runMs)} for a ${formatBytes(timing.modelBytes)} model; execution may have been skipped or short-circuited.`;
    timing.warnings.push(warning);
    warnings.push(warning);
  }
  return outputs;
}

async function runPipeline(
  modelFiles: readonly string[],
  backend: Backend,
  inputMode: InputMode,
  imageFile: File | null,
  onWebGpuInitialized?: (info: WebGpuInitInfo) => void,
): Promise<Omit<RunResult, 'fallbackUsed'>> {
  await loadLiteRtForBackend(backend);
  let webGpuInitInfo: WebGpuInitInfo | undefined;
  if (backend === 'webgpu') {
    webGpuInitInfo = await initializeWebGpuForLiteRt();
    onWebGpuInitialized?.(webGpuInitInfo);
  }

  const { models, timings } = await compileModels(modelFiles, backend);
  const warnings: string[] = [];
  let currentTensors: Tensor[] = [];
  let finalTensor: Tensor | null = null;

  try {
    const allInputShapes = timings.flatMap((stage) => stage.inputShapes);
    if (allInputShapes.some((shape) => shape.some((dim) => dim <= 0))) {
      throw new Error('Dynamic or invalid input shapes are not supported by this check app.');
    }
    if (inputMode === 'image' && !imageFile) {
      throw new Error('Image input mode requires an image file.');
    }

    const firstInputShapes = timings[0].inputShapes;
    currentTensors = await Promise.all(
      firstInputShapes.map(async (shape) => {
        const inputValues =
          inputMode === 'image' && imageFile && isImageShape(shape)
            ? await imageFileToInput(imageFile, shape)
            : makeDummyInput(shape);
        return new Tensor(inputValues, shape);
      }),
    );

    for (let index = 0; index < models.length; index += 1) {
      const model = models[index];
      const outputs = await runStage(model, timings[index], currentTensors, warnings);
      deleteTensors(currentTensors);
      currentTensors = outputs;
    }

    const totalMs = timings.reduce((sum, stage) => sum + stage.runMs, 0);
    const selectedOutput = currentTensors.reduce((best, tensor) =>
      tensorElementCount(tensor) > tensorElementCount(best) ? tensor : best,
    );
    finalTensor = await selectedOutput.moveTo('wasm');
    const outputValues = finalTensor.toTypedArray();
    const outputPreview = Array.from(outputValues.slice(0, 12), (value) => Number(value));

    return {
      backend,
      totalMs,
      stages: timings,
      outputLength: outputValues.length,
      outputNorm: l2Norm(outputValues),
      outputPreview,
      warnings,
      webGpuInitInfo,
    };
  } finally {
    deleteTensors(currentTensors);
    if (finalTensor && !finalTensor.deleted) {
      finalTensor.delete();
    }
    for (const model of models) {
      model.delete();
    }
  }
}

async function runSplitRuntimePipeline(
  modelFiles: typeof TRANSFACE_L_MODEL_FILES,
  stage12Backend: Backend,
  inputMode: InputMode,
  imageFile: File | null,
  onWebGpuInitialized?: (info: WebGpuInitInfo) => void,
): Promise<Omit<RunResult, 'fallbackUsed'>> {
  const timings: StageTiming[] = [];
  const warnings: string[] = [];
  let webGpuInitInfo: WebGpuInitInfo | undefined;
  let currentTensors: Tensor[] = [];
  let materializedStageOutputs: MaterializedTensor[] = [];
  let finalTensor: Tensor | null = null;

  await loadLiteRtForBackend(stage12Backend);
  if (stage12Backend === 'webgpu') {
    webGpuInitInfo = await initializeWebGpuForLiteRt();
    onWebGpuInitialized?.(webGpuInitInfo);
  }

  const stage12Models: CompiledModel[] = [];
  try {
    for (const filename of modelFiles.slice(0, 2)) {
      const { model, timing } = await compileModelForStage(filename, stage12Backend);
      stage12Models.push(model);
      timings.push(timing);
    }

    const allInputShapes = timings.flatMap((stage) => stage.inputShapes);
    if (allInputShapes.some((shape) => shape.some((dim) => dim <= 0))) {
      throw new Error('Dynamic or invalid input shapes are not supported by this check app.');
    }
    if (inputMode === 'image' && !imageFile) {
      throw new Error('Image input mode requires an image file.');
    }

    currentTensors = await Promise.all(
      timings[0].inputShapes.map(async (shape) => {
        const inputValues =
          inputMode === 'image' && imageFile && isImageShape(shape)
            ? await imageFileToInput(imageFile, shape)
            : makeDummyInput(shape);
        return new Tensor(inputValues, shape);
      }),
    );

    for (let index = 0; index < stage12Models.length; index += 1) {
      const outputs = await runStage(stage12Models[index], timings[index], currentTensors, warnings);
      deleteTensors(currentTensors);
      currentTensors = outputs;
    }

    materializedStageOutputs = await materializeTensors(currentTensors);
  } finally {
    deleteTensors(currentTensors);
    currentTensors = [];
    for (const model of stage12Models) {
      model.delete();
    }
    unloadLiteRt();
  }

  await loadLiteRtForBackend('wasm');
  const wasmModels: CompiledModel[] = [];
  try {
    const { model, timing } = await compileModelForStage(
      modelFiles[2],
      'wasm',
      `Executed in a separate non-JSPI WASM runtime after ${stage12Backend.toUpperCase()} Stage 1/2.`,
    );
    wasmModels.push(model);
    timings.push(timing);
    currentTensors = materializedStageOutputs.map((tensor) => new Tensor(tensor.values, tensor.shape));

    const outputs = await runStage(model, timing, currentTensors, warnings);
    deleteTensors(currentTensors);
    currentTensors = outputs;

    const totalMs = timings.reduce((sum, stage) => sum + stage.runMs, 0);
    const selectedOutput = currentTensors.reduce((best, tensor) =>
      tensorElementCount(tensor) > tensorElementCount(best) ? tensor : best,
    );
    finalTensor = await selectedOutput.moveTo('wasm');
    const outputValues = finalTensor.toTypedArray();
    const outputPreview = Array.from(outputValues.slice(0, 12), (value) => Number(value));

    return {
      backend: stage12Backend,
      totalMs,
      stages: timings,
      outputLength: outputValues.length,
      outputNorm: l2Norm(outputValues),
      outputPreview,
      warnings,
      webGpuInitInfo,
    };
  } finally {
    deleteTensors(currentTensors);
    if (finalTensor && !finalTensor.deleted) {
      finalTensor.delete();
    }
    for (const model of wasmModels) {
      model.delete();
    }
  }
}

function formatMs(value: number): string {
  return `${value.toFixed(2)} ms`;
}

function formatBytes(value: number | null): string {
  if (value === null) {
    return '-';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatNamedShapes(names: readonly string[], shapes: readonly number[][]): string {
  return shapes.map((shape, index) => `${names[index]} [${shape.join(', ')}]`).join(' | ');
}

function formatBufferTypes(bufferTypes: readonly string[][]): string {
  return bufferTypes.map((types) => types.join('/')).join(' | ');
}

function formatStats(stats: readonly TensorStats[]): string {
  if (stats.length === 0) {
    return '-';
  }
  return stats
    .map(
      (stat) =>
        `len=${stat.length} norm=${stat.norm.toFixed(6)} min=${stat.min.toFixed(6)} max=${stat.max.toFixed(6)} first=${stat.preview
          .map((value) => value.toFixed(4))
          .join(',')}`,
    )
    .join(' | ');
}

function App() {
  const [modelSetId, setModelSetId] = useState<ModelSetId>('transface-l');
  const [inputMode, setInputMode] = useState<InputMode>('dummy');
  const [stagePreference, setStagePreference] = useState<StagePreference>('webgpu-stage12');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [versions, setVersions] = useState<VersionInfo | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    setLogs((current) => [...current, { time: nowLabel(), level, message }]);
  }, []);

  useEffect(() => {
    void window.transface.listModels().then(setModels).catch((error: unknown) => {
      addLog('error', formatError(error));
    });
    void window.transface.getVersions().then(setVersions).catch((error: unknown) => {
      addLog('error', formatError(error));
    });
  }, [addLog]);

  useEffect(() => {
    const onError = (event: ErrorEvent) => addLog('error', formatError(event.error ?? event.message));
    const onRejection = (event: PromiseRejectionEvent) => addLog('error', formatError(event.reason));
    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onRejection);
    return () => {
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onRejection);
    };
  }, [addLog]);

  const selectedModelSet = MODEL_SETS[modelSetId];
  const selectedModelStatuses = useMemo<ModelStatus[]>(
    () =>
      selectedModelSet.files.map(
        (filename) => models.find((model) => model.filename === filename) ?? { filename, exists: false, sizeBytes: null },
      ),
    [modelSetId, models, selectedModelSet.files],
  );
  const allModelsReady = useMemo(
    () => selectedModelStatuses.length > 0 && selectedModelStatuses.every((model) => model.exists),
    [selectedModelStatuses],
  );
  const backendPreferenceLabel = stagePreference === 'wasm-stage12' ? 'WASM' : 'WebGPU';
  const backendControlLabel = selectedModelSet.runtime === 'split' ? 'Stage 1/2' : 'Backend';
  const runtimeDescription =
    selectedModelSet.runtime === 'split'
      ? stagePreference === 'wasm-stage12'
        ? 'WASM split runtime'
        : 'WebGPU delegate with WASM Stage 3'
      : stagePreference === 'wasm-stage12'
        ? 'WASM single-model runtime'
        : 'WebGPU single-model runtime';

  const logWarnings = useCallback(
    (warnings: readonly string[]) => {
      for (const warning of warnings) {
        addLog('warn', warning);
      }
    },
    [addLog],
  );

  const runCheck = useCallback(async () => {
    setRunning(true);
    setResult(null);
    addLog(
      'info',
      `Starting ${selectedModelSet.label} ${inputMode} inference. ${backendControlLabel} preference: ${backendPreferenceLabel}. WebGPU supported: ${isWebGPUSupported() ? 'yes' : 'no'}.`,
    );

    try {
      if (selectedModelSet.runtime === 'single') {
        if (stagePreference === 'wasm-stage12') {
          const wasmResult = await runPipeline(selectedModelSet.files, 'wasm', inputMode, imageFile);
          setResult({ ...wasmResult, fallbackUsed: false });
          logWarnings(wasmResult.warnings);
          addLog('info', `WASM inference completed in ${formatMs(wasmResult.totalMs)}.`);
          return;
        }

        let webgpuError: unknown = null;
        try {
          const webgpuResult = await runPipeline(selectedModelSet.files, 'webgpu', inputMode, imageFile, (info) => {
            addLog('info', `WebGPU initialized: ${JSON.stringify(info)}`);
          });
          setResult({ ...webgpuResult, fallbackUsed: false });
          logWarnings(webgpuResult.warnings);
          addLog('info', `WebGPU inference completed in ${formatMs(webgpuResult.totalMs)}.`);
          return;
        } catch (error) {
          webgpuError = error;
          addLog('error', `WebGPU pipeline failed; falling back to WASM.\n${formatError(error)}`);
        }

        try {
          const wasmResult = await runPipeline(selectedModelSet.files, 'wasm', inputMode, imageFile);
          setResult({ ...wasmResult, fallbackUsed: true });
          logWarnings(wasmResult.warnings);
          addLog('info', `WASM inference completed in ${formatMs(wasmResult.totalMs)}.`);
        } catch (wasmError) {
          throw new Error(`WASM fallback failed after WebGPU error.\nWebGPU: ${formatError(webgpuError)}\nWASM: ${formatError(wasmError)}`);
        }
        return;
      }

      if (stagePreference === 'wasm-stage12') {
        const wasmResult = await runSplitRuntimePipeline(TRANSFACE_L_MODEL_FILES, 'wasm', inputMode, imageFile);
        setResult({ ...wasmResult, fallbackUsed: false });
        logWarnings(wasmResult.warnings);
        addLog('info', `WASM-preferred inference completed in ${formatMs(wasmResult.totalMs)}.`);
        return;
      }

      let webgpuError: unknown = null;
      try {
        const webgpuResult = await runSplitRuntimePipeline(TRANSFACE_L_MODEL_FILES, 'webgpu', inputMode, imageFile, (info) => {
          addLog('info', `WebGPU initialized: ${JSON.stringify(info)}`);
        });
        setResult({ ...webgpuResult, fallbackUsed: false });
        logWarnings(webgpuResult.warnings);
        addLog('info', `WebGPU inference completed in ${formatMs(webgpuResult.totalMs)}.`);
        return;
      } catch (error) {
        webgpuError = error;
        addLog('error', `WebGPU pipeline failed; falling back to WASM.\n${formatError(error)}`);
      }

      try {
        const wasmResult = await runSplitRuntimePipeline(TRANSFACE_L_MODEL_FILES, 'wasm', inputMode, imageFile);
        setResult({ ...wasmResult, fallbackUsed: true });
        logWarnings(wasmResult.warnings);
        addLog('info', `WASM inference completed in ${formatMs(wasmResult.totalMs)}.`);
      } catch (wasmError) {
        throw new Error(`WASM fallback failed after WebGPU error.\nWebGPU: ${formatError(webgpuError)}\nWASM: ${formatError(wasmError)}`);
      }
    } catch (error) {
      addLog('error', formatError(error));
    } finally {
      setRunning(false);
    }
  }, [
    addLog,
    backendControlLabel,
    backendPreferenceLabel,
    imageFile,
    inputMode,
    logWarnings,
    selectedModelSet.files,
    selectedModelSet.label,
    selectedModelSet.runtime,
    stagePreference,
  ]);

  return (
    <main className="app-shell">
      <section className="toolbar">
        <div>
          <h1>TransFace LiteRT Check</h1>
          <p>LiteRT.js + Electron runtime check for TransFace-L and TransFace-S models.</p>
        </div>
        <button type="button" onClick={runCheck} disabled={running || !allModelsReady || (inputMode === 'image' && !imageFile)}>
          {running ? 'Running...' : 'Run'}
        </button>
      </section>

      <section className="panel controls-panel">
        <div className="control-group">
          <span className="label">Model</span>
          <div className="segmented">
            <button
              className={modelSetId === 'transface-l' ? 'active' : ''}
              type="button"
              onClick={() => {
                setModelSetId('transface-l');
                setResult(null);
              }}
            >
              TransFace-L
            </button>
            <button
              className={modelSetId === 'transface-s' ? 'active' : ''}
              type="button"
              onClick={() => {
                setModelSetId('transface-s');
                setResult(null);
              }}
            >
              TransFace-S
            </button>
          </div>
        </div>

        <div className="control-group">
          <span className="label">Input</span>
          <div className="segmented">
            <button className={inputMode === 'dummy' ? 'active' : ''} type="button" onClick={() => setInputMode('dummy')}>
              Dummy
            </button>
            <button className={inputMode === 'image' ? 'active' : ''} type="button" onClick={() => setInputMode('image')}>
              Image
            </button>
          </div>
        </div>

        <label className="file-control">
          <span className="label">Image file</span>
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            disabled={inputMode !== 'image'}
            onChange={(event) => setImageFile(event.currentTarget.files?.[0] ?? null)}
          />
        </label>

        <div className="control-group">
          <span className="label">{backendControlLabel}</span>
          <div className="segmented">
            <button
              className={stagePreference === 'webgpu-stage12' ? 'active' : ''}
              type="button"
              onClick={() => setStagePreference('webgpu-stage12')}
            >
              WebGPU
            </button>
            <button
              className={stagePreference === 'wasm-stage12' ? 'active' : ''}
              type="button"
              onClick={() => setStagePreference('wasm-stage12')}
            >
              WASM
            </button>
          </div>
        </div>

        <div className="runtime-box">
          <span className="label">Backend</span>
          <strong>{runtimeDescription}</strong>
          <span>{isWebGPUSupported() ? 'WebGPU API detected' : 'WebGPU API not detected'}</span>
        </div>
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Models</h2>
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Status</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {selectedModelStatuses.map((model) => (
                <tr key={model.filename}>
                  <td>{model.filename}</td>
                  <td className={model.exists ? 'ok' : 'bad'}>{model.exists ? 'Found' : 'Missing'}</td>
                  <td>{formatBytes(model.sizeBytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel">
          <h2>Runtime</h2>
          {versions ? (
            <dl className="versions">
              <dt>pnpm</dt>
              <dd>{versions.packageManager}</dd>
              <dt>Electron</dt>
              <dd>{versions.electron}</dd>
              <dt>Chrome</dt>
              <dd>{versions.chrome}</dd>
              <dt>Node</dt>
              <dd>{versions.node}</dd>
              <dt>LiteRT.js</dt>
              <dd>{versions.dependencies['@litertjs/core']}</dd>
            </dl>
          ) : (
            <p>Loading runtime versions...</p>
          )}
        </div>
      </section>

      {result && (
        <section className="panel">
          <h2>Result</h2>
          <div className="metrics">
            <div>
              <span>Backend</span>
              <strong>{result.backend.toUpperCase()}{result.fallbackUsed ? ' fallback' : ''}</strong>
            </div>
            <div>
              <span>Total</span>
              <strong>{formatMs(result.totalMs)}</strong>
            </div>
            <div>
              <span>Output length</span>
              <strong>{result.outputLength}</strong>
            </div>
            <div>
              <span>L2 norm</span>
              <strong>{result.outputNorm.toFixed(6)}</strong>
            </div>
          </div>

          {result.warnings.length > 0 && (
            <div className="warning-box">
              {result.warnings.map((warning) => (
                <div key={warning}>{warning}</div>
              ))}
            </div>
          )}

          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Backend</th>
                <th>Bytes</th>
                <th>Compile</th>
                <th>Run</th>
                <th>Input</th>
                <th>Input Stats</th>
                <th>Input Buffers</th>
                <th>Output</th>
                <th>Output Stats</th>
                <th>Output Buffers</th>
                <th>Warnings</th>
                <th>Accelerated</th>
              </tr>
            </thead>
            <tbody>
              {result.stages.map((stage, index) => (
                <tr key={stage.model}>
                  <td>{index + 1}</td>
                  <td className={stage.stageBackend === 'webgpu' ? 'ok' : 'warn'}>
                    {stage.stageBackend.toUpperCase()}
                    {stage.compileNote ? ` (${stage.compileNote})` : ''}
                  </td>
                  <td>{formatBytes(stage.modelBytes)}</td>
                  <td>{formatMs(stage.compileMs)}</td>
                  <td>{formatMs(stage.runMs)}</td>
                  <td>{formatNamedShapes(stage.inputNames, stage.inputShapes)}</td>
                  <td>{formatStats(stage.inputStats)}</td>
                  <td>{formatBufferTypes(stage.inputBufferTypes)}</td>
                  <td>{formatNamedShapes(stage.outputNames, stage.outputShapes)}</td>
                  <td>{formatStats(stage.outputStats)}</td>
                  <td>{formatBufferTypes(stage.outputBufferTypes)}</td>
                  <td>{stage.warnings.length === 0 ? '-' : stage.warnings.join(' | ')}</td>
                  <td className={stage.fullyAccelerated ? 'ok' : 'warn'}>{stage.fullyAccelerated ? 'Yes' : 'Partial'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <pre className="preview">{result.outputPreview.map((value) => value.toFixed(6)).join(', ')}</pre>
        </section>
      )}

      <section className="panel">
        <h2>Error Log</h2>
        <pre className="log">
          {logs.length === 0
            ? 'No logs yet.'
            : logs.map((entry) => `[${entry.time}] ${entry.level.toUpperCase()} ${entry.message}`).join('\n\n')}
        </pre>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root') as HTMLElement).render(<App />);
