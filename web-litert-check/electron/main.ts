import { app, BrowserWindow, ipcMain, net, protocol } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const MODEL_FILES = [
  'glint360k_model_TransFace_L_0001_float32.tflite',
  'glint360k_model_TransFace_L_0002_float32.tflite',
  'glint360k_model_TransFace_L_0003_float32.tflite',
] as const;

const APP_DIR = path.resolve(__dirname, '..');
const MODEL_ROOT = path.join(APP_DIR, 'public', 'models');
const WASM_ROOT = path.join(APP_DIR, 'node_modules', '@litertjs', 'core', 'wasm');
const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const packageJson = JSON.parse(
  fs.readFileSync(path.join(APP_DIR, 'package.json'), 'utf8'),
) as {
  version: string;
  packageManager: string;
  dependencies: Record<string, string>;
  devDependencies: Record<string, string>;
};

const CHROMIUM_SWITCHES: Array<[name: string, value?: string]> = [
  ['ignore-gpu-blocklist'],
  ['enable-zero-copy'],
  ['disable-gpu-sandbox'],
  ['enable-unsafe-webgpu'],
  ['enable-webgpu-developer-features'],
  ['enable-features', process.platform === 'win32' ? 'WebGPU,WebGPUService' : 'Vulkan,WebGPU,WebGPUService'],
  ['use-webgpu-adapter', 'default'],
  ['disable-features', 'UseSkiaRenderer,UseChromeOSDirectVideoDecoder'],
];

for (const [name, value] of CHROMIUM_SWITCHES) {
  app.commandLine.appendSwitch(name, value);
}

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'transface-wasm',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      corsEnabled: true,
    },
  },
]);

function contentTypeFor(filePath: string): string {
  if (filePath.endsWith('.wasm')) {
    return 'application/wasm';
  }
  if (filePath.endsWith('.js')) {
    return 'text/javascript';
  }
  if (filePath.endsWith('.tflite')) {
    return 'application/octet-stream';
  }
  return 'application/octet-stream';
}

async function responseForFile(filePath: string): Promise<Response> {
  const response = await net.fetch(pathToFileURL(filePath).toString());
  const headers = new Headers(response.headers);
  headers.set('Content-Type', contentTypeFor(filePath));
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: {
      ...Object.fromEntries(headers.entries()),
      'Content-Length': String(fs.statSync(filePath).size),
    },
  });
}

function decodeRequestPath(request: Request): string {
  const url = new URL(request.url);
  const joined = url.hostname === 'assets' ? url.pathname : `${url.hostname}${url.pathname}`;
  return decodeURIComponent(joined).replace(/^\/+/, '');
}

function registerProtocols(): void {
  protocol.handle('transface-wasm', (request) => {
    const relativePath = decodeRequestPath(request);
    const wasmPath = path.resolve(WASM_ROOT, relativePath);
    const relativeFromRoot = path.relative(WASM_ROOT, wasmPath);

    if (relativeFromRoot.startsWith('..') || path.isAbsolute(relativeFromRoot)) {
      return new Response(`Wasm path is outside whitelist: ${relativePath}`, { status: 403 });
    }
    if (!fs.existsSync(wasmPath) || !fs.statSync(wasmPath).isFile()) {
      return new Response(`Wasm file does not exist: ${relativePath}`, { status: 404 });
    }

    return responseForFile(wasmPath);
  });
}

function createWindow(): void {
  const preload = path.join(__dirname, 'preload.js');
  const window = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 920,
    minHeight: 640,
    backgroundColor: '#f5f7fa',
    webPreferences: {
      preload,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  if (isDev) {
    void window.loadURL(process.env.VITE_DEV_SERVER_URL as string);
  } else {
    void window.loadFile(path.join(APP_DIR, 'dist-renderer', 'index.html'));
  }
}

ipcMain.handle('transface:list-models', () =>
  MODEL_FILES.map((filename) => {
    const filePath = path.join(MODEL_ROOT, filename);
    if (!fs.existsSync(filePath)) {
      return { filename, exists: false, sizeBytes: null };
    }
    return { filename, exists: true, sizeBytes: fs.statSync(filePath).size };
  }),
);

ipcMain.handle('transface:get-versions', () => ({
  appVersion: packageJson.version,
  packageManager: packageJson.packageManager,
  dependencies: packageJson.dependencies,
  devDependencies: packageJson.devDependencies,
  electron: process.versions.electron,
  chrome: process.versions.chrome,
  node: process.versions.node,
}));

app.whenReady().then(() => {
  registerProtocols();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
