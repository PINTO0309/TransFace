import { spawn } from 'node:child_process';
import process from 'node:process';

const isWindows = process.platform === 'win32';
const bin = (name) => (isWindows ? `${name}.cmd` : name);

const vite = spawn(bin('vite'), ['--host', '127.0.0.1'], {
  stdio: ['ignore', 'pipe', 'pipe'],
  shell: false,
});

let electron;
let started = false;

function electronEnv(extra = {}) {
  const env = { ...process.env, ...extra };
  delete env.ELECTRON_RUN_AS_NODE;
  return env;
}

function stopAll(exitCode = 0) {
  if (electron && !electron.killed) {
    electron.kill();
  }
  if (!vite.killed) {
    vite.kill();
  }
  process.exit(exitCode);
}

function startElectron() {
  if (started) {
    return;
  }
  started = true;
  electron = spawn(bin('electron'), ['.'], {
    stdio: 'inherit',
    shell: false,
    env: electronEnv({
      VITE_DEV_SERVER_URL: 'http://127.0.0.1:5173',
    }),
  });
  electron.on('exit', (code) => stopAll(code ?? 0));
}

vite.stdout.on('data', (data) => {
  const text = data.toString();
  process.stdout.write(text);
  if (text.includes('Local:') || text.includes('ready in')) {
    startElectron();
  }
});

vite.stderr.on('data', (data) => {
  process.stderr.write(data);
});

vite.on('exit', (code) => {
  if (!electron || !electron.killed) {
    stopAll(code ?? 1);
  }
});

process.on('SIGINT', () => stopAll(0));
process.on('SIGTERM', () => stopAll(0));
