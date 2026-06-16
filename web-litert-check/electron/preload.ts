import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('transface', {
  listModels: () => ipcRenderer.invoke('transface:list-models'),
  getVersions: () => ipcRenderer.invoke('transface:get-versions'),
});
