export {};

declare global {
  interface ModelStatus {
    filename: string;
    exists: boolean;
    sizeBytes: number | null;
  }

  interface VersionInfo {
    appVersion: string;
    packageManager: string;
    dependencies: Record<string, string>;
    devDependencies: Record<string, string>;
    electron: string;
    chrome: string;
    node: string;
  }

  interface Window {
    transface: {
      listModels(): Promise<ModelStatus[]>;
      getVersions(): Promise<VersionInfo>;
    };
  }
}
