export interface Contract {
  id: string;
  family: string;
  model: string;
  revision: string;
  quantization: string;
  machine: string;
  promptTokens: number;
  decodeTokens: number;
  warmupRuns: number;
  measuredRuns: number;
}

export interface BenchmarkResult {
  schemaVersion: 1;
  contractId: string;
  candidate: string;
  timestamp: string;
  correctness: { passed: boolean; mismatches: number };
  prefill: { tokens: number; secondsPerToken: number; tokensPerSecond: number };
  decode: { tokens: number; secondsPerToken: number; tokensPerSecond: number };
  ttft?: { seconds: number };
  telemetry: {
    gpuName: string;
    computeCapability?: string;
    driver?: string;
    totalMemoryGiB?: number;
    freeMemoryGiB?: number;
    temperatureC?: number;
  };
  eligible: boolean;
  failure?: string;
}

export interface ScoreRecord {
  schemaVersion: 1;
  contractId: string;
  candidate: string;
  score: number | null;
  prefillSpeedup: number | null;
  decodeSpeedup: number | null;
  ttftSpeedup: number | null;
  deltaPercent: number | null;
  eligible: boolean;
  createdAt: string;
  source: string;
  prefillTokensPerSecond?: number;
  decodeTokensPerSecond?: number;
  ttftSeconds?: number;
}
