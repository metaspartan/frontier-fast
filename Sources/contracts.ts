import type { Contract } from "./types";

export const TRACKS: Record<string, Contract> = {
  "laguna-xs-2.1-nvfp4-gb10-v1": {
    id: "laguna-xs-2.1-nvfp4-gb10-v1",
    family: "laguna-xs-2.1",
    model: "poolside/Laguna-XS-2.1-NVFP4",
    revision: "d32afde8b09af1539b49ff96ff5551c674485f8e",
    quantization: "nvfp4",
    machine: "dgx-spark-gb10-sm121",
    promptTokens: 512,
    decodeTokens: 128,
    warmupRuns: 2,
    measuredRuns: 5,
  },
  "laguna-s-2.1-nvfp4-gb10-v1": {
    id: "laguna-s-2.1-nvfp4-gb10-v1",
    family: "laguna-s-2.1",
    model: "poolside/Laguna-S-2.1-NVFP4",
    revision: "f8fdfcdc4e7b0c474a0102430a8cae0a3a358669",
    quantization: "nvfp4",
    machine: "dgx-spark-gb10-sm121",
    promptTokens: 512,
    decodeTokens: 128,
    warmupRuns: 2,
    measuredRuns: 5,
  },
};
