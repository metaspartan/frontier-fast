import type { Contract } from "./types";

export const TRACKS: Record<string, Contract> = {
  "laguna-xs-2.1-nvfp4-gb10-v1": {
    id: "laguna-xs-2.1-nvfp4-gb10-v1",
    family: "laguna-xs-2.1",
    model: "Poolside/Laguna-XS-2.1-NVFP4",
    revision: "pinned-by-deployment",
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
    model: "Poolside/Laguna-S-2.1-NVFP4",
    revision: "pinned-by-deployment",
    quantization: "nvfp4",
    machine: "dgx-spark-gb10-sm121",
    promptTokens: 512,
    decodeTokens: 128,
    warmupRuns: 2,
    measuredRuns: 5,
  },
};
