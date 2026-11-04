import assert from 'node:assert/strict';
import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { adaptModel, loadModel, widenModel } from './tinyllama.mts';

const artifact = 'runs/real-l4-v8';
const log = readFileSync(`${artifact}/commit.log`, 'utf8');
function number(pattern: RegExp): number {
  const match = pattern.exec(log);
  assert(match, `missing measurement: ${pattern}`);
  return Number(match[1]);
}
const model = adaptModel(widenModel(loadModel('models/stories260K/stories260K.bin')));
const results = JSON.parse(readFileSync(`${artifact}/verification/results.json`, 'utf8'));
const paths = {
  checkpoint: 'models/stories260K/stories260K.bin',
  convertedGGUF: 'models/stories260K/flashprove32/model.gguf',
  publicKey: `${artifact}/ivk/key.bin`,
  publicLeaf: `${artifact}/ivk/leaf/0.bin`,
  request: `${artifact}/in/example.req`,
  output: `${artifact}/out/example.y`,
  proof: `${artifact}/out/example.proof`,
};
const summary = {
  scope: 'One TinyStories260K run; no 0.7B model was tested. Pool peaks are not total GPU peak memory.',
  evidence: { commitmentLog: `${artifact}/commit.log`, hardware: `${artifact}/hardware.txt`,
    verification: `${artifact}/verification/results.json` },
  hardwareRecord: readFileSync(`${artifact}/hardware.txt`, 'utf8').trim(),
  workload: { tokens: 256, r: 255, e: 256, batch: 1, leaves: 1,
    convertedShape: { d: model.d, dh: model.d / model.heads, layers: model.layers,
      denseFFN: model.hidden, sharedFFN: model.hidden / 2, routedExperts: 2, vocab: model.vocab },
    convertedTensorElements: [...model.tensors.values()].reduce((sum, t) => sum + t.data.length, 0),
    committedTensors: 56 },
  files: Object.fromEntries(Object.entries(paths).map(([key, path]) => [key, { path, bytes: statSync(path).size }])),
  memory: {
    keyEnginePoolHighWaterBytes: number(/the key file: .*the engine's memory pool held (\d+)/),
    leafEnginePoolHighWaterBytes: number(/the entry: .*the engine's memory pool held (\d+)/),
    keySlabBytes: number(/the key file: GPU memory in bytes: the slab (\d+)/),
    leafSlabBytes: number(/the entry: GPU memory in bytes: the slab (\d+)/),
    l4EngineReserveBytes: number(/the key file: .*the least that is left to the engine: (\d+)/),
    totalCommitmentGPUHighWaterBytes: null,
    totalInferenceAndProofGPUHighWaterBytes: null,
    hostRSSHighWaterBytes: null,
  },
  timing: { commitmentPhases: [...log.matchAll(/^cp_commit: phase ([^:]+): ([\d.]+) s$/gm)]
    .map(m => ({ phase: m[1], seconds: Number(m[2]) })),
    macIndependentVerificationSeconds: results.results.find((r: { name: string }) => r.name === 'honest').seconds,
    inferenceAndProofWallSeconds: null,
    note: 'Single-run observations; commitment phases are rounded, not an end-to-end benchmark.' },
  hypothetical700MillionWeights: {
    assumption: 'Exactly 700,000,000 scalar weights; decimal bytes; payload only, excluding all metadata and proof overhead.',
    fp32Bytes: 700_000_000 * 4, fp16Bytes: 700_000_000 * 2,
    int8Bytes: 700_000_000, int4PackedBytes: 700_000_000 / 2,
    validatedHardwareRequirement: null,
  },
};
writeFileSync('runs/resource-summary.json', JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary, null, 2));
