import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { adaptModel, argmax, FloatModel, loadModel, widenModel } from './tinyllama.mts';

const root = 'models/stories260K';
const run = mkdtempSync('runs/real-model-');
const original = loadModel(`${root}/stories260K.bin`);
const adapted = adaptModel(original);
const maxima: Record<string, Record<string, number>> = {};
const dense = new FloatModel(original);
const padded = new FloatModel(adapted, true, (site, l, values) => {
  let max = maxima[site]?.[l] ?? 0;
  for (const value of values) max = Math.max(max, Math.abs(value));
  (maxima[site] ??= {})[l] = max;
});
const wideMaxima: Record<string, Record<string, number>> = {};
const wide = new FloatModel(adaptModel(widenModel(original)), true, (site, l, values) => {
  let max = wideMaxima[site]?.[l] ?? 0;
  for (const value of values) max = Math.max(max, Math.abs(value));
  (wideMaxima[site] ??= {})[l] = max;
});
const tokens: number[] = [1];
let worst = 0;
let wideWorst = 0;
for (let i = 0; i < 64; i++) {
  const a = dense.forward(tokens[i]), b = padded.forward(tokens[i]);
  for (let j = 0; j < a.length; j++) worst = Math.max(worst, Math.abs(a[j] - b[j]));
  assert.deepEqual(b, a, `adaptation changes logits at position ${i}`);
  const w = wide.forward(tokens[i]);
  assert.equal(argmax(w), argmax(a), `residual replication changes the greedy token at ${i}`);
  for (let j = 0; j < a.length; j++) wideWorst = Math.max(wideWorst, Math.abs(a[j] - w[j]));
  assert(wideWorst < 1e-3, 'replicated residual model exceeds float tolerance');
  tokens.push(argmax(a));
}
const tokenizer = readFileSync(`${root}/tok512.bin`);
let at = 4;
const pieces: string[] = [];
for (let i = 0; i < original.vocab; i++) {
  const size = tokenizer.readInt32LE(at + 4); at += 8;
  assert(size >= 0 && at + size <= tokenizer.length);
  pieces.push(tokenizer.subarray(at, at + size).toString('utf8')); at += size;
}
assert.equal(at, tokenizer.length);
const text = tokens.slice(1).map((id, i) => {
  let piece = pieces[id];
  if (i === 0 && piece.startsWith(' ')) piece = piece.slice(1);
  return piece.replace(/^<0x([0-9A-Fa-f]{2})>$/, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
}).join('');
const reference = spawnSync('build/macos/stories260K-reference', [
  `${root}/stories260K.bin`, '-z', `${root}/tok512.bin`, '-t', '0', '-n', '64',
], { encoding: 'utf8' });
assert.equal(reference.status, 0, reference.stderr);
assert.equal(text.trimEnd(), reference.stdout.trimEnd(), 'TypeScript greedy output differs from original C reference');
const hash = (file: string) => createHash('sha256').update(readFileSync(`${root}/${file}`)).digest('hex');
const result = {
  modelRevision: '0bd21da7698eaf29a0d7de3992de8a46ef624add',
  checkpointSHA256: hash('stories260K.bin'), tokenizerSHA256: hash('tok512.bin'),
  originalShape: { d: original.d, hidden: original.hidden, layers: original.layers,
    heads: original.heads, groups: original.groups, vocab: original.vocab },
  adaptedShape: { d: adapted.d, f: adapted.hidden, fe: adapted.hidden / 2, layers: adapted.layers, experts: 2 },
  comparedPositions: 64, comparedLogits: 64 * original.vocab, maximumLogitDifference: worst,
  replicatedResidualWidth: 256, replicatedMaximumLogitDifference: wideWorst,
  tokens, text, referenceExit: reference.status,
  scope: 'float model import and dense-to-MoE padding equivalence; no integer inference proof',
};
writeFileSync(`${run}/results.json`, JSON.stringify(result, null, 2) + '\n');
writeFileSync(`${run}/maxima.json`, JSON.stringify(maxima, null, 2) + '\n');
writeFileSync(`${run}/wide-maxima.json`, JSON.stringify(wideMaxima, null, 2) + '\n');
writeFileSync(`${run}/tokens.bin`, Buffer.from(new Uint32Array(tokens.slice(0, 64)).buffer));
console.log(JSON.stringify(result, null, 2));
console.log(`PASS: ${run}; real float inference, not yet a request proof`);
