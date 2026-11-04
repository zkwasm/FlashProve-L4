import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { adaptModel, loadModel, widenModel } from './tinyllama.mts';
import { IntegerTinyllama } from './integer-tinyllama.mts';

const [config, calibration] = process.argv.slice(2);
assert(config && calibration, 'usage: node scripts/check-integer-model.mts CONFIG CALIBRATION_RUN');
const model = adaptModel(widenModel(loadModel('models/stories260K/stories260K.bin')));
const reference = new IntegerTinyllama(model, config, 'models/stories260K/flashprove32/wsums.txt');
const float = JSON.parse(readFileSync(`${calibration}/results.json`, 'utf8'));
const tokens: number[] = [], intArgmax = (x: number[]) => x.reduce((best, v, i) => v > x[best] ? i : best, 0);
for (let i = 0; i < 64; i++) tokens.push(intArgmax(reference.forward(float.tokens[i])));
const agree = tokens.filter((t, i) => t === float.tokens[i + 1]).length;
assert(agree >= 60, 'integer model no longer agrees at 60/64 float reference positions');
const generator = new IntegerTinyllama(model, config, 'models/stories260K/flashprove32/wsums.txt');
const greedyTokens = [1];
for (let i = 0; i < 64; i++) greedyTokens.push(intArgmax(generator.forward(greedyTokens[i])));
assert.deepEqual(greedyTokens.slice(0, 60), float.tokens.slice(0, 60), 'integer greedy generation changed the opening');
const tokenizer = readFileSync('models/stories260K/tok512.bin');
let offset = 4;
const pieces: string[] = [];
for (let i = 0; i < model.vocab; i++) {
  const size = tokenizer.readInt32LE(offset + 4); offset += 8;
  pieces.push(tokenizer.subarray(offset, offset + size).toString('utf8')); offset += size;
}
const greedyText = greedyTokens.slice(1).map((id, i) => {
  const piece = i === 0 ? pieces[id].replace(/^ /, '') : pieces[id];
  return piece.replace(/^<0x([0-9A-Fa-f]{2})>$/, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
}).join('');
const policy = readFileSync(`${config}/policy.txt`, 'utf8');
const count = 2 ** Number(/^lt (\d+)$/m.exec(policy)![1]);
// Repeat the checked story prefix to fill the GPU's public sequence length.
const prompt = Array.from({ length: count }, (_, i) => float.tokens[i % 64]);
const full = new IntegerTinyllama(model, config, 'models/stories260K/flashprove32/wsums.txt');
let nextToken = 0;
for (const token of prompt) nextToken = intArgmax(full.forward(token));
const result = { scope: 'CPU integer inference and independent C++ weight conversion checksums; no proof',
  positions: 64, agree, nextToken, requestPositions: count, floatToken: float.tokens[64], tokens, greedyTokens, greedyText };
writeFileSync(`${config}/integer-reference.json`, JSON.stringify(result, null, 2) + '\n');
const request = Buffer.alloc((count + 2) * 4);
prompt.forEach((t: number, i: number) => request.writeUInt32LE(t, 4 * i));
request.writeUInt32LE(count - 1, count * 4); request.writeUInt32LE(count, (count + 1) * 4);
writeFileSync(`${config}/example.req`, request);
console.log(JSON.stringify(result));
