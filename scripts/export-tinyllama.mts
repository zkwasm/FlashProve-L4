import { mkdirSync, writeFileSync } from 'node:fs';
import { adaptModel, loadModel, widenModel } from './tinyllama.mts';

const out = process.argv[2];
if (!out) throw Error('usage: node scripts/export-tinyllama.mts NEW_DIRECTORY');
const model = adaptModel(widenModel(loadModel('models/stories260K/stories260K.bin')));
mkdirSync(out); // Refuse to overwrite an earlier conversion.
const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
const str = (s: string) => { const b = Buffer.from(s); return Buffer.concat([u64(b.length), b]); };
const metadata: Buffer[] = [];
function value(key: string, v: number | string) {
  metadata.push(str(key), u32(typeof v === 'string' ? 8 : 4), typeof v === 'string' ? str(v) : u32(v));
}
value('general.architecture', 'llama4');
value('general.name', 'TinyStories260K residual replication and zero-expert embedding');
value('flashprove.source', 'karpathy/tinyllamas@0bd21da7698eaf29a0d7de3992de8a46ef624add');
value('llama4.embedding_length', model.d);
value('llama4.block_count', model.layers);
value('llama4.attention.key_length', model.d / model.heads);
value('llama4.rope.freq_base', 10000);
// This is a conversion container for FlashProve. All layers use RoPE in KL.bin;
// the calibration tool's hardcoded Llama4 layer-3 exception must not be used.
value('flashprove.rope_layers', '0,1,2,3,4,5');
const infos: Buffer[] = [], data: Buffer[] = [];
let offset = 0;
for (const [name, t] of model.tensors) {
  // Routed tensors must use a block-decodable type. These added experts are
  // identically zero, so F16 stores their values exactly without quantization.
  const zeroExpert = /\.ffn_(gate|up|down)_exps\.weight$/.test(name);
  if (zeroExpert && t.data.some(v => v !== 0)) throw Error(`nonzero routed expert: ${name}`);
  infos.push(str(name), u32(t.dims.length), ...[...t.dims].reverse().map(u64), u32(zeroExpert ? 1 : 0), u64(offset));
  const b = Buffer.alloc(t.data.length * (zeroExpert ? 2 : 4));
  if (!zeroExpert) t.data.forEach((v, i) => b.writeFloatLE(v, i * 4));
  data.push(b);
  offset += b.length;
  const pad = (32 - offset % 32) % 32;
  data.push(Buffer.alloc(pad)); offset += pad;
}
const header = Buffer.concat([Buffer.from('GGUF'), u32(3), u64(model.tensors.size), u64(8), ...metadata, ...infos]);
const file = Buffer.concat([header, Buffer.alloc((32 - header.length % 32) % 32), ...data]);
writeFileSync(`${out}/model.gguf`, file, { flag: 'wx' });
const shape = { d: model.d, dh: model.d / model.heads, L: model.layers, H: model.heads,
  G: model.groups, f: model.hidden, fe: model.hidden / 2, NE: 2, V: model.vocab,
  eps: 1e-5, theta: 10000, moe: [1, 3, 5], rope: [0, 1, 2, 3, 4, 5] };
writeFileSync(`${out}/shape.json`, JSON.stringify(shape, null, 2) + '\n', { flag: 'wx' });
writeFileSync(`${out}/fold.txt`, 'eres 11\nqkrows asis\n', { flag: 'wx' });
console.log(`${out}: ${file.length} bytes, ${model.tensors.size} tensors; exact-zero routed experts F16, others F32`);
