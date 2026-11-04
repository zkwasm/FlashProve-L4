import { readFileSync } from 'node:fs';

export type Tensor = { dims: number[]; data: Float32Array };
export type Model = {
  d: number; hidden: number; layers: number; heads: number; groups: number;
  vocab: number; context: number; tensors: Map<string, Tensor>;
};

// karpathy/llama2.c, pinned reference-run.c: seven int32 header words, then F32 tensors.
export function loadModel(path: string): Model {
  const file = readFileSync(path);
  if (file.length < 28) throw Error('truncated model header');
  const [d, hidden, layers, heads, groups, signedVocab, context] =
    Array.from({ length: 7 }, (_, i) => file.readInt32LE(i * 4));
  const vocab = Math.abs(signedVocab);
  if (![d, hidden, layers, heads, groups, vocab, context].every(x => x > 0 && x <= 65536)
    || d % heads || heads % groups || (d / heads) % 2) throw Error('unsupported model shape');
  let offset = 28;
  const tensors = new Map<string, Tensor>();
  function take(name: string, dims: number[]) {
    const count = dims.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(count) || offset + count * 4 > file.length) throw Error('truncated tensor: ' + name);
    const data = Float32Array.from({ length: count }, (_, i) => file.readFloatLE(offset + i * 4));
    if (!data.every(Number.isFinite)) throw Error('nonfinite tensor: ' + name);
    offset += count * 4;
    tensors.set(name, { dims, data });
  }
  take('token_embd.weight', [vocab, d]);
  const kv = groups * d / heads;
  for (const [kind, dims] of [
    ['attn_norm', [d]], ['attn_q', [d, d]], ['attn_k', [kv, d]], ['attn_v', [kv, d]],
    ['attn_output', [d, d]], ['ffn_norm', [d]], ['ffn_gate', [hidden, d]],
    ['ffn_down', [d, hidden]], ['ffn_up', [hidden, d]],
  ] as [string, number[]][]) {
    for (let l = 0; l < layers; l++) take(`blk.${l}.${kind}.weight`, dims);
  }
  take('output_norm.weight', [d]);
  // Legacy stored rotary tables are ignored by the original reference too.
  offset += context * (d / heads) * 4;
  if (signedVocab < 0) take('output.weight', [vocab, d]);
  else tensors.set('output.weight', tensors.get('token_embd.weight')!);
  if (offset !== file.length) throw Error('model file has missing or extra bytes');
  return { d, hidden, layers, heads, groups, vocab, context, tensors };
}

export function tensor(m: Model, name: string): Tensor {
  const t = m.tensors.get(name);
  if (!t) throw Error('missing tensor: ' + name);
  return t;
}

// Four residual copies keep RMS epsilon unchanged and satisfy 256-column blocks.
// Place each original rotary pair in every fourth pair of a 32-wide head: its
// frequency is unchanged. Other head channels are zero. Double q to compensate
// for the attention denominator changing from sqrt(8) to sqrt(32).
export function widenModel(original: Model): Model {
  if (original.d !== 64) throw Error('widening is currently defined only for the 64-wide model');
  const copies = 4, tensors = new Map<string, Tensor>();
  for (const [name, t] of original.tensors) {
    if (t.dims.length === 1) {
      const data = new Float32Array(t.data.length * copies);
      for (let i = 0; i < copies; i++) data.set(t.data, i * t.data.length);
      tensors.set(name, { dims: [data.length], data });
      continue;
    }
    const project = /attn_(q|k|v)\.weight$/.exec(name);
    const place = (j: number) => Math.floor(j / 8) * 32 + Math.floor(j % 8 / 2) * 8 + j % 2;
    if (project) {
      const [r, c] = t.dims, data = new Float32Array(r * copies * c * copies);
      for (let i = 0; i < r; i++) for (let j = 0; j < c * copies; j++)
        data[place(i) * c * copies + j] = t.data[i * c + j % c] / copies * (project[1] === 'q' ? 2 : 1);
      tensors.set(name, { dims: [r * copies, c * copies], data });
      continue;
    }
    if (/attn_output\.weight$/.test(name)) {
      const [r, c] = t.dims, data = new Float32Array(r * copies * c * copies);
      for (let i = 0; i < r * copies; i++) for (let j = 0; j < c; j++)
        data[i * c * copies + place(j)] = t.data[i % r * c + j];
      tensors.set(name, { dims: [r * copies, c * copies], data });
      continue;
    }
    const repeatRows = /attn_(q|k|v|output)\.weight$|ffn_down\.weight$/.test(name);
    const repeatCols = name !== 'token_embd.weight' && !/ffn_down\.weight$/.test(name);
    const [r, c] = t.dims, rows = r * (repeatRows ? copies : 1), cols = c * copies;
    const actualCols = /ffn_down\.weight$/.test(name) ? c : cols;
    const data = new Float32Array(rows * actualCols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < actualCols; j++)
      data[i * actualCols + j] = t.data[(i % r) * c + j % c] / (repeatCols ? copies : 1);
    tensors.set(name, { dims: [rows, actualCols], data });
  }
  return { ...original, d: original.d * copies, tensors };
}

// Exact real-arithmetic embedding of a dense FFN into the existing alternating MoE form:
// pad hidden units with zeros, use the shared expert, and zero the routed experts.
// Append an identity residual layer if needed. Numerical equivalence is tested separately.
export function adaptModel(original: Model): Model {
  const { d, vocab } = original;
  const expertWidth = 2 ** Math.ceil(Math.log2(original.hidden));
  const layers = original.layers + original.layers % 2;
  const tensors = new Map<string, Tensor>();
  for (const name of ['token_embd.weight', 'output_norm.weight', 'output.weight'])
    tensors.set(name, tensor(original, name));
  function put(l: number, kind: string, dims: number[], source?: string) {
    const data = new Float32Array(dims.reduce((a, b) => a * b, 1));
    if (source && l < original.layers) {
      const from = tensor(original, `blk.${l}.${source}.weight`);
      if (dims.length === 1) data.set(from.data);
      else for (let r = 0; r < from.dims[0]; r++)
        data.set(from.data.subarray(r * from.dims[1], (r + 1) * from.dims[1]), r * dims[1]);
    } else if (dims.length === 1) data.fill(1);
    tensors.set(`blk.${l}.${kind}.weight`, { dims, data });
  }
  for (let l = 0; l < layers; l++) {
    const kv = original.groups * d / original.heads;
    for (const [kind, dims] of [
      ['attn_norm', [d]], ['attn_q', [d, d]], ['attn_k', [kv, d]],
      ['attn_v', [kv, d]], ['attn_output', [d, d]], ['ffn_norm', [d]],
    ] as [string, number[]][]) put(l, kind, dims, kind);
    const width = l % 2 ? expertWidth : 2 * expertWidth;
    const ending = l % 2 ? '_shexp' : '';
    put(l, 'ffn_gate' + ending, [width, d], 'ffn_gate');
    put(l, 'ffn_up' + ending, [width, d], 'ffn_up');
    put(l, 'ffn_down' + ending, [d, width], 'ffn_down');
    if (l % 2) {
      put(l, 'ffn_gate_inp', [2, d]);
      put(l, 'ffn_gate_exps', [2, expertWidth, d]);
      put(l, 'ffn_up_exps', [2, expertWidth, d]);
      put(l, 'ffn_down_exps', [2, d, expertWidth]);
    }
  }
  return { ...original, layers, hidden: 2 * expertWidth, tensors };
}

const f = Math.fround;
function mat(t: Tensor, x: Float32Array): Float32Array {
  const [rows, cols] = t.dims;
  if (x.length !== cols) throw Error('matrix shape');
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) sum = f(sum + f(t.data[r * cols + c] * x[c]));
    y[r] = sum;
  }
  return y;
}
function norm(x: Float32Array): Float32Array {
  let sum = 0;
  for (const v of x) sum = f(sum + f(v * v));
  const scale = f(1 / f(Math.sqrt(f(f(sum / x.length) + f(1e-5)))));
  return x.map(v => f(v * scale));
}

export type Observer = (site: string, layer: number, values: Float32Array) => void;
export class FloatModel {
  readonly model: Model;
  readonly adapted: boolean;
  readonly observe?: Observer;
  private keys: Float32Array[][] = [];
  private values: Float32Array[][] = [];
  private position = 0;
  constructor(model: Model, adapted = false, observe?: Observer) {
    this.model = model; this.adapted = adapted; this.observe = observe;
    this.keys = Array.from({ length: model.layers }, () => []);
    this.values = Array.from({ length: model.layers }, () => []);
  }
  forward(tokenId: number): Float32Array {
    const m = this.model, { d, heads, groups } = m, dh = d / heads, pos = this.position;
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= m.vocab || pos >= m.context) throw Error('token or position out of range');
    let x = tensor(m, 'token_embd.weight').data.slice(tokenId * d, (tokenId + 1) * d);
    const see = (site: string, l: number, v: Float32Array) => this.observe?.(site, l, v);
    for (let l = 0; l < m.layers; l++) {
      const w = (kind: string) => tensor(m, `blk.${l}.${kind}.weight`);
      let a = norm(x); see('n1', l, a);
      a = a.map((v, j) => f(v * w('attn_norm').data[j]));
      const q = mat(w('attn_q'), a), k = mat(w('attn_k'), a), v = mat(w('attn_v'), a);
      for (const vec of [q, k]) for (let j = 0; j < vec.length; j += 2) {
        const angle = f(pos * f(1 / f(Math.pow(10000, (j % dh) / dh))));
        const co = f(Math.cos(angle)), si = f(Math.sin(angle)), a0 = vec[j], a1 = vec[j + 1];
        vec[j] = f(f(a0 * co) - f(a1 * si)); vec[j + 1] = f(f(a0 * si) + f(a1 * co));
      }
      see('q', l, q.map(z => z / Math.sqrt(dh))); see('k', l, k); see('v', l, v);
      this.keys[l].push(k); this.values[l].push(v);
      const attention = new Float32Array(d);
      for (let h = 0; h < heads; h++) {
        const scores = new Float32Array(pos + 1), kv = Math.floor(h / (heads / groups)) * dh;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          for (let c = 0; c < dh; c++) s = f(s + f(q[h * dh + c] * this.keys[l][t][kv + c]));
          scores[t] = f(s / f(Math.sqrt(dh)));
        }
        const max = Math.max(...scores); let total = 0;
        for (let t = 0; t <= pos; t++) { scores[t] = f(Math.exp(f(scores[t] - max))); total = f(total + scores[t]); }
        for (let t = 0; t <= pos; t++) {
          const weight = f(scores[t] / total);
          for (let c = 0; c < dh; c++) attention[h * dh + c] = f(attention[h * dh + c] + f(weight * this.values[l][t][kv + c]));
        }
      }
      const out = mat(w('attn_output'), attention);
      x = x.map((z, j) => f(z + out[j]));
      let b = norm(x); see('n2', l, b);
      b = b.map((z, j) => f(z * w('ffn_norm').data[j]));
      const ending = this.adapted && l % 2 ? '_shexp' : '';
      const gate = mat(w('ffn_gate' + ending), b), up = mat(w('ffn_up' + ending), b);
      see('u', l, up);
      const hidden = gate.map((z, j) => f(f(z * f(1 / f(1 + f(Math.exp(-z))))) * up[j]));
      const down = mat(w('ffn_down' + ending), hidden);
      x = x.map((z, j) => f(z + down[j]));
      // The routed branch of the adapted model consists entirely of exact zero tensors.
      // It is still exported and will be executed/proved by the upstream prover.
      if (this.adapted && l % 2) see('uR', l, new Float32Array(up.length));
    }
    x = norm(x); see('nF', m.layers, x);
    x = x.map((z, j) => f(z * tensor(m, 'output_norm.weight').data[j]));
    this.position++;
    return mat(tensor(m, 'output.weight'), x);
  }
}

export function argmax(values: Float32Array): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
