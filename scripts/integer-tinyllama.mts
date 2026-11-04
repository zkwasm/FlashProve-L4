import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tensor, type Model } from './tinyllama.mts';

const clamp = (x: number, bound = 2047) => Math.min(Math.max(x, -bound), bound);
const rha = (x: number) => x >= 0 ? Math.floor(x + 0.5) : -Math.floor(-x + 0.5);
function rne(x: number): number {
  const n = Math.floor(x), part = x - n;
  return part < .5 ? n : part > .5 ? n + 1 : n % 2 === 0 ? n : n + 1;
}
function exact(x: number): number {
  if (!Number.isSafeInteger(x)) throw Error('integer reference exceeded exact Number arithmetic');
  return x;
}
const rq = (x: number, k: number) => Math.floor((exact(x) + 2 ** (k - 1)) / 2 ** k);
function isqrt(n: bigint): bigint {
  if (n < 0n) throw Error('negative square root');
  if (n < 2n) return n;
  let x = 1n << BigInt(Math.ceil(n.toString(2).length / 2));
  for (;;) { const y = (x + n / x) / 2n; if (y >= x) return x; x = y; }
}

// CPU reference of docs/model.md, for the exported model whose routed branches are zero.
// This evaluates the integer function; it is deliberately not a proof generator.
export class IntegerTinyllama {
  readonly model: Model;
  readonly words: { KL: Record<string, number>[]; W: Record<string, number> };
  private matrices = new Map<string, { rows: number; cols: number; data: Float64Array }>();
  private keys: number[][][];
  private values: number[][][];
  private position = 0;
  private exp: number[] = [];
  constructor(model: Model, config: string, sums: string) {
    this.model = model;
    this.words = JSON.parse(readFileSync(`${config}/words.json`, 'utf8'));
    const ew = new Map<string, number>();
    const hashes = new Map<string, string>();
    for (const line of readFileSync(sums, 'utf8').trim().split('\n')) {
      const [name, , e, , hash] = line.split(' ');
      if (name !== 'END') { ew.set(name, Number(e)); hashes.set(name, hash); }
    }
    for (const [name, t] of model.tensors) {
      if (/_exps\.weight$|ffn_gate_inp\.weight$/.test(name) && t.data.some(v => v !== 0))
        throw Error('CPU reference only supports identically zero routed branches');
      if (t.dims.length === 3) {
        const hash = createHash('sha256').update(Buffer.alloc(t.data.length * 2)).digest('hex');
        if (hash !== hashes.get(name)) throw Error('zero integer digest differs: ' + name);
        continue;
      }
      if (t.dims.length !== 2) continue;
      let gain: Float32Array | undefined, factor = 1;
      if (name === 'output.weight') gain = tensor(model, 'output_norm.weight').data;
      const match = /^blk\.(\d+)\.(.+)\.weight$/.exec(name);
      if (match && /^(attn_[qkv]|ffn_(gate|up).*)$/.test(match[2])) {
        const norm = match[2].startsWith('attn_') ? 'attn_norm' : 'ffn_norm';
        gain = tensor(model, `blk.${match[1]}.${norm}.weight`).data;
        if (match[2] === 'attn_q') factor = Math.fround(1 / Math.sqrt(model.d / model.heads));
      }
      const exponent = ew.get(name);
      if (exponent === undefined) throw Error('missing integer tensor exponent: ' + name);
      const [rows, cols] = t.dims;
      const data = Float64Array.from(t.data, (v, i) => {
        const gamma = gain ? Math.fround(gain[i % cols] * factor) : 1;
        return rne(Math.fround(Math.fround(gamma * 2 ** exponent) * v));
      });
      const bytes = name === 'token_embd.weight' ? 4 : 2;
      const encoded = Buffer.alloc(data.length * bytes);
      data.forEach((v, i) => bytes === 4 ? encoded.writeInt32LE(v, i * bytes) : encoded.writeInt16LE(v, i * bytes));
      if (createHash('sha256').update(encoded).digest('hex') !== hashes.get(name))
        throw Error('integer conversion differs from independent C++ weight_sums: ' + name);
      this.matrices.set(name, { rows, cols, data });
    }
    this.keys = Array.from({ length: model.layers }, () => []);
    this.values = Array.from({ length: model.layers }, () => []);
    for (let i = 0;; i++) { const e = rne(4096 * Math.exp(-i / 256)); if (!e) break; this.exp.push(e); }
  }
  private mat(name: string, x: number[]): number[] {
    const t = this.matrices.get(name); if (!t || x.length !== t.cols) throw Error('integer matrix shape: ' + name);
    return Array.from({ length: t.rows }, (_, r) => {
      let sum = 0; for (let j = 0; j < t.cols; j++) sum += x[j] * t.data[r * t.cols + j];
      return exact(sum);
    });
  }
  private norm(x: number[], M: number, K: number, epsilon: number): number[] {
    const squares = x.reduce((n, v) => n + BigInt(exact(v)) ** 2n, BigInt(epsilon));
    if (squares >= 1n << 78n) throw Error('norm range');
    const sigma = isqrt(squares), rc = ((1n << BigInt(M)) + sigma / 2n) / sigma;
    return x.map(v => {
      const n = BigInt(v) * rc + (1n << BigInt(K - 1));
      const w = Number(n >> BigInt(K));
      if (w < -32768 || w > 32639) throw Error('norm quotient range');
      return clamp(w);
    });
  }
  forward(token: number): number[] {
    const m = this.model, dh = m.d / m.heads, pos = this.position;
    if (pos >= m.context || token < 0 || token >= m.vocab) throw Error('input range');
    const emb = this.matrices.get('token_embd.weight')!;
    let x = Array.from(emb.data.subarray(token * m.d, (token + 1) * m.d));
    for (let l = 0; l < m.layers; l++) {
      const kl = this.words.KL[l], W = this.words.W;
      const mat = (kind: string, a: number[]) => this.mat(`blk.${l}.${kind}.weight`, a);
      const a = this.norm(x, kl.M1, kl.K1, W.eps1);
      const q = mat('attn_q', a), k = mat('attn_k', a);
      for (const [vec, shift] of [[q, kl.kq], [k, kl.kk]] as [number[], number][]) {
        for (let j = 0; j < vec.length; j += 2) {
          const angle = pos * Math.pow(W.theta, -(j % dh) / dh);
          const co = kl.rp ? rha(4096 * Math.cos(angle)) : 4096;
          const si = kl.rp ? rha(4096 * Math.sin(angle)) : 0;
          const a0 = vec[j], a1 = vec[j + 1];
          const u = rq(co * a0 - si * a1, shift), v = rq(si * a0 + co * a1, shift);
          if (Math.min(u, v) < -32768 || Math.max(u, v) > 32639) throw Error('query/key range');
          vec[j] = clamp(u); vec[j + 1] = clamp(v);
        }
      }
      const v = mat('attn_v', a).map(z => { const n = rq(z, kl.kv); if (n < -32768 || n > 32639) throw Error('value range'); return clamp(n); });
      this.keys[l].push(k); this.values[l].push(v);
      const o = new Array(m.d).fill(0);
      for (let h = 0; h < m.heads; h++) {
        const kv = Math.floor(h / (m.heads / m.groups)) * dh;
        const scores = this.keys[l].map(key => {
          let sum = 0; for (let c = 0; c < dh; c++) sum += q[h * dh + c] * key[kv + c]; return exact(sum);
        });
        const max = Math.max(...scores), e = scores.map(s => this.exp[Math.floor((max - s) / 2 ** kl.sh)] ?? 0);
        const Z = e.reduce((a, b) => a + b, 0);
        for (let c = 0; c < dh; c++) {
          const acc = e.reduce((sum, weight, t) => sum + weight * this.values[l][t][kv + c], 0);
          o[h * dh + c] = Math.floor((2 * exact(acc) + Z) / (2 * Z));
        }
      }
      const attention = mat('attn_output', o).map(z => rq(z, kl.ko));
      x = x.map((z, j) => exact(z + attention[j]));
      const b = this.norm(x, kl.M2, kl.K2, W.eps2), ending = l % 2 ? '_shexp' : '';
      const gate = mat('ffn_gate' + ending, b), up = mat('ffn_up' + ending, b).map(z => rq(z, kl.ku));
      const hidden = gate.map((z, j) => {
        const gw = rq(z, kl.kg), low = ((gw + 128) % 256 + 256) % 256 - 128, high = (gw - low) / 256;
        if (high < -16384 || high > 16383 || up[j] < -32768 || up[j] > 32639) throw Error('feed-forward range');
        const key = 256 * clamp(high, 127) + low;
        const real = Math.min(Math.max(key, -16384), 16383) / 256;
        return clamp(rha(64 * (real / (1 + Math.exp(-real)))), 4095) * up[j];
      });
      const down = mat('ffn_down' + ending, hidden).map(z => rq(z, kl.kd));
      x = x.map((z, j) => exact(z + down[j]));
    }
    const W = this.words.W, final = this.norm(x, W.MF, W.KF, W.epsF);
    this.position++;
    return this.mat('output.weight', final);
  }
}
