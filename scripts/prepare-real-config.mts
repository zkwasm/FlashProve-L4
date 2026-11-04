import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const [verifier, calibration] = process.argv.slice(2);
assert(verifier && calibration, 'usage: node scripts/prepare-real-config.mts CP_VERIFY CALIBRATION_RUN');
const model = process.argv[4] ?? 'models/stories260K/flashprove32';
const out = mkdtempSync('runs/real-config-');
const shape = JSON.parse(readFileSync(`${model}/shape.json`, 'utf8'));
const maxima = JSON.parse(readFileSync(`${calibration}/wide-maxima.json`, 'utf8'));
const exponents: Record<string, Record<string, number> | number> = {};
for (const line of readFileSync(`${model}/wsums.txt`, 'utf8').trim().split('\n')) {
  const [name, , e] = line.split(' ');
  if (name === 'END') continue;
  const match = /^blk\.(\d+)\.(.+)\.weight$/.exec(name);
  if (match) {
    // Any exponent represents the same exact zero. Use 12 for fitting the
    // identically zero routed/identity branches, keeping their shifts positive.
    const zero = match[1] === '5' || match[2].endsWith('_exps') || match[2] === 'ffn_gate_inp';
    const byLayer = (exponents[match[2]] ??= {}) as Record<string, number>;
    byLayer[match[1]] = zero ? 12 : Number(e);
  }
  else if (name === 'output.weight') exponents.output = Number(e);
}
for (const site of Object.keys(maxima)) for (const layer of Object.keys(maxima[site]))
  if (maxima[site][layer] === 0) maxima[site][layer] = 1; // A conservative bound on an identically zero site.
for (const l of shape.moe) {
  // The routed branch is identically zero. Give the fitter an equivalent scale for
  // its zero down product, so it need not lower the active shared branch's precision.
  (exponents.ffn_down_exps as Record<string, number>)[l] =
    (exponents.ffn_down_shexp as Record<string, number>)[l];
  maxima.uR[l] = maxima.u[l];
}
writeFileSync(`${out}/maxima.json`, JSON.stringify(maxima));
writeFileSync(`${out}/exponents.json`, JSON.stringify(exponents));
function run(name: string, cmd: string, args: string[], expected: number) {
  const r = spawnSync(cmd, args, { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
  writeFileSync(`${out}/${name}.log`, (r.stdout ?? '') + (r.stderr ?? ''));
  assert.equal(r.status, expected, `${name}: ${r.stdout} ${r.stderr}`);
  return r.stdout;
}
run('fit', 'python3', ['upstream/scales/fit_scales.py', `${model}/shape.json`, `${out}/maxima.json`,
  `${out}/exponents.json`, '13', `${out}/words.json`, '--eres', '11'], 0);
const words = JSON.parse(readFileSync(`${out}/words.json`, 'utf8'));
// The upstream words_to_policy helper only permits theta=500000. The independent
// verifier supports 10000, so write these public words directly and let it check them.
const replacements: Record<string, number | string> = {
  slots: 1, 'work positions': '4 6 8', 'ranges-part': 10,
  // 28,672 matrix rows * 2^(9-6) = 229,376 cells, plus record pads.
  // The GPU requires the mask offset to be even and a multiple of 8,192.
  table: '6 22 262144 8191 0 9',
  L: shape.L, ll: 1, d: shape.d, H: shape.H, G: shape.G, dh: shape.dh,
  f: shape.f, fe: shape.fe, NE: shape.NE, V: shape.V, lt: 8, la: 10,
  lmS: 12, kFM: 13, kFD: 14, iota_3: 1, MF: words.W.MF, KF: words.W.KF,
  s: words.W.E_logit,
};
for (const [line, key] of [['eps1', 'eps1'], ['eps2', 'eps2'], ['epsF', 'epsF'], ['theta0', 'theta']]) {
  const n = BigInt(words.W[key]); replacements[`${line}_lo`] = Number(n & 0xffffffffn);
  replacements[`${line}_hi`] = Number(n >> 32n);
}
const policy = readFileSync('upstream/prover/examples/target_small_f2_roomy/policy.txt', 'utf8')
  .split('\n').map(line => {
    const key = line.startsWith('work positions ') ? 'work positions' : line.split(' ')[0];
    return key in replacements ? `${key} ${replacements[key]}` : line;
  }).join('\n');
writeFileSync(`${out}/policy.txt`, policy);
const names = ['kq', 'kk', 'kv', 'ko', 'kg', 'ku', 'kd', 'M1', 'K1', 'M2', 'K2', 'sh', 'rp'];
const raw = words.KL.flatMap((row: Record<string, number>) => names.map(n => row[n]));
for (const l of shape.moe) for (const n of ['kr', 'kgR', 'kuR']) raw.push(words.KL[l][n]);
const scales = Buffer.alloc(raw.length * 4); raw.forEach((n: number, i: number) => scales.writeUInt32LE(n, i * 4));
writeFileSync(`${out}/KL.bin`, scales);
writeFileSync(`${out}/policy.txt`, run('tables', 'python3', ['upstream/scales/policy_tables.py', `${out}/policy.txt`], 0));
const arrays = run('arrays', verifier, ['-plan', 'arrays', out, '128'], 4);
writeFileSync(`${out}/arrays.txt`, arrays);
writeFileSync(`${out}/layout.txt`, run('layout', 'python3', ['upstream/prover/tools/pack_layout.py',
  '--arrays', `${out}/arrays.txt`, '17', '4'], 0));
run('accounting', verifier, ['-plan', 'accounting', out, '128'], 4);
console.log(out);
