import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
assert.ok(process.argv[2], 'Usage: node scripts/prepare-cpu-config.mts PATH_TO_CP_VERIFY');
const verifier = resolve(process.argv[2]);
mkdirSync(join(root, 'runs'), { recursive: true });
const out = mkdtempSync(join(root, 'runs', 'cpu-config-'));
const base = join(root, 'upstream/prover/examples/target_small_f2_roomy');
// A synthetic configuration for exercising the CPU commitment implementation.
// Keep the 128-bit target; lower work bits ask the planner for more queries.
const replacements = new Map([
  ['slots', 'slots 1'], ['work positions', 'work positions 4 6 8'],
  // 2,048 matrix rows * 2^(9-3) entries, plus three record pads per table row.
  ['table', 'table 3 21 131075 8191 0 9'], ['ranges-part', 'ranges-part 8'],
  ['L', 'L 2'], ['ll', 'll 1'], ['d', 'd 32'], ['dh', 'dh 32'], ['H', 'H 1'], ['G', 'G 1'],
  ['f', 'f 32'], ['fe', 'fe 16'], ['NE', 'NE 2'], ['V', 'V 128'],
  ['iota_3', 'iota_3 0'],
  ['lmS', 'lmS 10'], // log2(G) + lt + lB = 0 + 6 + 4
  ['kFD', 'kFD 14'], // dh=32: the KFD table includes u0 and ceil(4*Smax/rho).
]);
const policy = readFileSync(join(base, 'policy.txt'), 'utf8').split('\n').map(line => {
  const key = line.startsWith('work positions ') ? 'work positions' : line.split(' ')[0];
  return replacements.get(key) ?? line;
}).join('\n');
writeFileSync(join(out, 'policy.txt'), policy);
const original = readFileSync(join(base, 'KL.bin'));
const scales = Buffer.concat([original.subarray(0, 2 * 13 * 4), original.subarray(8 * 13 * 4, (8 * 13 + 3) * 4)]);
writeFileSync(join(out, 'KL.bin'), scales);
function run(name: string, cmd: string, args: string[], expected: number): string {
  const result = spawnSync(cmd, args, { cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
  writeFileSync(join(out, `${name}.log`), (result.stdout ?? '') + (result.stderr ?? ''));
  assert.equal(result.status, expected, `${name}: ${result.stdout} ${result.stderr}`);
  return result.stdout;
}
const tables = run('tables', 'python3', [join(root, 'upstream/scales/policy_tables.py'), join(out, 'policy.txt')], 0);
writeFileSync(join(out, 'policy.txt'), tables);
const arrays = run('arrays', verifier, ['-plan', 'arrays', out, '128'], 4);
writeFileSync(join(out, 'arrays.txt'), arrays);
// Execute the unmodified upstream layout packer; no new Python implementation.
const layout = run('layout', 'python3', [join(root, 'upstream/prover/tools/pack_layout.py'),
  '--arrays', join(out, 'arrays.txt'), '17', '4'], 0);
writeFileSync(join(out, 'layout.txt'), layout);
run('accounting', verifier, ['-plan', 'accounting', out, '128'], 4);
console.log(out);
