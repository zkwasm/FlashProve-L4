import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
assert.ok(process.argv[2], 'Usage: node scripts/check-cpu-commit.mts PATH_TO_CP_VERIFY');
const verifier = resolve(process.argv[2]);
const prepared = spawnSync(process.execPath, ['scripts/prepare-cpu-config.mts', verifier], { cwd: root, encoding: 'utf8' });
assert.equal(prepared.status, 0, prepared.stderr);
const config = prepared.stdout.trim();
console.log(`CPU commitment test: ${config}`);
const reports: { name: string; exitCode: number | null; seconds: number; lastLine: string }[] = [];
function run(name: string, cmd: string, args: string[], expected: number): string {
  console.log(`${name}: running`);
  const start = performance.now();
  const r = spawnSync(cmd, args, { cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
  writeFileSync(join(config, `${name}.log`), (r.stdout ?? '') + (r.stderr ?? ''));
  const lastLine = (r.stdout ?? '').trim().split('\n').at(-1) ?? '';
  reports.push({ name, exitCode: r.status, seconds: (performance.now() - start) / 1000, lastLine });
  writeFileSync(join(config, 'results.json'), JSON.stringify(reports, null, 2) + '\n');
  assert.equal(r.status, expected, `${name}: ${r.stdout} ${r.stderr}`);
  console.log(`${name}: exit=${r.status}, ${lastLine}`);
  return r.stdout;
}
const cpu = join(root, 'build/macos/cpu_commit');
const shape = run('inspect', cpu, [config, '--inspect', 'unused'], 0)
  .match(/cpu-shape rows=(\d+) columns=(\d+) encoded-bytes-per-table=(\d+)/);
assert.ok(shape);
const size = Number(shape[1]) * Number(shape[2]);
assert.ok(Number.isSafeInteger(size) && size <= 64 * 1024 * 1024);
const matrix = Buffer.alloc(size);
for (let i = 0; i < matrix.length; i++) matrix.writeInt8(i % 3 - 1, i);
const weights = join(config, 'matrix.bin');
writeFileSync(weights, matrix);
const ivk = join(config, 'ivk');
const commit = run('commit', cpu, [config, weights, ivk], 0);
// Use the ID printed by our committer, not an ID supplied with an unknown proof.
const id = commit.match(/^commitment ID: ([a-f0-9]{64})$/m)?.[1];
assert.ok(id);
assert.equal(readFileSync(join(ivk, 'ID.txt'), 'utf8'), id);
function verify(name: string, kind: 'key' | 'leaf', file: string, commitment = id!, expected = 3): string {
  const args = kind === 'key' ? ['-key', config, file] : ['-leaf', config, join(ivk, 'key.bin'), file];
  return run(name, verifier, [...args, '-id', commitment, '-q'], expected);
}
const key = join(ivk, 'key.bin'), leaf = join(ivk, 'leaf/0.bin');
assert.match(verify('verify-key', 'key', key), /\nPART PASSED: key\s*$/);
assert.match(verify('verify-leaf', 'leaf', leaf), /\nPART PASSED: leaf\s*$/);
for (const [kind, original] of [['key', key], ['leaf', leaf]] as const) {
  const bytes = readFileSync(original);
  bytes[Math.floor(bytes.length / 2)] ^= 1;
  const changed = join(config, `changed-${kind}.bin`);
  writeFileSync(changed, bytes);
  assert.match(verify(`reject-${kind}`, kind, changed, id, 1), /\nNO: [^\n]+\s*$/);
}
const wrongId = (id[0] === '0' ? '1' : '0') + id.slice(1);
assert.match(verify('reject-id', 'key', key, wrongId, 1), /\nNO: model-statement\s*$/);
// Every digit fits a signed byte, but a pair 127 + 256*127 exceeds the
// configured signed 13-bit weight range. WF must refuse to create a key.
writeFileSync(join(config, 'out-of-range.bin'), Buffer.alloc(size, 127));
run('reject-weight-range', cpu, [config, join(config, 'out-of-range.bin'), join(config, 'bad-ivk')], 2);
assert.equal(existsSync(join(config, 'bad-ivk/key.bin')), false);
console.log('PASS: newly generated CPU key and leaf; not a model inference or request proof.');
