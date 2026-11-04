import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Build the unchanged upstream sources on macOS. This checks synthetic examples,
// not a real pretrained model or a CPU model prover.
assert.equal(process.platform, 'darwin', 'This baseline build targets macOS/Apple Clang.');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = join(root, 'upstream');
mkdirSync(join(root, 'runs'), { recursive: true });
const out = mkdtempSync(join(root, 'runs', 'baseline-'));
const reports: { name: string; exitCode: number | null; seconds: number; lastLine: string }[] = [];
console.log(`Logs and binaries: ${out}`);

function run(name: string, command: string, args: string[], expected = 0, cwd = source): string {
  console.log(`${name}: running`);
  const start = performance.now();
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
  const stdout = result.stdout ?? '';
  const stderr = result.stderr ?? '';
  writeFileSync(join(out, `${name}.log`), stdout + stderr);
  const lastLine = stdout.trim().split('\n').at(-1) ?? '';
  const report = { name, exitCode: result.status, seconds: (performance.now() - start) / 1000, lastLine };
  reports.push(report);
  writeFileSync(join(out, 'results.json'), JSON.stringify(reports, null, 2) + '\n');
  console.log(`${name}: exit=${result.status}, ${report.seconds.toFixed(2)}s, ${lastLine}`);
  if (result.error) throw result.error;
  assert.equal(result.status, expected, `${name}: see ${join(out, `${name}.log`)}`);
  return stdout;
}

function sha(bytes: Buffer | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

// Reproduce verifier/build.sh's source digest, including sha256sum's filenames
// and two-space separator. No platform-dependent shell hashing tools are needed.
const verifierDir = join(source, 'verifier', 'src');
const first = readdirSync(verifierDir).sort();
for (const part of readdirSync(join(source, 'prover'), { withFileTypes: true }).sort((a, b) => a.name < b.name ? -1 : 1)) {
  if (part.isDirectory()) {
    const entries = readdirSync(join(source, 'prover', part.name));
    if (entries.includes('plan_rounds.h')) first.push(`../../prover/${part.name}/plan_rounds.h`);
  }
}
first.push('../../prover/hiding/token_plan.h');
function headers(dir: string, prefix = '.'): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const name = `${prefix}/${entry.name}`;
    if (entry.isDirectory()) return entry.name === 'tests' ? [] : headers(join(dir, entry.name), name);
    return entry.name.endsWith('.h') ? [name] : [];
  });
}
const blockHeaders = headers(join(source, 'blocks')).sort();
const sums = first.map(name => `${sha(readFileSync(resolve(verifierDir, name)))}  ${name}\n`).join('')
  + blockHeaders.map(name => `${sha(readFileSync(resolve(source, 'blocks', name)))}  ${name}\n`).join('');
const digest = sha(sums);
writeFileSync(join(out, 'source-digest.txt'), digest + '\n');
run('revision', '/usr/bin/git', ['rev-parse', 'HEAD']);
run('compiler', 'clang++', ['--version']);

const flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-fstack-protector-strong',
  '-ftrivial-auto-var-init=zero', '-DBLK_FIELD=2', '-DBLK_HASH=1', '-Iblocks'];
for (const [name, file] of [
  ['field_test', 'blocks/field/tests/field_test.cpp'],
  ['engine_test', 'blocks/engine/tests/engine_test.cpp'],
  ['handover_proof', 'blocks/frame/handover_proof.cpp'],
] as const) {
  run(`build-${name}`, 'clang++', [...flags, file, '-o', join(out, name)]);
}
run('build-cp_verify', 'clang++', [...flags, '-pthread', '-Wuninitialized', '-Werror=unused-result',
  '-Wno-misleading-indentation', '-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE',
  '-DBLK_COUNT_HASH_CALLS=1', '-DSERVING_BUILD=1', '-DDIGEST_WORDS=8',
  `-DVF_SOURCE_DIGEST="${digest}"`, '-Iverifier/src', '-Iprover',
  'verifier/src/cp_verify.cpp', '-o', join(out, 'cp_verify')]);
assert.match(run('field', join(out, 'field_test'), []), /: ok\s*$/);
assert.match(run('engine', join(out, 'engine_test'), []), /engine: ok\s*$/);
const toy = join(out, 'toy');
mkdirSync(join(toy, 'on-the-first-table'), { recursive: true });
mkdirSync(join(toy, 'on-the-copy'));
assert.match(run('toy', join(out, 'handover_proof'), [toy]),
  /54 altered copies and calls, 0 accepted; 0 honest files refused\s*$/);

const example = join(source, 'examples', 'accepted-proof');
for (const line of readFileSync(join(example, 'SHA256SUMS'), 'utf8').trim().split('\n')) {
  const [expected, name] = line.split('  ');
  assert.equal(sha(readFileSync(join(example, name))), expected, `Example checksum: ${name}`);
}
const id = readFileSync(join(example, 'ID.txt'), 'utf8').trim();
const request = join(example, 'example.req');
const reply = join(example, 'example.y');
const proof = join(example, 'example.proof');
function verify(name: string, req = request, y = reply, p = proof, commitment = id, expected = 0): string {
  return run(name, join(out, 'cp_verify'), [join(example, 'config'), join(example, 'ivk'),
    req, y, p, '-id', commitment, '-q'], expected);
}
assert.match(verify('verify-honest'), /\nYES\s*$/);
function changed(name: string, path: string, offset: number): string {
  const bytes = readFileSync(path);
  bytes[offset] ^= 1;
  const target = join(out, name);
  writeFileSync(target, bytes);
  return target;
}
const cases: [string, string, string, string, string][] = [
  ['request', changed('changed.req', request, 0), reply, proof, id],
  ['output', request, changed('changed.y', reply, 0), proof, id],
  ['proof', request, reply, changed('changed.proof', proof, Math.floor(readFileSync(proof).length / 2)), id],
  ['commitment', request, reply, proof, (id[0] === '0' ? '1' : '0') + id.slice(1)],
];
for (const [name, req, y, p, commitment] of cases) {
  assert.match(verify(`reject-${name}`, req, y, p, commitment, 1), /\nNO: [^\n]+\s*$/);
}
console.log(`PASS: synthetic baseline only. Results: ${relative(root, join(out, 'results.json'))}`);
