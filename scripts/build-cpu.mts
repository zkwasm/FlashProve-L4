import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifierSourceDigest } from './source-digest.mts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'build/macos');
assert.equal(process.platform, 'darwin');
mkdirSync(out, { recursive: true });
const digest = verifierSourceDigest(join(root, 'upstream'));
const flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Wno-misleading-indentation',
  '-fstack-protector-strong', '-ftrivial-auto-var-init=zero',
  '-DBLK_FIELD=2', '-DBLK_HASH=1', '-DSERVING_BUILD=1', '-DBLK_COUNT_HASH_CALLS=1',
  '-DDIGEST_WORDS=8', `-DVF_SOURCE_DIGEST="${digest}"`,
  '-Iupstream/blocks', '-Iupstream/verifier/src', '-Iupstream/prover'];
const commands = [
  [...flags, '-DCP_VERIFY_AS_A_UNIT', '-c', 'upstream/verifier/src/cp_verify.cpp', '-o', join(out, 'verifier_unit.o')],
  [...flags, '-c', 'cpu/commit.cpp', '-o', join(out, 'cpu_commit.o')],
  [join(out, 'cpu_commit.o'), join(out, 'verifier_unit.o'), '-o', join(out, 'cpu_commit')],
];
for (const [i, args] of commands.entries()) {
  console.log(`CPU build ${i + 1}/${commands.length}`);
  const result = spawnSync('clang++', args, { cwd: root, encoding: 'utf8' });
  writeFileSync(join(out, `cpu-build-${i + 1}.log`), (result.stdout ?? '') + (result.stderr ?? ''));
  assert.equal(result.status, 0, result.stderr);
}
writeFileSync(join(out, 'cpu-source-digest.txt'), digest + '\n');
console.log(join(out, 'cpu_commit'));
