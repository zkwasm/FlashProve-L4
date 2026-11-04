import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { verifierSourceDigest } from './source-digest.mts';
assert.equal(process.platform, 'darwin', 'This wrapper targets Apple Clang; Linux uses upstream/verifier/build.sh');
mkdirSync('build/macos', { recursive: true });
const digest = verifierSourceDigest(resolve('upstream'));
const result = spawnSync('clang++', ['-std=c++17', '-O2', '-Wall', '-Wextra', '-fstack-protector-strong',
  '-ftrivial-auto-var-init=zero', '-DBLK_FIELD=2', '-DBLK_HASH=1', '-Iblocks', '-pthread',
  '-Wuninitialized', '-Werror=unused-result', '-Wno-misleading-indentation',
  '-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE', '-DBLK_COUNT_HASH_CALLS=1',
  '-DSERVING_BUILD=1', '-DDIGEST_WORDS=8', `-DVF_SOURCE_DIGEST="${digest}"`,
  '-Iverifier/src', '-Iprover', 'verifier/src/cp_verify.cpp', '-o', resolve('build/macos/cp_verify')],
{ cwd: 'upstream', encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
writeFileSync('build/macos/verifier-build.log', (result.stdout ?? '') + (result.stderr ?? ''));
assert.equal(result.status, 0, 'verifier compilation failed: see build/macos/verifier-build.log');
writeFileSync('build/macos/verifier-source-digest.txt', digest + '\n');
console.log(`Built independent CPU verifier: build/macos/cp_verify; build=${digest}`);
