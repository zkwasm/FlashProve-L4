import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { cpSync, mkdtempSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
process.chdir(root);
const manifest = JSON.parse(readFileSync('MANIFEST.json', 'utf8'));
for (const [path, expected] of Object.entries(manifest.files)) {
  const actual = createHash('sha256').update(readFileSync(path)).digest('hex');
  assert.equal(actual, expected, `published file checksum mismatch: ${path}`);
}
console.log(`PASS: ${Object.keys(manifest.files).length} published file checksums`);
function run(command: string, args: string[]) {
  const result = spawnSync(command, args, { stdio: 'inherit' });
  assert.equal(result.status, 0, `${command} failed`);
}
let verifier = process.argv[2] && resolve(process.argv[2]);
if (!verifier) {
  if (process.platform === 'darwin') {
    run(process.execPath, ['scripts/build-verifier.mts']);
    verifier = resolve('build/macos/cp_verify');
  } else {
    assert.equal(process.platform, 'linux', 'Use macOS or Linux, or supply a compatible verifier');
    run('bash', ['upstream/verifier/build.sh']);
    verifier = resolve('upstream/bin/cp_verify');
  }
}
const work = mkdtempSync('runs/recheck-');
for (const path of ['ivk', 'in', 'out', 'commit.log', 'commit.exit', 'prove.exit', 'verify.exit'])
  cpSync(join('runs/real-l4-v8', path), join(work, path), { recursive: true });
run(process.execPath, ['scripts/check-real-proof.mts', verifier, work, 'runs/real-config-tu9cL1']);
console.log(`Fresh verification logs: ${work}/verification/`);
console.log('The manifest detects file changes; it does not authenticate the repository publisher.');
