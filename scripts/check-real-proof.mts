import assert from 'node:assert/strict';
import { cpSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
const [verifierArg, artifactArg, configArg] = process.argv.slice(2);
assert(verifierArg && artifactArg && configArg, 'usage: node scripts/check-real-proof.mts CP_VERIFY ARTIFACT CONFIG');
const verifier = resolve(verifierArg), artifact = resolve(artifactArg), config = resolve(configArg);
for (const stage of ['commit', 'prove', 'verify'])
  assert.equal(readFileSync(join(artifact, `${stage}.exit`), 'utf8').trim(), '0', `${stage} did not finish successfully on L4`);
// Trust only the ID printed by our own committer; do not read one supplied inside ivk.
const commitLog = readFileSync(join(artifact, 'commit.log'), 'utf8');
const ids = [...commitLog.matchAll(/^commitment ID: ([0-9a-f ]+)$/gm)].map(m => m[1].replaceAll(' ', ''));
assert.equal(ids.length, 1, 'committer must print exactly one commitment ID');
const id = ids[0]; assert.match(id, /^[0-9a-f]{64}$/);
const out = join(artifact, 'verification'); mkdirSync(out);
const req = join(artifact, 'in/example.req'), y = join(artifact, 'out/example.y');
const proof = join(artifact, 'out/example.proof'), ivk = join(artifact, 'ivk');
const results: Record<string, unknown>[] = [];
function check(name: string, key = ivk, request = req, answer = y, p = proof, commitment = id, expected = 1) {
  const start = performance.now();
  const run = spawnSync(verifier, [config, key, request, answer, p, '-id', commitment],
    { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
  const log = (run.stdout ?? '') + (run.stderr ?? '');
  writeFileSync(join(out, `${name}.log`), log);
  const last = log.trimEnd().split('\n').at(-1);
  results.push({ name, exit: run.status, last, seconds: (performance.now() - start) / 1000 });
  assert.equal(run.status, expected, `${name}: ${log}`);
  assert(expected === 0 ? last === 'YES' : last?.startsWith('NO:'), `${name}: ${last}`);
}
check('honest', ivk, req, y, proof, id, 0);
const reference = JSON.parse(readFileSync(join(config, 'integer-reference.json'), 'utf8'));
assert.equal(readFileSync(y).readUInt32LE(), reference.nextToken, 'proved output differs from CPU integer inference');
function changed(name: string, source: string, edit: (b: Buffer) => void) {
  const bytes = Buffer.from(readFileSync(source)); edit(bytes);
  const path = join(out, name); writeFileSync(path, bytes); return path;
}
const badReq = changed('changed.req', req, b => b.writeUInt32LE((b.readUInt32LE(7 * 4) + 1) % 512, 7 * 4));
check('changed-input', ivk, badReq);
const badY = changed('changed.y', y, b => b.writeUInt32LE((b.readUInt32LE() + 1) % 512));
check('changed-output', ivk, req, badY);
const badProof = changed('changed.proof', proof, b => { b[Math.floor(b.length / 2)] ^= 1; });
check('changed-proof', ivk, req, y, badProof);
check('changed-commitment-id', ivk, req, y, proof, (id[0] === '0' ? '1' : '0') + id.slice(1));
const badKey = join(out, 'changed-ivk'); cpSync(ivk, badKey, { recursive: true });
const keyFile = join(badKey, 'key.bin'), bytes = Buffer.from(readFileSync(keyFile));
bytes[Math.floor(bytes.length / 2)] ^= 1; writeFileSync(keyFile, bytes);
check('changed-key', badKey);
writeFileSync(join(out, 'results.json'), JSON.stringify({ id, nextToken: reference.nextToken, results }, null, 2) + '\n');
console.log(JSON.stringify({ id, nextToken: reference.nextToken, results }, null, 2));
console.log('PASS: fresh real-model inference proof and five independent tampering rejections');
