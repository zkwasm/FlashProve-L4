import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
const root = 'models/stories260K';
mkdirSync(root, { recursive: true }); mkdirSync('build/macos', { recursive: true });
const model = 'https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K';
const code = 'https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb';
for (const [name, url, hash] of [
  ['stories260K.bin', `${model}/stories260K.bin`, 'b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696'],
  ['tok512.bin', `${model}/tok512.bin`, '037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312'],
  ['reference-run.c', `${code}/run.c`, '9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd'],
  ['REFERENCE-LICENSE', `${code}/LICENSE`, 'e87b912002f04cdfa837eaf7f70548ab79b9cb62a285495edf575f3da875aec4'],
]) {
  const path = `${root}/${name}`;
  let bytes: Buffer;
  if (existsSync(path)) bytes = readFileSync(path);
  else {
    const response = await fetch(url);
    assert(response.ok, `download failed: ${name}: ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
  }
  assert.equal(createHash('sha256').update(bytes).digest('hex'), hash, `checksum mismatch: ${name}`);
  if (!existsSync(path)) writeFileSync(path, bytes, { flag: 'wx' });
}
const compile = spawnSync('clang', ['-O2', `${root}/reference-run.c`, '-lm', '-o', 'build/macos/stories260K-reference'], { stdio: 'inherit' });
assert.equal(compile.status, 0, 'original C reference compilation failed');
console.log('PASS: pinned public model, tokenizer, reference source and license checksums; C reference compiled');
