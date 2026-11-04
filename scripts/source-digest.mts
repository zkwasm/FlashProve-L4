import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

export function verifierSourceDigest(source: string): string {
  const hash = (bytes: Buffer | string) => createHash('sha256').update(bytes).digest('hex');
  const dir = join(source, 'verifier/src');
  const files = readdirSync(dir).sort();
  for (const part of readdirSync(join(source, 'prover'), { withFileTypes: true }).sort((a, b) => a.name < b.name ? -1 : 1)) {
    if (part.isDirectory() && readdirSync(join(source, 'prover', part.name)).includes('plan_rounds.h'))
      files.push(`../../prover/${part.name}/plan_rounds.h`);
  }
  files.push('../../prover/hiding/token_plan.h');
  function headers(at: string, prefix = '.'): string[] {
    return readdirSync(at, { withFileTypes: true }).flatMap(e => e.isDirectory()
      ? e.name === 'tests' ? [] : headers(join(at, e.name), `${prefix}/${e.name}`)
      : e.name.endsWith('.h') ? [`${prefix}/${e.name}`] : []);
  }
  // Exactly verifier/build.sh's sha256sum listing, including relative filenames.
  const sums = files.map(name => `${hash(readFileSync(resolve(dir, name)))}  ${name}\n`).join('')
    + headers(join(source, 'blocks')).sort().map(name =>
      `${hash(readFileSync(resolve(source, 'blocks', name)))}  ${name}\n`).join('');
  return hash(sums);
}
