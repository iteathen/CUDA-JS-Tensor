import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);

test('foundation package is publication-guarded and owns no CUDA-JS or consumer dependency yet', async () => {
  const packageJson = JSON.parse(await readFile(new URL('package.json', root), 'utf8'));
  assert.equal(packageJson.name, 'cuda-js-tensor');
  assert.equal(packageJson.private, true);
  assert.equal(packageJson.dependencies, undefined);
  assert.equal(packageJson.exports, undefined);
});

test('deferred training-system plan authorizes no current repository mutation', async () => {
  const plan = await readFile(new URL('docs/integrations/the_restaurant.md', root), 'utf8');
  assert.match(plan, /Status:\*\* Deferred/);
  assert.match(plan, /No clone, branch, issue, code, dependency, Node upgrade, or setting change/);
});

