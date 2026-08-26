import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);

test('foundation package is publication-guarded and depends only on the exact public CUDA-JS revision', async () => {
  const packageJson = JSON.parse(await readFile(new URL('package.json', root), 'utf8'));
  assert.equal(packageJson.name, 'cuda-js-tensor');
  assert.equal(packageJson.private, true);
  assert.deepEqual(Object.keys(packageJson.dependencies), ['cuda-js']);
  assert.equal(packageJson.dependencies['cuda-js'], 'https://codeload.github.com/iteathen/CUDA-JS/tar.gz/2da65ff2e4287450171c477031dd380a21fa095f');
  assert.equal(packageJson.exports['.'].import, './components/tensor-value/index.mjs');
});

test('deferred training-system plan authorizes no current repository mutation', async () => {
  const plan = await readFile(new URL('docs/integrations/the_restaurant.md', root), 'utf8');
  assert.match(plan, /Status:\*\* Deferred/);
  assert.match(plan, /No clone, branch, issue, code, dependency, Node upgrade, or setting change/);
});
