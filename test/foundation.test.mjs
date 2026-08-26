import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { CUDA_JS_TENSOR_COMPATIBILITY, TensorPlan, TensorProgram } from 'cuda-js-tensor';

const root = new URL('../', import.meta.url);

test('foundation package is publication-guarded and depends only on the exact public CUDA-JS revision', async () => {
  const packageJson = JSON.parse(await readFile(new URL('package.json', root), 'utf8'));
  assert.equal(packageJson.name, 'cuda-js-tensor');
  assert.equal(packageJson.private, true);
  assert.deepEqual(Object.keys(packageJson.dependencies), ['cuda-js']);
  assert.equal(packageJson.dependencies['cuda-js'], 'https://codeload.github.com/iteathen/CUDA-JS/tar.gz/af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d');
  assert.equal(packageJson.exports['.'].import, './components/public-api/index.mjs');
  assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.package.version, packageJson.version);
  assert.equal(typeof TensorProgram.create, 'function');
  assert.equal(typeof TensorPlan.create, 'function');
});

test('deferred training-system plan authorizes no current repository mutation', async () => {
  const plan = await readFile(new URL('docs/integrations/the_restaurant.md', root), 'utf8');
  assert.match(plan, /Status:\*\* Deferred/);
  assert.match(plan, /No clone, branch, issue, code, dependency, Node upgrade, or setting change/);
});
