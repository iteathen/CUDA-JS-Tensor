import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { CUDA_JS_TENSOR_COMPATIBILITY } from 'cuda-js-tensor';

const EXPECTED_CUDA_JS_REVISION = '45a9ef15537b52d6fd7c615b7e596676dfd00587';
const EXPECTED_DEPENDENCY = `https://codeload.github.com/iteathen/CUDA-JS/tar.gz/${EXPECTED_CUDA_JS_REVISION}`;

const root = new URL('../', import.meta.url);

test('Tensor pins the protected CUDA-JS bounded-source compatible pair everywhere it owns revision identity', async () => {
  const packageJson = JSON.parse(await readFile(new URL('package.json', root), 'utf8'));
  const packageLock = JSON.parse(await readFile(new URL('package-lock.json', root), 'utf8'));

  assert.equal(packageJson.dependencies['cuda-js'], EXPECTED_DEPENDENCY);
  assert.equal(packageLock.packages[''].dependencies['cuda-js'], EXPECTED_DEPENDENCY);
  assert.equal(packageLock.packages['node_modules/cuda-js'].resolved, EXPECTED_DEPENDENCY);
  assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.name, 'cuda-js');
  assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version, '0.1.0-alpha.18');
  assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.publicApiSchema, 1);
  assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.protectedMainRevision, EXPECTED_CUDA_JS_REVISION);
});
