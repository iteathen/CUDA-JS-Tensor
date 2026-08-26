import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { copyFile, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const node = process.env.CUDA_JS_TENSOR_NODE ?? process.execPath;
const npmCli = process.env.CUDA_JS_TENSOR_NPM_CLI ?? process.env.npm_execpath;
assert.equal(process.version, 'v26.7.0', 'Native Tensor conformance requires exact Node v26.7.0.');
assert(npmCli, 'Native Tensor conformance requires an explicit npm CLI path.');

function run(args, cwd) {
  const result = spawnSync(node, args, { cwd, encoding: 'utf8', shell: false, env: { ...process.env, CUDA_JS_TENSOR_NODE: node, CUDA_JS_TENSOR_NPM_CLI: npmCli, npm_config_audit: 'false', npm_config_fund: 'false' } });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `${node} ${args.join(' ')} failed\n${result.stdout}\n${result.stderr}`);
  return result.stdout.trim();
}

const directory = await mkdtemp(path.join(tmpdir(), 'cuda-js-tensor-native-'));
try {
  run([npmCli, 'pack', '--json', '--pack-destination', directory], root);
  const tarball = (await readdir(directory)).find((name) => name.endsWith('.tgz'));
  assert(tarball, 'npm pack did not produce a CUDA-JS-Tensor tarball.');
  await writeFile(path.join(directory, 'package.json'), `${JSON.stringify({ name: 'cuda-js-tensor-native-consumer', private: true, type: 'module' }, null, 2)}\n`);
  await copyFile(path.join(root, 'conformance', 'native', 'fixtures', 'resolved-simt-consumer.mjs'), path.join(directory, 'consumer.mjs'));
  run([npmCli, 'install', '--ignore-scripts', '--package-lock=false', path.join(directory, tarball)], directory);
  const installed = JSON.parse(await readFile(path.join(directory, 'node_modules', 'cuda-js-tensor', 'package.json'), 'utf8'));
  assert.equal(installed.version, '0.1.0-alpha.4');
  const output = run(['--experimental-ffi', 'consumer.mjs'], directory);
  const observation = JSON.parse(output.split(/\r?\n/).at(-1));
  assert.equal(observation.consumer, 'installed-native-resolved-dense');
  assert.equal(observation.sessionGraceful, true);
  assert.equal(observation.runtimeGraceful, true);
  assert.equal(observation.summary.composite.replays, 2);
  assert.deepEqual(observation.summary.composite.compiler, { architecture: 'compute_75', headerProfile: 'cuda-numeric' });
  assert.deepEqual({ backend: observation.summary.accelerated.backend, kernels: observation.summary.accelerated.kernels, cublasLtNodes: observation.summary.accelerated.cublasLtNodes, replays: observation.summary.accelerated.replays }, { backend: 'mixed', kernels: 2, cublasLtNodes: 1, replays: 2 });
  assert.deepEqual(observation.summary.transposedAccelerated, { backend: 'cublaslt', cublasLtNodes: 1, output: [58, 64, 139, 154] });
  assert.deepEqual(observation.summary.cast, [0, 2_147_483_647, -2_147_483_648, 3, -3]);
  assert.deepEqual(observation.summary.fixedTree.output, [0]);
  console.log(JSON.stringify(observation));
} finally {
  await rm(directory, { recursive: true, force: true });
}
