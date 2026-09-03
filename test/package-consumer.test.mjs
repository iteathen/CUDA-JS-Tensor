import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const root = path.resolve(import.meta.dirname, '..');

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', shell: false, env: { ...process.env, npm_config_audit: 'false', npm_config_fund: 'false' } });
  assert.equal(result.status, 0, `${command} ${args.join(' ')} failed\n${result.stdout}\n${result.stderr}`);
  return result.stdout;
}

test('packed package installs and an unrelated public consumer uses only canonical exports', { timeout: 120_000 }, async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'cuda-js-tensor-consumer-'));
  try {
    const npmCli = process.env.npm_execpath;
    const npmCommand = npmCli ? process.execPath : (process.platform === 'win32' ? 'npm.cmd' : 'npm');
    const npmPrefix = npmCli ? [npmCli] : [];
    run(npmCommand, [...npmPrefix, 'pack', '--json', '--pack-destination', directory], root);
    const tarball = (await readdir(directory)).find((name) => name.endsWith('.tgz'));
    assert(tarball, 'npm pack did not produce a tarball');
    await writeFile(path.join(directory, 'package.json'), JSON.stringify({ name: 'tensor-unrelated-consumer', private: true, type: 'module' }, null, 2));
    run(npmCommand, [...npmPrefix, 'install', '--ignore-scripts', '--package-lock=false', path.join(directory, tarball)], directory);
    await writeFile(path.join(directory, 'consumer.mjs'), `
import { compileDeviceProgram } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { compileTensorDeviceProgram, CUDA_JS_TENSOR_COMPATIBILITY, resolveTensorPlan, TENSOR_BACKEND_POLICIES, TensorPlan, TensorProgram, TensorSession, TensorSpec } from 'cuda-js-tensor';
let runtimeClosed = false;
const runtime = {
  async describe() { return { package: { name: 'cuda-js', version: '0.1.0-alpha.17', publicApiSchema: 1 }, state: 'open', profile: 'consumer-double', device: null }; },
  async allocateDevice({ byteLength }) {
    return {
      async view(options) { return { ...options, async status() { return { state: 'open' }; }, async close() { return { state: 'closed' }; } }; },
      async close() { return { state: 'closed', byteLength }; },
    };
  },
  async close() { runtimeClosed = true; return { graceful: true }; },
};
const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
const spec = TensorSpec.create('f32', [2, 3]);
const tensor = await session.allocate(spec);
if (tensor.byteLength !== 24 || tensor.spec.compatibilityIdentity !== spec.compatibilityIdentity) throw new Error('tensor contract mismatch');
const program = TensorProgram.define((graph) => graph.copy(graph.input('input', spec)));
const plan = TensorPlan.create(program);
if (plan.totalDistinctBytes !== 24 || plan.executable !== false || !plan.unresolved.includes('backend-selection')) throw new Error('static plan contract mismatch');
const emptyProgram = TensorProgram.define((graph) => graph.fill({ dtype: 'f32', capacityShape: [0] }, 0));
const resolved = await resolveTensorPlan(session, emptyProgram);
const result = await resolved.run();
if (result.output.capacityShape[0] !== 0 || result.execution.realization !== 'empty' || resolved.backend !== 'simt' || resolved.backendPolicy !== 'simt' || !TENSOR_BACKEND_POLICIES.includes('prefer-cublaslt')) throw new Error('resolved execution contract mismatch');
if (!(await result.close()).graceful || !(await resolved.close()).graceful) throw new Error('resolved cleanup contract mismatch');
const compilerRuntime = await openCudaRuntimeForTesting({ compiler: true });
const compilerSession = await TensorSession.open(compilerRuntime);
const itemProgram = TensorProgram.define((graph) => graph.unary('neg', graph.input('items', { dtype: 'f32', capacityShape: [2, 3], access: 'read' })));
const callable = await compileTensorDeviceProgram(compilerSession, itemProgram, { itemCapacity: 2, itemInputs: ['items'] });
const pointers = callable.function.parameters.slice(1);
const consumerParameters = [...pointers, { name: 'status', type: 'ptr<u32>' }];
const consumerSource = \`function consume(\${consumerParameters.map((entry) => entry.name).join(', ')}) { status[gpu.u32(0)] = runItem(gpu.u32(0), \${pointers.map((entry) => entry.name).join(', ')}); }\`;
const composed = await compileDeviceProgram(compilerRuntime, {
  source: consumerSource,
  functions: [{ name: 'consume', kind: 'kernel', parameters: consumerParameters, returns: 'void' }],
  imports: [callable.importAs('runItem')],
});
if (composed.linker.artifact.format !== 'cubin' || callable.totalWorkspaceBytes < 1 || callable.itemCapacity !== 2) throw new Error('device-callable package contract mismatch');
if (!(await compilerSession.close()).graceful || !(await compilerRuntime.close()).graceful) throw new Error('device-callable compiler cleanup mismatch');
try {
  await import('cuda-js-tensor/components/tensor-value/internal.mjs');
  throw new Error('package-internal port escaped');
} catch (error) {
  if (error.code !== 'ERR_PACKAGE_PATH_NOT_EXPORTED') throw error;
}
const terminal = await session.close();
if (!terminal.graceful || !runtimeClosed || CUDA_JS_TENSOR_COMPATIBILITY.package.version !== '0.1.0-alpha.6' || CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version !== '0.1.0-alpha.17') throw new Error('terminal contract mismatch');
console.log('installed CUDA-JS-Tensor consumer passed');
`);
    const output = run(process.execPath, [path.join(directory, 'consumer.mjs')], directory);
    assert.match(output, /installed CUDA-JS-Tensor consumer passed/);

    const installedPackage = JSON.parse(await readFile(path.join(directory, 'node_modules', 'cuda-js-tensor', 'package.json'), 'utf8'));
    assert.equal(installedPackage.version, '0.1.0-alpha.6');
    assert.deepEqual(Object.keys(installedPackage.exports), ['.']);
    const installedComponentEntries = await readdir(path.join(directory, 'node_modules', 'cuda-js-tensor', 'components', 'tensor-value'));
    assert.equal(installedComponentEntries.includes('test'), false);
    const installedProgramEntries = await readdir(path.join(directory, 'node_modules', 'cuda-js-tensor', 'components', 'tensor-program'));
    assert.equal(installedProgramEntries.includes('test'), false);
    const installedExecutionEntries = await readdir(path.join(directory, 'node_modules', 'cuda-js-tensor', 'components', 'tensor-execution'));
    assert.equal(installedExecutionEntries.includes('test'), false);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
