import assert from 'node:assert/strict';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
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

test('packed public package compiles accepted device-callable gather and concat child through CUDA-JS', { timeout: 120_000 }, async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'cuda-js-tensor-gather-concat-'));
  try {
    const npmCli = process.env.npm_execpath;
    const npmCommand = npmCli ? process.execPath : (process.platform === 'win32' ? 'npm.cmd' : 'npm');
    const npmPrefix = npmCli ? [npmCli] : [];
    run(npmCommand, [...npmPrefix, 'pack', '--json', '--pack-destination', directory], root);
    const tarball = (await readdir(directory)).find((name) => name.endsWith('.tgz'));
    assert(tarball, 'npm pack did not produce a tarball');
    await writeFile(path.join(directory, 'package.json'), JSON.stringify({ name: 'tensor-gather-concat-consumer', private: true, type: 'module' }, null, 2));
    run(npmCommand, [...npmPrefix, 'install', '--ignore-scripts', '--package-lock=false', path.join(directory, tarball)], directory);
    await writeFile(path.join(directory, 'consumer.mjs'), `
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { compileTensorDeviceProgram, TENSOR_DEVICE_GATHER_CONCAT_CONTRACT, TensorProgram, TensorSession } from 'cuda-js-tensor';

const runtime = await openCudaRuntimeForTesting({ compiler: true });
const session = await TensorSession.open(runtime);
try {
  const gather = TensorProgram.define((graph) => graph.gather(graph.input('items', { dtype: 'f32', capacityShape: [4, 3, 4], access: 'read' }), 1, [2, 0, 2]));
  const gatherCompiled = await compileTensorDeviceProgram(session, gather, { itemCapacity: 4, itemInputs: ['items'] });
  if (gatherCompiled.contract !== TENSOR_DEVICE_GATHER_CONCAT_CONTRACT || gatherCompiled.canonical.contract !== TENSOR_DEVICE_GATHER_CONCAT_CONTRACT) throw new Error('gather child contract mismatch');
  if (gatherCompiled.function.name !== 'tensorRunItem' || gatherCompiled.function.returns !== 'u32') throw new Error('gather callable ABI mismatch');

  const concat = TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4, 3], access: 'read' });
    return graph.concat([left, right], 1);
  });
  const concatCompiled = await compileTensorDeviceProgram(session, concat, { itemCapacity: 4, itemInputs: ['left', 'right'] });
  if (concatCompiled.contract !== TENSOR_DEVICE_GATHER_CONCAT_CONTRACT) throw new Error('concat child contract mismatch');
  if (concatCompiled.parameters.some((entry) => !['item-index', 'input', 'output', 'workspace'].includes(entry.role))) throw new Error('unexpected concat ABI role');
} finally {
  if (!(await session.close()).graceful) throw new Error('TensorSession cleanup failed');
  if (!(await runtime.close()).graceful) throw new Error('CUDA-JS runtime cleanup failed');
}
console.log('installed device-callable gather/concat consumer passed');
`);
    const output = run(process.execPath, [path.join(directory, 'consumer.mjs')], directory);
    assert.match(output, /installed device-callable gather\/concat consumer passed/u);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
