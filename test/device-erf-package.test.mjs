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

test('packed public package compiles device-callable unary erf through installed cuda-js', { timeout: 120_000 }, async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'cuda-js-tensor-erf-consumer-'));
  try {
    const npmCli = process.env.npm_execpath;
    const npmCommand = npmCli ? process.execPath : (process.platform === 'win32' ? 'npm.cmd' : 'npm');
    const npmPrefix = npmCli ? [npmCli] : [];
    run(npmCommand, [...npmPrefix, 'pack', '--json', '--pack-destination', directory], root);
    const tarball = (await readdir(directory)).find((name) => name.endsWith('.tgz'));
    assert(tarball, 'npm pack did not produce a tarball');
    await writeFile(path.join(directory, 'package.json'), JSON.stringify({ name: 'tensor-erf-consumer', private: true, type: 'module' }, null, 2));
    run(npmCommand, [...npmPrefix, 'install', '--ignore-scripts', '--package-lock=false', path.join(directory, tarball)], directory);
    await writeFile(path.join(directory, 'consumer.mjs'), `
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { compileTensorDeviceProgram, TensorProgram, TensorSession } from 'cuda-js-tensor';

const runtime = await openCudaRuntimeForTesting({ compiler: true });
const session = await TensorSession.open(runtime);
try {
  for (const dtype of ['f32', 'f64']) {
    const program = TensorProgram.define((graph) => graph.unary('erf', graph.input('items', { dtype, capacityShape: [4, 2], access: 'read' })));
    const compiled = await compileTensorDeviceProgram(session, program, { itemCapacity: 4, itemInputs: ['items'] });
    if (compiled.function.name !== 'tensorRunItem' || compiled.function.returns !== 'u32') throw new Error('unexpected callable ABI');
    if (!compiled.library.contract.includes('SPEC-0030-erf-v1')) throw new Error('installed public CUDA-JS erf contract was not selected');
  }
} finally {
  if (!(await session.close()).graceful) throw new Error('TensorSession cleanup failed');
  if (!(await runtime.close()).graceful) throw new Error('CUDA-JS runtime cleanup failed');
}
console.log('installed device-callable erf consumer passed');
`);
    const output = run(process.execPath, [path.join(directory, 'consumer.mjs')], directory);
    assert.match(output, /installed device-callable erf consumer passed/u);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
