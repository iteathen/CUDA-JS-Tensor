import assert from 'node:assert/strict';
import { TensorSession } from 'cuda-js-tensor';

const session = await TensorSession.open();
let terminal;
try {
  const tensor = await session.allocate({ dtype: 'f32', capacityShape: [2, 3] });
  const status = await tensor.status();
  assert.equal(status.spec.byteLength, 24);
  assert.equal(status.deviceViewState, 'open');
} finally {
  terminal = await session.close();
}

assert.equal(terminal.graceful, true);
console.log(JSON.stringify({ schemaVersion: 1, claim: 'native-lifecycle-smoke-only', node: process.version, tensorBytes: 24, terminalState: terminal.state, graceful: terminal.graceful }));
