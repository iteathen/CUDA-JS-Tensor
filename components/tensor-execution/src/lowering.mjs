import { tensorDtypeWidth } from '../../tensor-value/index.mjs';
import { TensorPlan } from '../../tensor-program/index.mjs';

import { checkedAdd, checkedMultiply, deepFreeze, fail, identity, TENSOR_SIMT_LIMITS } from './contract.mjs';

const FLOAT_DTYPES = new Set(['f16', 'bf16', 'f32', 'f64']);

function u64(value) { return `gpu.u64(${BigInt(value)}n)`; }

function scalar(record) {
  const { dtype, value } = record;
  if (value === 'Infinity') return `gpu.${dtype}.positiveInfinity()`;
  if (value === '-Infinity') return `gpu.${dtype}.negativeInfinity()`;
  if (value === '-0') return `gpu.${dtype}(-0)`;
  if (dtype === 'u64') return `gpu.u64(${value}n)`;
  return `gpu.${dtype}(${String(value)})`;
}

function cast(target, source, expression) {
  return target === source ? expression : `gpu.cast.${target}(${expression})`;
}

function combine(operator, dtype, left, right) {
  if (operator === 'minimum' || operator === 'maximum') return `gpu.math.${operator}(${left}, ${right})`;
  const token = { sum: '+', product: '*', add: '+', sub: '-', mul: '*', div: '/' }[operator];
  if (!token) fail('TENSOR_SIMT_OPERATOR_UNSUPPORTED', 'unsupported', 'SIMT lowering does not own the requested operator.', { operator });
  if (dtype === 'i32' && token !== '/') return `gpu.cast.i32(gpu.cast.u32(${left}) ${token} gpu.cast.u32(${right}))`;
  return `(${left} ${token} ${right})`;
}

function unary(operator, dtype, value) {
  if (operator === 'abs') return `gpu.math.abs(${value})`;
  if (operator === 'exp' || operator === 'log' || operator === 'sqrt') return `gpu.math.${operator}(${value})`;
  if (operator === 'neg') return dtype === 'i32' ? `gpu.cast.i32(gpu.u32(0) - gpu.cast.u32(${value}))` : `(-${value})`;
  fail('TENSOR_SIMT_OPERATOR_UNSUPPORTED', 'unsupported', 'SIMT lowering does not own the requested unary operator.', { operator });
}

function coordinates(index, shape, prefix) {
  const lines = [`let ${prefix}Remaining = ${index};`];
  const names = new Array(shape.length);
  for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
    names[axis] = `${prefix}${axis}`;
    lines.push(`let ${names[axis]} = ${prefix}Remaining % ${u64(shape[axis])};`);
    lines.push(`${prefix}Remaining = ${prefix}Remaining / ${u64(shape[axis])};`);
  }
  return { lines, names };
}

function offsetLines(name, spec, originByteOffset, coordinateExpressions) {
  const width = tensorDtypeWidth(spec.dtype);
  const relativeBytes = spec.byteOffset - originByteOffset;
  if (!Number.isSafeInteger(relativeBytes) || relativeBytes < 0 || relativeBytes % width !== 0) {
    fail('TENSOR_SIMT_VIEW_OFFSET_INVALID', 'validation', 'A planned view offset is not representable relative to its storage binding.');
  }
  const lines = [`let ${name} = ${u64(relativeBytes / width)};`];
  for (let axis = 0; axis < spec.rank; axis += 1) {
    const coordinate = coordinateExpressions[axis];
    if (coordinate !== '0' && spec.strides[axis] !== 0) lines.push(`${name} += ${coordinate} * ${u64(spec.strides[axis])};`);
  }
  return lines;
}

function broadcastCoordinates(inputSpec, outputCoordinates) {
  const rankOffset = outputCoordinates.length - inputSpec.rank;
  return inputSpec.logicalShape.map((dimension, axis) => dimension === 1 ? '0' : outputCoordinates[rankOffset + axis]);
}

function accessFor(reference, argumentIndex, mode) {
  const relative = reference.spec.byteOffset - reference.originByteOffset;
  return { argumentIndex, byteOffset: relative, byteLength: reference.spec.byteLength, mode };
}

function nextPowerOfTwo(value) {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

function reductionLayout(inputSpec, outputSpec, axes, keepDimensions) {
  const reduced = new Set(axes);
  const outputToInput = new Map();
  if (keepDimensions) {
    for (let axis = 0; axis < inputSpec.rank; axis += 1) if (!reduced.has(axis)) outputToInput.set(axis, axis);
  } else {
    let outputAxis = 0;
    for (let axis = 0; axis < inputSpec.rank; axis += 1) if (!reduced.has(axis)) outputToInput.set(axis, outputAxis++);
  }
  let reductionCount = 1;
  for (const axis of axes) reductionCount = checkedMultiply(reductionCount, inputSpec.logicalShape[axis], 'reductionCount');
  return { reduced, outputToInput, reductionCount, outputCount: outputSpec.logicalElementCount };
}

function materialReference(binding, spec, valueId) {
  return Object.freeze({ binding, spec, originByteOffset: 0, baseValueId: valueId });
}

function bindingRecord(name, role, dtype, byteLength, valueId = null) {
  return Object.freeze({ name, role, dtype, byteLength, valueId });
}

export function lowerSimtPlan(plan, { blockSize = 256, maxWorkspaceBytes = TENSOR_SIMT_LIMITS.maxWorkspaceBytes } = {}) {
  if (!(plan instanceof TensorPlan)) fail('TENSOR_RESOLVE_PLAN_INVALID', 'validation', 'SIMT lowering requires a TensorPlan.');
  const program = plan.program;
  const references = new Map();
  const bindings = [];
  const workspaces = [];
  const functions = [];
  const kernels = [];
  let totalWorkspaceBytes = 0;

  function addBinding(role, dtype, byteLength, valueId = null) {
    const name = `b${bindings.length}`;
    bindings.push(bindingRecord(name, role, dtype, byteLength, valueId));
    if (bindings.length > TENSOR_SIMT_LIMITS.maxBindings) fail('TENSOR_SIMT_BINDING_LIMIT', 'pressure', 'Resolved SIMT execution exceeds the finite prepared-binding limit.', { maximum: TENSOR_SIMT_LIMITS.maxBindings });
    return name;
  }

  for (const input of program.inputs) {
    const binding = addBinding('input', input.spec.dtype, input.spec.byteLength, input.valueId);
    references.set(input.valueId, Object.freeze({ binding, spec: input.spec, originByteOffset: input.spec.byteOffset, baseValueId: input.valueId }));
  }

  for (const node of program.nodes) {
    if (node.materialization === 'view') {
      const source = references.get(node.inputIds[0]);
      references.set(node.id, Object.freeze({ ...source, spec: node.outputSpec }));
    } else {
      const binding = addBinding('material', node.outputSpec.dtype, Math.max(node.outputSpec.requiredByteEnd, node.outputSpec.alignment), node.id);
      references.set(node.id, materialReference(binding, node.outputSpec, node.id));
    }
  }

  let priorKernel = null;
  function addKernel(parameterRecords, bodyLines, workItems, accessRecords, semanticNode) {
    if (workItems === 0) return null;
    if (!Number.isSafeInteger(workItems) || workItems < 1 || workItems > TENSOR_SIMT_LIMITS.maxLogicalWorkItems) {
      fail('TENSOR_SIMT_WORK_LIMIT', 'pressure', 'A generated SIMT kernel exceeds the finite logical-work limit.', { workItems, maximum: TENSOR_SIMT_LIMITS.maxLogicalWorkItems });
    }
    if (kernels.length >= TENSOR_SIMT_LIMITS.maxKernels) fail('TENSOR_SIMT_KERNEL_LIMIT', 'pressure', 'Resolved SIMT execution exceeds the finite prepared-kernel limit.', { maximum: TENSOR_SIMT_LIMITS.maxKernels });
    const functionName = `tensorKernel${kernels.length}`;
    const id = `kernel${kernels.length}`;
    const parameters = parameterRecords.map((parameter, index) => ({ name: `p${index}`, type: `ptr<${parameter.dtype}>` }));
    functions.push(Object.freeze({ name: functionName, kind: 'kernel', parameters: Object.freeze(parameters), returns: 'void' }));
    const source = [`function ${functionName}(${parameters.map((entry) => entry.name).join(', ')}) {`, ...bodyLines.map((line) => `  ${line}`), '}'].join('\n');
    const kernel = Object.freeze({
      id,
      semanticNode,
      functionName,
      source,
      parameterRecords: Object.freeze(parameterRecords.map((entry) => Object.freeze({ ...entry }))),
      workItems,
      grid: Object.freeze({ x: Math.ceil(workItems / blockSize), y: 1, z: 1 }),
      block: Object.freeze({ x: blockSize, y: 1, z: 1 }),
      after: Object.freeze(priorKernel ? [priorKernel] : []),
      accesses: Object.freeze(accessRecords.map((entry) => Object.freeze({ ...entry }))),
    });
    kernels.push(kernel);
    priorKernel = id;
    return kernel;
  }

  for (const node of program.nodes) {
    if (node.materialization !== 'materialize') continue;
    const output = references.get(node.id);
    const outputCount = node.outputSpec.logicalElementCount;
    if (outputCount === 0) continue;
    const inputs = node.inputIds.map((id) => references.get(id));

    if (node.op === 'reduce') {
      const input = inputs[0];
      const layout = reductionLayout(input.spec, node.outputSpec, node.options.axes, node.options.keepDimensions);
      const padded = nextPowerOfTwo(Math.max(1, layout.reductionCount));
      const outputCoordinates = coordinates('outputIndex', node.outputSpec.logicalShape, 'o');
      const reductionCoordinates = coordinates('reductionIndex', node.options.axes.map((axis) => input.spec.logicalShape[axis]), 'r');
      const inputCoordinates = new Array(input.spec.rank).fill('0');
      for (let axis = 0; axis < input.spec.rank; axis += 1) {
        const reductionPosition = node.options.axes.indexOf(axis);
        if (reductionPosition >= 0) inputCoordinates[axis] = reductionCoordinates.names[reductionPosition];
        else inputCoordinates[axis] = outputCoordinates.names[layout.outputToInput.get(axis)];
      }
      const identityValue = scalar(node.options.identity);
      if (padded === 1) {
        const body = [
          'let outputIndex = gpu.cast.u64(gpu.thread.globalX());',
          `if (outputIndex < ${u64(layout.outputCount)}) {`,
          ...(layout.reductionCount === 0 ? [] : ['  let reductionIndex = gpu.u64(0n);', ...outputCoordinates.lines.map((line) => `  ${line}`), ...reductionCoordinates.lines.map((line) => `  ${line}`), ...offsetLines('inputOffset', input.spec, input.originByteOffset, inputCoordinates).map((line) => `  ${line}`)]),
          `  p1[outputIndex] = ${layout.reductionCount === 0 ? identityValue : cast(node.options.accumulatorDtype, input.spec.dtype, 'p0[inputOffset]')};`,
          '}',
        ];
        addKernel(
          [{ binding: input.binding, dtype: input.spec.dtype }, { binding: output.binding, dtype: output.spec.dtype }],
          body,
          layout.outputCount,
          [accessFor(input, 0, 'read'), accessFor(output, 1, 'write')],
          node.id,
        );
        continue;
      }

      const perOutputElements = 2 * padded - 2;
      const workspaceElements = checkedMultiply(layout.outputCount, perOutputElements, 'reductionWorkspaceElements');
      const workspaceBytes = checkedMultiply(workspaceElements, tensorDtypeWidth(node.options.accumulatorDtype), 'reductionWorkspaceBytes');
      totalWorkspaceBytes = checkedAdd(totalWorkspaceBytes, workspaceBytes, 'totalWorkspaceBytes');
      if (totalWorkspaceBytes > maxWorkspaceBytes) fail('TENSOR_SIMT_WORKSPACE_LIMIT', 'pressure', 'Resolved SIMT reduction workspace exceeds the selected finite limit.', { required: totalWorkspaceBytes, maximum: maxWorkspaceBytes });
      const workspaceBinding = addBinding('workspace', node.options.accumulatorDtype, workspaceBytes, `workspace:${node.id}`);
      const workspace = Object.freeze({ id: `workspace:${node.id}`, binding: workspaceBinding, dtype: node.options.accumulatorDtype, elementCount: workspaceElements, byteLength: workspaceBytes });
      workspaces.push(workspace);

      const gatherWork = checkedMultiply(layout.outputCount, padded, 'reductionGatherWork');
      const gatherBody = [
        'let index = gpu.cast.u64(gpu.thread.globalX());',
        `if (index < ${u64(gatherWork)}) {`,
        `  let outputIndex = index / ${u64(padded)};`,
        `  let reductionIndex = index % ${u64(padded)};`,
        `  let destination = outputIndex * ${u64(perOutputElements)} + reductionIndex;`,
        `  if (reductionIndex < ${u64(layout.reductionCount)}) {`,
        ...outputCoordinates.lines.map((line) => `    ${line}`),
        ...reductionCoordinates.lines.map((line) => `    ${line}`),
        ...offsetLines('inputOffset', input.spec, input.originByteOffset, inputCoordinates).map((line) => `    ${line}`),
        `    p1[destination] = ${cast(node.options.accumulatorDtype, input.spec.dtype, 'p0[inputOffset]')};`,
        '  } else {',
        `    p1[destination] = ${identityValue};`,
        '  }',
        '}',
      ];
      addKernel(
        [{ binding: input.binding, dtype: input.spec.dtype }, { binding: workspaceBinding, dtype: node.options.accumulatorDtype }],
        gatherBody,
        gatherWork,
        [accessFor(input, 0, 'read'), { argumentIndex: 1, byteOffset: 0, byteLength: workspaceBytes, mode: 'write' }],
        node.id,
      );

      let inputSize = padded;
      let inputRegion = 0;
      let outputRegion = padded;
      while (inputSize > 1) {
        const outputSize = inputSize / 2;
        const stageWork = checkedMultiply(layout.outputCount, outputSize, 'reductionStageWork');
        const finalStage = outputSize === 1;
        const stageBody = [
          'let index = gpu.cast.u64(gpu.thread.globalX());',
          `if (index < ${u64(stageWork)}) {`,
          `  let outputIndex = index / ${u64(outputSize)};`,
          `  let pairIndex = index % ${u64(outputSize)};`,
          `  let base = outputIndex * ${u64(perOutputElements)};`,
          `  let leftOffset = base + ${u64(inputRegion)} + pairIndex * gpu.u64(2n);`,
          '  let rightOffset = leftOffset + gpu.u64(1n);',
          `  let combined = ${combine(node.options.operator, node.options.accumulatorDtype, 'p0[leftOffset]', 'p0[rightOffset]')};`,
          finalStage ? '  p1[outputIndex] = combined;' : `  p0[base + ${u64(outputRegion)} + pairIndex] = combined;`,
          '}',
        ];
        addKernel(
          finalStage
            ? [{ binding: workspaceBinding, dtype: node.options.accumulatorDtype }, { binding: output.binding, dtype: output.spec.dtype }]
            : [{ binding: workspaceBinding, dtype: node.options.accumulatorDtype }],
          stageBody,
          stageWork,
          finalStage
            ? [{ argumentIndex: 0, byteOffset: 0, byteLength: workspaceBytes, mode: 'read' }, accessFor(output, 1, 'write')]
            : [{ argumentIndex: 0, byteOffset: 0, byteLength: workspaceBytes, mode: 'read-write' }],
          node.id,
        );
        inputRegion = outputRegion;
        inputSize = outputSize;
        outputRegion += outputSize;
      }
      continue;
    }

    const outputCoordinates = coordinates('index', node.outputSpec.logicalShape, 'c');
    const parameterRecords = inputs.map((input) => ({ binding: input.binding, dtype: input.spec.dtype }));
    parameterRecords.push({ binding: output.binding, dtype: output.spec.dtype });
    const accessRecords = inputs.map((input, index) => accessFor(input, index, 'read'));
    accessRecords.push(accessFor(output, parameterRecords.length - 1, 'write'));
    const body = [
      'let index = gpu.cast.u64(gpu.thread.globalX());',
      `if (index < ${u64(outputCount)}) {`,
      ...outputCoordinates.lines.map((line) => `  ${line}`),
    ];
    let expression;

    if (node.op === 'fill') {
      expression = scalar(node.options.value);
    } else if (node.op === 'matmul') {
      const left = inputs[0];
      const right = inputs[1];
      const rank = left.spec.rank;
      const row = outputCoordinates.names[rank - 2];
      const column = outputCoordinates.names[rank - 1];
      const batch = rank === 3 ? outputCoordinates.names[0] : null;
      const contraction = node.options.transposeA ? left.spec.logicalShape[rank - 2] : left.spec.logicalShape[rank - 1];
      body.push(`  let accumulator = gpu.${node.options.accumulatorDtype}(0);`);
      body.push(`  for (let k = gpu.u64(0n); k < ${u64(contraction)}; k += gpu.u64(1n)) {`);
      const leftCoordinates = rank === 3
        ? [left.spec.logicalShape[0] === 1 ? '0' : batch, node.options.transposeA ? 'k' : row, node.options.transposeA ? row : 'k']
        : [node.options.transposeA ? 'k' : row, node.options.transposeA ? row : 'k'];
      const rightCoordinates = rank === 3
        ? [right.spec.logicalShape[0] === 1 ? '0' : batch, node.options.transposeB ? column : 'k', node.options.transposeB ? 'k' : column]
        : [node.options.transposeB ? column : 'k', node.options.transposeB ? 'k' : column];
      body.push(...offsetLines('leftOffset', left.spec, left.originByteOffset, leftCoordinates).map((line) => `    ${line}`));
      body.push(...offsetLines('rightOffset', right.spec, right.originByteOffset, rightCoordinates).map((line) => `    ${line}`));
      const leftValue = cast(node.options.accumulatorDtype, left.spec.dtype, 'p0[leftOffset]');
      const rightValue = cast(node.options.accumulatorDtype, right.spec.dtype, 'p1[rightOffset]');
      body.push(`    accumulator = ${combine('add', node.options.accumulatorDtype, 'accumulator', combine('mul', node.options.accumulatorDtype, leftValue, rightValue))};`);
      body.push('  }');
      expression = cast(node.outputSpec.dtype, node.options.accumulatorDtype, 'accumulator');
    } else {
      const inputOffsets = inputs.map((input, inputIndex) => {
        const coordinatesForInput = node.op === 'binary' ? broadcastCoordinates(input.spec, outputCoordinates.names) : outputCoordinates.names;
        body.push(...offsetLines(`inputOffset${inputIndex}`, input.spec, input.originByteOffset, coordinatesForInput).map((line) => `  ${line}`));
        return `p${inputIndex}[inputOffset${inputIndex}]`;
      });
      if (node.op === 'copy' || node.op === 'contiguous') expression = inputOffsets[0];
      else if (node.op === 'cast') expression = cast(node.outputSpec.dtype, inputs[0].spec.dtype, inputOffsets[0]);
      else if (node.op === 'unary') expression = unary(node.options.operator, node.outputSpec.dtype, inputOffsets[0]);
      else if (node.op === 'binary') expression = combine(node.options.operator, node.outputSpec.dtype, inputOffsets[0], inputOffsets[1]);
      else fail('TENSOR_SIMT_OPERATION_UNSUPPORTED', 'unsupported', 'A material tensor operation lacks a SIMT lowering.', { op: node.op });
    }
    body.push(`  p${parameterRecords.length - 1}[index] = ${expression};`);
    body.push('}');
    addKernel(parameterRecords, body, outputCount, accessRecords, node.id);
  }

  const source = kernels.map((kernel) => kernel.source).join('\n\n');
  const outputRecords = program.outputs.map((output) => {
    const reference = references.get(output.valueId);
    return Object.freeze({ name: output.name, valueId: output.valueId, baseValueId: reference.baseValueId, spec: output.spec });
  });
  const canonical = deepFreeze({
    contract: 'SPEC-0005-generated-simt-lowering-v1',
    planIdentity: plan.compatibilityIdentity,
    blockSize,
    limits: { ...TENSOR_SIMT_LIMITS, maxWorkspaceBytes },
    bindings: bindings.map((entry) => ({ ...entry })),
    workspaces: workspaces.map((entry) => ({ ...entry })),
    kernels: kernels.map((kernel) => ({ id: kernel.id, semanticNode: kernel.semanticNode, functionName: kernel.functionName, workItems: kernel.workItems, grid: kernel.grid, block: kernel.block, after: [...kernel.after], parameters: kernel.parameterRecords.map((entry) => ({ ...entry })), accesses: kernel.accesses.map((entry) => ({ ...entry })) })),
    totalWorkspaceBytes,
    outputs: outputRecords.map((entry) => ({ name: entry.name, valueId: entry.valueId, baseValueId: entry.baseValueId, specIdentity: entry.spec.compatibilityIdentity })),
  });
  return Object.freeze({
    source,
    functions: Object.freeze(functions),
    kernels: Object.freeze(kernels),
    bindings: Object.freeze(bindings),
    workspaces: Object.freeze(workspaces),
    references,
    outputs: Object.freeze(outputRecords),
    totalWorkspaceBytes,
    canonical,
    compatibilityIdentity: identity('tensor-simt-lowering-v1', canonical),
  });
}
