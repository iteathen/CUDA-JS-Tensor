import { tensorDtypeWidth, TENSOR_DTYPES } from '../../tensor-value/index.mjs';
import { TensorPlan } from '../../tensor-program/index.mjs';

import { checkedAdd, checkedMultiply, deepFreeze, fail, identity, TENSOR_SIMT_LIMITS } from './contract.mjs';
import { CUDA_JS_TENSOR_COMPATIBILITY, requireTensorDeviceLibraryOutput } from './cuda-js-compatibility.mjs';

export const TENSOR_DEVICE_PROGRAM_CONTRACT = 'SPEC-0009-item-parallel-device-tensor-program-v1';
export const TENSOR_DEVICE_PROGRAM_OUTPUTS = Object.freeze(['ptx', 'lto-ir']);
export const TENSOR_DEVICE_PROGRAM_LIMITS = Object.freeze({
  maxItemCapacity: 0xffff_ffff,
  defaultMaxWorkspaceBytes: TENSOR_SIMT_LIMITS.maxWorkspaceBytes,
  maxWorkspaceBytes: Number.MAX_SAFE_INTEGER,
});

const OUTPUT_SET = new Set(TENSOR_DEVICE_PROGRAM_OUTPUTS);
const OUTPUT_ORDER = new Map(TENSOR_DTYPES.map((dtype, index) => [dtype, index]));

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
  if (!token) fail('TENSOR_DEVICE_OPERATOR_UNSUPPORTED', 'unsupported', 'Device-callable lowering does not own the requested operator.', { operator });
  if (dtype === 'i32' && token !== '/') return `gpu.cast.i32(gpu.cast.u32(${left}) ${token} gpu.cast.u32(${right}))`;
  return `(${left} ${token} ${right})`;
}

function unary(operator, dtype, value) {
  if (operator === 'abs') return `gpu.math.abs(${value})`;
  if (operator === 'exp' || operator === 'log' || operator === 'sqrt') return `gpu.math.${operator}(${value})`;
  if (operator === 'neg') return dtype === 'i32' ? `gpu.cast.i32(gpu.u32(0) - gpu.cast.u32(${value}))` : `(-${value})`;
  fail('TENSOR_DEVICE_OPERATOR_UNSUPPORTED', 'unsupported', 'Device-callable lowering does not own the requested unary operator.', { operator });
}

function shapeProduct(shape, field) {
  let result = 1;
  for (const dimension of shape) result = checkedMultiply(result, dimension, field);
  return result;
}

function nextPowerOfTwo(value) {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

function alignUp(value, alignment, field) {
  const remainder = value % alignment;
  return remainder === 0 ? value : checkedAdd(value, alignment - remainder, field);
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

function sameArray(left, right) {
  return left.length === right.length && left.every((entry, index) => entry === right[index]);
}

function requireStaticSpec(spec, valueId) {
  if (spec.activeAxis0 !== null) {
    fail('TENSOR_DEVICE_ACTIVE_AXIS_UNSUPPORTED', 'unsupported', 'The first device-callable profile uses caller-owned item occupancy and requires static TensorSpecs.', { valueId });
  }
}

function sharedInvariantAtItemAxis(spec, outputRank) {
  const sourceAxis = spec.rank - outputRank;
  return sourceAxis < 0 || spec.capacityShape[sourceAxis] === 1 || spec.strides[sourceAxis] === 0;
}

function classifyItemValues(plan, itemCapacity, itemInputNames) {
  const program = plan.program;
  const selected = new Set(itemInputNames);
  const itemByValue = new Map();

  for (const input of program.inputs) {
    requireStaticSpec(input.spec, input.valueId);
    const item = selected.has(input.name);
    if (item && (input.spec.rank < 1 || input.spec.capacityShape[0] !== itemCapacity)) {
      fail('TENSOR_DEVICE_ITEM_INPUT_INVALID', 'validation', 'Each item input must expose axis 0 at the exact selected item capacity.', { input: input.name, itemCapacity });
    }
    itemByValue.set(input.valueId, item);
  }

  for (const node of program.nodes) {
    requireStaticSpec(node.outputSpec, node.id);
    const inputItems = node.inputIds.map((id) => itemByValue.get(id));
    const inputSpecs = node.inputIds.map((id) => program.valueSpec(id));
    let item = false;

    if (node.materialization === 'view') {
      item = inputItems[0];
      if (item) {
        const input = inputSpecs[0];
        const output = node.outputSpec;
        if (node.op === 'reshape') {
          const inputPerItem = shapeProduct(input.capacityShape.slice(1), 'device.reshape.inputPerItem');
          const outputPerItem = shapeProduct(output.capacityShape.slice(1), 'device.reshape.outputPerItem');
          if (input.rank < 1 || output.rank < 1 || input.capacityShape[0] !== itemCapacity || output.capacityShape[0] !== itemCapacity || inputPerItem !== outputPerItem) {
            fail('TENSOR_DEVICE_ITEM_VIEW_INVALID', 'unsupported', 'reshape must preserve the complete item axis and per-item element order.', { node: node.id });
          }
        } else if (node.op === 'permute') {
          if (node.options.axes[0] !== 0) fail('TENSOR_DEVICE_ITEM_VIEW_INVALID', 'unsupported', 'permute must leave the item axis at axis 0.', { node: node.id });
        } else if (node.op === 'slice') {
          const axis = node.options.slices[0];
          if (!axis || axis.start !== 0 || axis.length !== itemCapacity || axis.step !== 1) fail('TENSOR_DEVICE_ITEM_VIEW_INVALID', 'unsupported', 'slice must retain the complete identity item axis.', { node: node.id });
        } else if (node.op === 'broadcast') {
          if (input.rank !== output.rank || output.capacityShape[0] !== itemCapacity) fail('TENSOR_DEVICE_ITEM_VIEW_INVALID', 'unsupported', 'broadcast cannot shift or change the item axis.', { node: node.id });
        }
      }
    } else if (['copy', 'cast', 'contiguous', 'unary'].includes(node.op)) {
      item = inputItems[0];
    } else if (node.op === 'binary') {
      item = inputItems.some(Boolean);
      if (item) {
        if (node.outputSpec.rank < 1 || node.outputSpec.capacityShape[0] !== itemCapacity) fail('TENSOR_DEVICE_ITEM_BINARY_INVALID', 'unsupported', 'An item-varying binary result must retain item axis 0.', { node: node.id });
        for (let index = 0; index < inputSpecs.length; index += 1) {
          if (inputItems[index] && inputSpecs[index].rank !== node.outputSpec.rank) fail('TENSOR_DEVICE_ITEM_BINARY_INVALID', 'unsupported', 'An item-varying binary operand cannot shift its item axis under broadcasting.', { node: node.id, input: index });
          if (!inputItems[index] && !sharedInvariantAtItemAxis(inputSpecs[index], node.outputSpec.rank)) fail('TENSOR_DEVICE_SHARED_INPUT_VARIES', 'unsupported', 'A shared binary operand varies along the aligned item axis.', { node: node.id, input: index });
        }
      }
    } else if (node.op === 'reduce') {
      item = inputItems[0];
      if (item && node.options.axes.includes(0)) fail('TENSOR_DEVICE_CROSS_ITEM_REDUCTION', 'unsupported', 'The first device-callable profile cannot reduce the item axis.', { node: node.id });
    } else if (node.op === 'matmul') {
      const left = inputSpecs[0];
      const right = inputSpecs[1];
      if (left.rank === 2) {
        item = inputItems[0] && !inputItems[1] && node.options.transposeA === false;
        if (!item || left.capacityShape[0] !== itemCapacity || node.outputSpec.capacityShape[0] !== itemCapacity) {
          fail('TENSOR_DEVICE_ITEM_MATMUL_INVALID', 'unsupported', 'Rank-2 item matmul requires item-varying non-transposed A rows and shared B.', { node: node.id });
        }
      } else {
        item = inputItems.some(Boolean);
        if (!item || node.outputSpec.capacityShape[0] !== itemCapacity) fail('TENSOR_DEVICE_ITEM_MATMUL_INVALID', 'unsupported', 'Rank-3 item matmul requires at least one item-varying batch operand.', { node: node.id });
        for (let index = 0; index < inputSpecs.length; index += 1) {
          if (inputItems[index] && inputSpecs[index].capacityShape[0] !== itemCapacity) fail('TENSOR_DEVICE_ITEM_MATMUL_INVALID', 'unsupported', 'An item-varying rank-3 matmul operand has the wrong batch capacity.', { node: node.id, input: index });
          if (!inputItems[index] && inputSpecs[index].capacityShape[0] !== 1) fail('TENSOR_DEVICE_SHARED_INPUT_VARIES', 'unsupported', 'A shared rank-3 matmul operand must broadcast batch dimension one.', { node: node.id, input: index });
        }
      }
    }

    if (node.materialization === 'materialize' && !item) {
      fail('TENSOR_DEVICE_SHARED_MATERIAL_UNSUPPORTED', 'unsupported', 'Every materialized node must be independently owned by one item in the first device-callable profile.', { node: node.id, op: node.op });
    }
    if (item && (node.outputSpec.rank < 1 || node.outputSpec.capacityShape[0] !== itemCapacity)) {
      fail('TENSOR_DEVICE_ITEM_AXIS_LOST', 'unsupported', 'An item-varying value lost the exact item axis.', { node: node.id });
    }
    itemByValue.set(node.id, item);
  }

  for (const output of program.outputs) {
    if (!itemByValue.get(output.valueId)) fail('TENSOR_DEVICE_SHARED_OUTPUT_UNSUPPORTED', 'unsupported', 'Every public device-callable output must vary independently by item.', { output: output.name });
    if (output.spec.access === 'write') fail('TENSOR_DEVICE_OUTPUT_SOURCE_NOT_READABLE', 'validation', 'A public device-callable output source must be readable.', { output: output.name });
  }
  return itemByValue;
}

function allocateWorkspace(plan, itemCapacity) {
  const partitions = new Map();
  const materials = new Map();
  const reductionScratch = new Map();

  function partition(dtype) {
    let result = partitions.get(dtype);
    if (!result) {
      result = { dtype, width: tensorDtypeWidth(dtype), cursor: 0, alignmentElements: 1, materials: [], scratches: [] };
      partitions.set(dtype, result);
    }
    return result;
  }

  function allocate(record, elements, alignmentBytes, collection, field) {
    const target = partition(record.dtype);
    const alignmentElements = Math.max(1, alignmentBytes / target.width);
    target.alignmentElements = Math.max(target.alignmentElements, alignmentElements);
    const offsetElements = alignUp(target.cursor, alignmentElements, `${field}.offset`);
    const reservedElements = Math.max(elements, alignmentElements);
    target.cursor = checkedAdd(offsetElements, reservedElements, `${field}.end`);
    const allocated = Object.freeze({ ...record, offsetElements, elementCount: elements, reservedElements, alignmentBytes });
    target[collection].push(allocated);
    return allocated;
  }

  for (const node of plan.program.nodes) {
    if (node.materialization !== 'materialize') continue;
    const perItemElements = shapeProduct(node.outputSpec.logicalShape.slice(1), 'device.material.perItemElements');
    materials.set(node.id, allocate({ id: node.id, dtype: node.outputSpec.dtype }, perItemElements, node.outputSpec.alignment, 'materials', 'device.material'));
    if (node.op === 'reduce') {
      const input = plan.program.valueSpec(node.inputIds[0]);
      const reductionCount = shapeProduct(node.options.axes.map((axis) => input.logicalShape[axis]), 'device.reduction.count');
      const padded = nextPowerOfTwo(Math.max(1, reductionCount));
      if (padded > 1) {
        const outputCount = perItemElements;
        const scratchElements = checkedMultiply(outputCount, 2 * padded - 2, 'device.reduction.scratchElements');
        reductionScratch.set(node.id, allocate({ id: `reduction:${node.id}`, dtype: node.options.accumulatorDtype, padded, reductionCount, outputCount }, scratchElements, tensorDtypeWidth(node.options.accumulatorDtype), 'scratches', 'device.reduction.scratch'));
      }
    }
  }

  const ordered = [...partitions.values()].sort((left, right) => OUTPUT_ORDER.get(left.dtype) - OUTPUT_ORDER.get(right.dtype));
  let totalBytes = 0;
  for (const entry of ordered) {
    entry.perItemElements = alignUp(entry.cursor, entry.alignmentElements, 'device.workspace.perItemElements');
    entry.capacityElements = checkedMultiply(entry.perItemElements, itemCapacity, 'device.workspace.capacityElements');
    entry.byteLength = checkedMultiply(entry.capacityElements, entry.width, 'device.workspace.byteLength');
    entry.alignmentBytes = entry.alignmentElements * entry.width;
    totalBytes = checkedAdd(totalBytes, entry.byteLength, 'device.workspace.totalBytes');
  }
  return { partitions: ordered, materials, reductionScratch, totalBytes };
}

function referenceCoordinates(spec, outputCoordinates) {
  const rankOffset = outputCoordinates.length - spec.rank;
  return spec.logicalShape.map((dimension, axis) => dimension === 1 ? '0' : outputCoordinates[rankOffset + axis]);
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
  const reductionCount = shapeProduct(axes.map((axis) => inputSpec.logicalShape[axis]), 'device.reduction.count');
  return { reduced, outputToInput, reductionCount };
}

function lowerDeviceItemPlan(plan, profile) {
  const program = plan.program;
  const references = new Map();
  const partitionByDtype = new Map(profile.workspace.map((entry) => [entry.dtype, entry]));

  for (const input of profile.inputs) {
    references.set(input.valueId, { storage: 'input', parameterName: input.parameterName, spec: input.spec, originByteOffset: input.spec.byteOffset, itemVarying: input.itemVarying });
  }
  for (const node of program.nodes) {
    if (node.materialization === 'view') {
      references.set(node.id, { ...references.get(node.inputIds[0]), spec: node.outputSpec });
    } else {
      const material = profile.materialById.get(node.id);
      references.set(node.id, { storage: 'workspace', parameterName: partitionByDtype.get(node.outputSpec.dtype).parameterName, spec: node.outputSpec, originByteOffset: 0, itemVarying: true, materialOffsetElements: material.offsetElements, workspaceStrideElements: partitionByDtype.get(node.outputSpec.dtype).perItemElements });
    }
  }

  function offsetLines(name, reference, coordinateExpressions) {
    const width = tensorDtypeWidth(reference.spec.dtype);
    const relativeBytes = reference.spec.byteOffset - reference.originByteOffset;
    if (!Number.isSafeInteger(relativeBytes) || relativeBytes < 0 || relativeBytes % width !== 0) fail('TENSOR_DEVICE_VIEW_OFFSET_INVALID', 'validation', 'A device-callable view offset is not representable relative to its binding.', { parameter: reference.parameterName });
    const base = reference.storage === 'workspace'
      ? `item * ${u64(reference.workspaceStrideElements)} + ${u64(reference.materialOffsetElements + relativeBytes / width)}`
      : u64(relativeBytes / width);
    const lines = [`let ${name} = ${base};`];
    if (reference.storage === 'input' && reference.itemVarying) lines.push(`${name} += item * ${u64(reference.spec.strides[0])};`);
    const firstAxis = reference.itemVarying ? 1 : 0;
    for (let axis = firstAxis; axis < reference.spec.rank; axis += 1) {
      const coordinate = coordinateExpressions[axis];
      if (coordinate !== '0' && reference.spec.strides[axis] !== 0) lines.push(`${name} += ${coordinate} * ${u64(reference.spec.strides[axis])};`);
    }
    return lines;
  }

  function readExpression(lines, prefix, reference, coordinates_) {
    lines.push(...offsetLines(prefix, reference, coordinates_));
    return `${reference.parameterName}[${prefix}]`;
  }

  function writeLine(prefix, reference, coordinates_, expression) {
    return [...offsetLines(prefix, reference, coordinates_), `${reference.parameterName}[${prefix}] = ${expression};`];
  }

  const body = [
    `if (itemIndex >= gpu.u32(${profile.itemCapacity})) {`,
    '  return gpu.u32(1);',
    '}',
    'let item = gpu.cast.u64(itemIndex);',
  ];

  for (let nodeIndex = 0; nodeIndex < program.nodes.length; nodeIndex += 1) {
    const node = program.nodes[nodeIndex];
    if (node.materialization !== 'materialize') continue;
    const output = references.get(node.id);
    const inputs = node.inputIds.map((id) => references.get(id));
    const localShape = node.outputSpec.logicalShape.slice(1);
    const outputCount = shapeProduct(localShape, 'device.node.outputCount');
    if (outputCount === 0) continue;
    const outputCoordinates = coordinates('index', localShape, `n${nodeIndex}c`);
    const fullOutputCoordinates = ['item', ...outputCoordinates.names];
    body.push(`for (let index = gpu.u64(0n); index < ${u64(outputCount)}; index += gpu.u64(1n)) {`);
    body.push(...outputCoordinates.lines.map((line) => `  ${line}`));

    let expression;
    if (node.op === 'reduce') {
      const input = inputs[0];
      const layout = reductionLayout(input.spec, node.outputSpec, node.options.axes, node.options.keepDimensions);
      const inputCoordinates = new Array(input.spec.rank).fill('0');
      inputCoordinates[0] = 'item';
      for (let axis = 1; axis < input.spec.rank; axis += 1) {
        const reductionPosition = node.options.axes.indexOf(axis);
        inputCoordinates[axis] = reductionPosition >= 0 ? `n${nodeIndex}r${reductionPosition}` : fullOutputCoordinates[layout.outputToInput.get(axis)];
      }
      const identityValue = scalar(node.options.identity);
      if (layout.reductionCount === 0) expression = identityValue;
      else if (layout.reductionCount === 1) {
        const reductionCoordinates = coordinates('gpu.u64(0n)', node.options.axes.map((axis) => input.spec.logicalShape[axis]), `n${nodeIndex}r`);
        body.push(...reductionCoordinates.lines.map((line) => `  ${line}`));
        expression = cast(node.options.accumulatorDtype, input.spec.dtype, readExpression(body, `n${nodeIndex}InputOffset`, input, inputCoordinates));
      } else {
        const scratch = profile.scratchByNode.get(node.id);
        const workspace = partitionByDtype.get(node.options.accumulatorDtype);
        const scratchBase = `item * ${u64(workspace.perItemElements)} + ${u64(scratch.offsetElements)} + index * ${u64(2 * scratch.padded - 2)}`;
        body.push(`  let n${nodeIndex}ScratchBase = ${scratchBase};`);
        body.push(`  for (let reductionIndex = gpu.u64(0n); reductionIndex < ${u64(scratch.padded)}; reductionIndex += gpu.u64(1n)) {`);
        body.push(`    if (reductionIndex < ${u64(layout.reductionCount)}) {`);
        const reductionCoordinates = coordinates('reductionIndex', node.options.axes.map((axis) => input.spec.logicalShape[axis]), `n${nodeIndex}r`);
        body.push(...reductionCoordinates.lines.map((line) => `      ${line}`));
        const gatherLines = [];
        const gathered = readExpression(gatherLines, `n${nodeIndex}InputOffset`, input, inputCoordinates);
        body.push(...gatherLines.map((line) => `      ${line}`));
        body.push(`      ${workspace.parameterName}[n${nodeIndex}ScratchBase + reductionIndex] = ${cast(node.options.accumulatorDtype, input.spec.dtype, gathered)};`);
        body.push('    } else {');
        body.push(`      ${workspace.parameterName}[n${nodeIndex}ScratchBase + reductionIndex] = ${identityValue};`);
        body.push('    }');
        body.push('  }');
        body.push(`  let n${nodeIndex}Reduced = ${identityValue};`);
        let inputSize = scratch.padded;
        let inputRegion = 0;
        let outputRegion = scratch.padded;
        while (inputSize > 1) {
          const outputSize = inputSize / 2;
          body.push(`  for (let pairIndex = gpu.u64(0n); pairIndex < ${u64(outputSize)}; pairIndex += gpu.u64(1n)) {`);
          body.push(`    let leftOffset = n${nodeIndex}ScratchBase + ${u64(inputRegion)} + pairIndex * gpu.u64(2n);`);
          body.push('    let rightOffset = leftOffset + gpu.u64(1n);');
          const combined = combine(node.options.operator, node.options.accumulatorDtype, `${workspace.parameterName}[leftOffset]`, `${workspace.parameterName}[rightOffset]`);
          if (outputSize === 1) body.push(`    n${nodeIndex}Reduced = ${combined};`);
          else body.push(`    ${workspace.parameterName}[n${nodeIndex}ScratchBase + ${u64(outputRegion)} + pairIndex] = ${combined};`);
          body.push('  }');
          inputRegion = outputRegion;
          inputSize = outputSize;
          outputRegion += outputSize;
        }
        expression = `n${nodeIndex}Reduced`;
      }
    } else if (node.op === 'matmul') {
      const left = inputs[0];
      const right = inputs[1];
      const rank = left.spec.rank;
      const column = fullOutputCoordinates.at(-1);
      const row = rank === 2 ? 'item' : fullOutputCoordinates.at(-2);
      const contraction = node.options.transposeA ? left.spec.logicalShape[rank - 2] : left.spec.logicalShape[rank - 1];
      body.push(`  let accumulator = gpu.${node.options.accumulatorDtype}(0);`);
      body.push(`  for (let k = gpu.u64(0n); k < ${u64(contraction)}; k += gpu.u64(1n)) {`);
      const leftCoordinates = rank === 3
        ? [left.itemVarying ? 'item' : '0', node.options.transposeA ? 'k' : row, node.options.transposeA ? row : 'k']
        : [node.options.transposeA ? 'k' : row, node.options.transposeA ? row : 'k'];
      const rightCoordinates = rank === 3
        ? [right.itemVarying ? 'item' : '0', node.options.transposeB ? column : 'k', node.options.transposeB ? 'k' : column]
        : [node.options.transposeB ? column : 'k', node.options.transposeB ? 'k' : column];
      const matmulLines = [];
      const leftValue = readExpression(matmulLines, `n${nodeIndex}LeftOffset`, left, leftCoordinates);
      const rightValue = readExpression(matmulLines, `n${nodeIndex}RightOffset`, right, rightCoordinates);
      body.push(...matmulLines.map((line) => `    ${line}`));
      body.push(`    accumulator = ${combine('add', node.options.accumulatorDtype, 'accumulator', combine('mul', node.options.accumulatorDtype, cast(node.options.accumulatorDtype, left.spec.dtype, leftValue), cast(node.options.accumulatorDtype, right.spec.dtype, rightValue)))};`);
      body.push('  }');
      expression = cast(node.outputSpec.dtype, node.options.accumulatorDtype, 'accumulator');
    } else {
      const inputExpressions = inputs.map((input, inputIndex) => {
        const inputCoordinates = node.op === 'binary' ? referenceCoordinates(input.spec, fullOutputCoordinates) : fullOutputCoordinates;
        const lines = [];
        const value = readExpression(lines, `n${nodeIndex}Input${inputIndex}Offset`, input, inputCoordinates);
        body.push(...lines.map((line) => `  ${line}`));
        return value;
      });
      if (node.op === 'copy' || node.op === 'contiguous') expression = inputExpressions[0];
      else if (node.op === 'cast') expression = cast(node.outputSpec.dtype, inputs[0].spec.dtype, inputExpressions[0]);
      else if (node.op === 'unary') expression = unary(node.options.operator, node.outputSpec.dtype, inputExpressions[0]);
      else if (node.op === 'binary') expression = combine(node.options.operator, node.outputSpec.dtype, inputExpressions[0], inputExpressions[1]);
      else fail('TENSOR_DEVICE_OPERATION_UNSUPPORTED', 'unsupported', 'A material Tensor operation lacks item-parallel lowering.', { node: node.id, op: node.op });
    }

    const outputLines = writeLine(`n${nodeIndex}OutputOffset`, output, fullOutputCoordinates, expression);
    body.push(...outputLines.map((line) => `  ${line}`));
    body.push('}');
  }

  for (let outputIndex = 0; outputIndex < profile.outputs.length; outputIndex += 1) {
    const output = profile.outputs[outputIndex];
    const reference = references.get(output.valueId);
    const localShape = output.spec.logicalShape.slice(1);
    const count = output.perItemElements;
    if (count === 0) continue;
    const outputCoordinates = coordinates('outputIndex', localShape, `o${outputIndex}c`);
    const fullCoordinates = ['item', ...outputCoordinates.names];
    body.push(`for (let outputIndex = gpu.u64(0n); outputIndex < ${u64(count)}; outputIndex += gpu.u64(1n)) {`);
    body.push(...outputCoordinates.lines.map((line) => `  ${line}`));
    const readLines = [];
    const value = readExpression(readLines, `o${outputIndex}SourceOffset`, reference, fullCoordinates);
    body.push(...readLines.map((line) => `  ${line}`));
    body.push(`  ${output.parameterName}[item * ${u64(count)} + outputIndex] = ${value};`);
    body.push('}');
  }
  body.push('return gpu.u32(0);');

  const parameters = profile.parameters.map((entry) => ({ name: entry.parameterName, type: entry.type }));
  const source = ['function tensorRunItem(' + parameters.map((entry) => entry.name).join(', ') + ') {', ...body.map((line) => `  ${line}`), '}'].join('\n');
  return Object.freeze({
    source,
    function: Object.freeze({ name: 'tensorRunItem', kind: 'device', parameters: Object.freeze(parameters.map((entry) => Object.freeze(entry))), returns: 'u32' }),
  });
}

export function createDeviceItemProfile(plan, options) {
  if (!(plan instanceof TensorPlan)) fail('TENSOR_DEVICE_PLAN_INVALID', 'validation', 'Device-callable Tensor compilation requires a TensorPlan.');
  const itemCapacity = options.itemCapacity;
  if (!Number.isSafeInteger(itemCapacity) || itemCapacity < 1 || itemCapacity > TENSOR_DEVICE_PROGRAM_LIMITS.maxItemCapacity) fail('TENSOR_DEVICE_ITEM_CAPACITY_INVALID', 'validation', 'itemCapacity must be a positive u32-range safe integer.', { maximum: TENSOR_DEVICE_PROGRAM_LIMITS.maxItemCapacity });
  if (!Array.isArray(options.itemInputs) || options.itemInputs.length < 1 || options.itemInputs.some((name) => typeof name !== 'string')) fail('TENSOR_DEVICE_ITEM_INPUTS_INVALID', 'validation', 'itemInputs must be a nonempty array of exact program input names.');
  if (new Set(options.itemInputs).size !== options.itemInputs.length) fail('TENSOR_DEVICE_ITEM_INPUTS_INVALID', 'validation', 'itemInputs must not contain duplicates.');
  const programInputNames = new Set(plan.program.inputs.map((entry) => entry.name));
  for (const name of options.itemInputs) if (!programInputNames.has(name)) fail('TENSOR_DEVICE_ITEM_INPUT_UNKNOWN', 'validation', 'itemInputs contains an unknown program input.', { name });
  const normalizedItemInputs = plan.program.inputs.filter((entry) => options.itemInputs.includes(entry.name)).map((entry) => entry.name);
  const output = options.output ?? 'ptx';
  if (!OUTPUT_SET.has(output)) fail('TENSOR_DEVICE_OUTPUT_INVALID', 'validation', 'output must be ptx or lto-ir.', { output });
  requireTensorDeviceLibraryOutput(output);
  const maxWorkspaceBytes = options.maxWorkspaceBytes ?? TENSOR_DEVICE_PROGRAM_LIMITS.defaultMaxWorkspaceBytes;
  if (!Number.isSafeInteger(maxWorkspaceBytes) || maxWorkspaceBytes < 0 || maxWorkspaceBytes > TENSOR_DEVICE_PROGRAM_LIMITS.maxWorkspaceBytes) fail('TENSOR_DEVICE_WORKSPACE_LIMIT_INVALID', 'validation', 'maxWorkspaceBytes must be a nonnegative safe integer within the device-callable profile limit.', { maximum: TENSOR_DEVICE_PROGRAM_LIMITS.maxWorkspaceBytes });

  const itemByValue = classifyItemValues(plan, itemCapacity, normalizedItemInputs);
  const allocation = allocateWorkspace(plan, itemCapacity);
  if (allocation.totalBytes > maxWorkspaceBytes) fail('TENSOR_DEVICE_WORKSPACE_LIMIT', 'pressure', 'The device-callable program exceeds the selected finite workspace limit.', { required: allocation.totalBytes, maximum: maxWorkspaceBytes });

  let parameterIndex = 0;
  const parameters = [Object.freeze({ parameterIndex: parameterIndex++, parameterName: 'itemIndex', role: 'item-index', type: 'u32', dtype: 'u32', access: 'read', itemVarying: false })];
  const inputs = plan.program.inputs.map((entry, index) => {
    const parameterName = `input${index}`;
    const width = tensorDtypeWidth(entry.spec.dtype);
    const elementCount = Math.max(1, Math.ceil((entry.spec.requiredByteEnd - entry.spec.byteOffset) / width));
    const byteLength = checkedMultiply(elementCount, width, 'device.input.byteLength');
    const record = Object.freeze({ parameterIndex: parameterIndex++, parameterName, role: 'input', type: `ptr<${entry.spec.dtype}>`, dtype: entry.spec.dtype, access: 'read', itemVarying: itemByValue.get(entry.valueId), name: entry.name, valueId: entry.valueId, spec: entry.spec, elementCount, byteLength });
    parameters.push(record);
    return record;
  });
  const outputs = plan.program.outputs.map((entry, index) => {
    const parameterName = `output${index}`;
    const perItemElements = shapeProduct(entry.spec.logicalShape.slice(1), 'device.output.perItemElements');
    const elementCount = checkedMultiply(itemCapacity, perItemElements, 'device.output.elementCount');
    const width = tensorDtypeWidth(entry.spec.dtype);
    const logicalByteLength = checkedMultiply(elementCount, width, 'device.output.byteLength');
    const record = Object.freeze({ parameterIndex: parameterIndex++, parameterName, role: 'output', type: `ptr<${entry.spec.dtype}>`, dtype: entry.spec.dtype, access: 'write', itemVarying: true, name: entry.name, valueId: entry.valueId, spec: entry.spec, perItemElements, elementCount, byteLength: Math.max(width, logicalByteLength) });
    parameters.push(record);
    return record;
  });
  const workspace = allocation.partitions.map((entry, index) => {
    const parameterName = `workspace${index}`;
    const record = Object.freeze({ parameterIndex: parameterIndex++, parameterName, role: 'workspace', type: `ptr<${entry.dtype}>`, dtype: entry.dtype, access: 'read-write', itemVarying: true, perItemElements: entry.perItemElements, elementCount: entry.capacityElements, byteLength: entry.byteLength, alignmentBytes: entry.alignmentBytes, materials: Object.freeze(entry.materials.map((material) => Object.freeze({ ...material }))), scratches: Object.freeze(entry.scratches.map((scratch) => Object.freeze({ ...scratch }))) });
    parameters.push(record);
    return record;
  });
  const parameterLimit = CUDA_JS_TENSOR_COMPATIBILITY.deviceJsLimits.parametersPerFunction;
  if (parameters.length > parameterLimit) fail('TENSOR_DEVICE_PARAMETER_LIMIT', 'pressure', 'The device-callable ABI exceeds the public CUDA-JS function parameter limit.', { required: parameters.length, maximum: parameterLimit });

  const canonical = deepFreeze({
    contract: TENSOR_DEVICE_PROGRAM_CONTRACT,
    planIdentity: plan.compatibilityIdentity,
    itemCapacity,
    itemInputs: [...normalizedItemInputs],
    output,
    maxWorkspaceBytes,
    cudaJsCompatibility: {
      deviceJsLimits: { ...CUDA_JS_TENSOR_COMPATIBILITY.deviceJsLimits },
      compilerOutputFormats: [...CUDA_JS_TENSOR_COMPATIBILITY.compilerOutputFormats],
    },
    parameters: parameters.map((entry) => ({ parameterIndex: entry.parameterIndex, parameterName: entry.parameterName, role: entry.role, type: entry.type, dtype: entry.dtype, access: entry.access, itemVarying: entry.itemVarying, name: entry.name ?? null, valueId: entry.valueId ?? null, elementCount: entry.elementCount ?? null, byteLength: entry.byteLength ?? null })),
    workspace: workspace.map((entry) => ({ dtype: entry.dtype, parameterIndex: entry.parameterIndex, perItemElements: entry.perItemElements, elementCount: entry.elementCount, byteLength: entry.byteLength, alignmentBytes: entry.alignmentBytes, materials: entry.materials.map((material) => ({ id: material.id, offsetElements: material.offsetElements, elementCount: material.elementCount, reservedElements: material.reservedElements, alignmentBytes: material.alignmentBytes })), scratches: entry.scratches.map((scratch) => ({ id: scratch.id, offsetElements: scratch.offsetElements, elementCount: scratch.elementCount, reservedElements: scratch.reservedElements, padded: scratch.padded, reductionCount: scratch.reductionCount })) })),
    totalWorkspaceBytes: allocation.totalBytes,
    outputs: outputs.map((entry) => ({ name: entry.name, valueId: entry.valueId, dtype: entry.dtype, perItemElements: entry.perItemElements, elementCount: entry.elementCount, byteLength: entry.byteLength })),
  });
  const profile = {
    plan,
    itemCapacity,
    itemInputs: Object.freeze([...normalizedItemInputs]),
    output,
    maxWorkspaceBytes,
    parameters: Object.freeze(parameters),
    inputs: Object.freeze(inputs),
    outputs: Object.freeze(outputs),
    workspace: Object.freeze(workspace),
    totalWorkspaceBytes: allocation.totalBytes,
    materialById: allocation.materials,
    scratchByNode: allocation.reductionScratch,
    canonical,
    compatibilityIdentity: identity('tensor-device-item-profile-v1', canonical),
  };
  const lowering = lowerDeviceItemPlan(plan, profile);
  return Object.freeze({ ...profile, lowering });
}
