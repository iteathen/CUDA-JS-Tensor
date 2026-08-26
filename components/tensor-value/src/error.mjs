function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Array.isArray(value) ? value : Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

export class TensorError extends Error {
  constructor(code, category, message, details = {}, options = {}) {
    super(message, options);
    this.name = 'TensorError';
    this.code = code;
    this.category = category;
    this.details = freeze({ ...details });
  }
}

export function fail(code, category, message, details = {}) {
  throw new TensorError(code, category, message, details);
}

export function failureSummary(error) {
  const code = typeof error?.code === 'string' ? error.code : 'UNSTRUCTURED_FAILURE';
  const category = typeof error?.category === 'string' ? error.category : 'internal';
  const name = typeof error?.name === 'string' ? error.name : 'Error';
  return Object.freeze({ code, category, name });
}

export function deepFreeze(value) {
  return freeze(value);
}
