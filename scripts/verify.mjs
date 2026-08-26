import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const required = [
  'AGENTS.md', 'README.md', 'STATUS.md', 'LICENSE', 'LICENSING.md', 'CONTRIBUTING.md', 'SECURITY.md',
  'agent_files/AGENTS.md', 'agent_files/AI_RULES.md', 'agent_files/DESIGN_ALIGNMENT_CARD.md',
  'agent_files/SYSTEM_REGISTRY.md', 'agent_files/VALIDATION_POLICY.md', 'agent_files/general_foundation/PRINCIPLES.md',
  'docs/PROJECT_CHARTER.md', 'docs/architecture/TARGET_ARCHITECTURE.md',
  'docs/decisions/ADR-0001-separate-tensor-package-and-dependency-direction.md',
  'docs/specs/SPEC-0000-tensor-contract-map.md', 'docs/specs/SPEC-0001-tensor-session-spec-and-value-model.md',
  'docs/specs/SPEC-0002-tensor-program-plan-and-dense-operations.md',
  'docs/specs/SPEC-0003-accelerated-dense-backend-profiles.md',
  'docs/plans/2026-08-26-foundation-plan.md', 'docs/integrations/the_restaurant.md', 'next_step.yaml',
];

for (const file of required) assert(existsSync(path.join(root, file)), `Missing required artifact: ${file}`);

const packageJson = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
assert.equal(packageJson.name, 'cuda-js-tensor');
assert.equal(packageJson.version, '0.1.0-alpha.0');
assert.equal(packageJson.private, true, 'Package must remain publication-guarded during foundation work.');
assert.equal(packageJson.license, 'AGPL-3.0-or-later');
JSON.parse(readFileSync(path.join(root, 'next_step.yaml'), 'utf8'));

const ignored = new Set(['.git', 'build', 'node_modules']);
const files = [];
const visit = (directory) => {
  for (const name of readdirSync(directory)) {
    if (ignored.has(name)) continue;
    const target = path.join(directory, name);
    if (statSync(target).isDirectory()) visit(target);
    else files.push(path.relative(root, target).replaceAll('\\', '/'));
  }
};
visit(root);

const forbiddenNative = files.filter((file) => /\.(?:c|cc|cpp|cxx|cu|h|hh|hpp|ptx)$/i.test(file));
assert.deepEqual(forbiddenNative, [], `Forbidden maintained native source: ${forbiddenNative.join(', ')}`);

for (const file of files.filter((entry) => /\.(?:md|mjs|json|yaml|yml)$/i.test(entry))) {
  const text = readFileSync(path.join(root, file), 'utf8');
  assert(!/^(?:<<<<<<<|=======|>>>>>>>)/m.test(text), `Merge marker in ${file}`);
}

console.log(`CUDA-JS-Tensor foundation verification passed: ${files.length} maintained files, no native production source, package and authority identities aligned.`);
