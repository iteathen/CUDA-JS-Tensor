# Compact Doctrine

1. Start from domain truth and explicit authority.
2. Give each invariant, state, lifecycle, and compatibility fact one owner.
3. Compose small LEGO contracts through public injected ports.
4. Treat a LEGO as encapsulated composition, not necessarily an atomic leaf: a larger brick may recursively contain smaller internal bricks, while the parent remains the external semantic owner and hides child topology from neighbors.
5. Choose LEGO seams primarily by semantic/ontological ownership and lifecycle cohesion, then functional cohesion, stable dependency/substitution boundaries, and independently owned failure/resource behavior. File count, line count, method count, and agent context size are diagnostics, not architectural authorities.
6. For a very large function, split where independently meaningful invariants, state transitions, resources, failure domains, or reasons to change separate; do not split merely because the function is long. Keep genuinely indivisible algorithms together and improve them with private helpers, explicit phases, tables, or private state rather than fake public bricks.
7. Stop recursive decomposition when another split protects no independent ownership, lifecycle, substitution, failure/resource boundary, testing value, or change boundary. Avoid both monoliths and abstraction confetti; passing one giant shared context among arbitrary helpers is not LEGO decomposition.
8. Keep convenience as normalization over the same expert contract.
9. Prefer deterministic, finite, inspectable behavior and honest failure.
10. Account for complexity in callers, generated code, memory, synchronization, tooling, evidence, and cleanup.
11. Keep universal semantics independent of backend shape; specialize generated hot paths.
12. Treat acceleration as a qualified variant, not a semantic requirement.
13. Preserve complete deletion: removing one backend or consumer leaves no solely owned residue.
14. Measure the complete system before making performance or default-selection claims.
