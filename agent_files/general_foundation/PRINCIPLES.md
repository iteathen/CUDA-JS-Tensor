# Compact Doctrine

1. Start from domain truth and explicit authority.
2. Give each invariant, state, lifecycle, and compatibility fact one owner.
3. Compose small LEGO contracts through public injected ports. LEGO is the outer architectural discipline: it governs ownership, universality, replaceability, scope containment, damage-limiting encapsulation, and cognitive/context containment.
4. Treat a LEGO as encapsulated composition, not necessarily an atomic leaf: a larger brick may recursively contain smaller internal bricks, while the parent remains the external semantic owner and hides child topology from neighbors.
5. Size every LEGO for one agent's **full-attention envelope**. The complete authoritative working set—public contract, implementation, invariants, lifecycle/resource/failure rules, tests/conformance, and immediate dependency/consumer interfaces needed to understand consequences—must fit comfortably in focused attention with substantial headroom for reasoning and review. Merely fitting in a model's maximum context window is insufficient.
6. Choose LEGO seams using both cohesion and context fit. Strong seam signals include semantic/ontological ownership, lifecycle cohesion, functional cohesion, stable dependency/substitution boundaries, independently owned failure/resource behavior, volatility, and execution locality. When the authoritative working set exceeds full attention, recursively split at the strongest real seam or narrow scope. Context pressure is a deciding architectural constraint, but it does not justify arbitrary file/module boundaries that duplicate truth or force cross-boundary knowledge of internals.
7. For a very large function, split where independently meaningful invariants, state transitions, resources, failure domains, phases, or reasons to change separate. Do not split merely because the function is long. Keep genuinely indivisible external semantics together while using private helpers, explicit phases, tables, private state, or private child LEGOs so each reasoning unit remains attention-bounded.
8. Stop recursive decomposition when another split protects no independent ownership, lifecycle, substitution, failure/resource, testing/change, or attention boundary without creating duplicated truth or cross-boundary chatter. Avoid both monoliths and abstraction confetti; passing one giant shared context among arbitrary helpers is not LEGO decomposition.
9. Inside each valid LEGO, use SOLID to structure responsibilities and dependency direction; use CUPID to make that implementation composable, predictable, idiomatic, and domain-based; use KISS only to remove remaining unjustified complexity. A lower-level principle may not defeat a higher-level one.
10. Keep convenience as normalization over the same expert contract.
11. Prefer deterministic, finite, inspectable behavior and honest failure.
12. Account for complexity in callers, generated code, memory, synchronization, tooling, evidence, cleanup, and context reconstruction.
13. Keep universal semantics independent of backend shape; specialize generated hot paths.
14. Treat acceleration as a qualified variant, not a semantic requirement.
15. Preserve complete deletion: removing one backend or consumer leaves no solely owned residue.
16. Measure the complete system before making performance or default-selection claims.
