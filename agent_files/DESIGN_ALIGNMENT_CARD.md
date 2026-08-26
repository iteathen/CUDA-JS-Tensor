# Design Alignment Card

For every reusable boundary answer:

1. What invariant does it own?
2. Which materially different consumers belong to the same equivalence class?
3. What is explicitly excluded?
4. What happens when the first consumer is deleted?
5. Which public dependency port does it use?
6. What finite memory, workspace, synchronization, failure, and cleanup plan applies?
7. Which convenience overloads normalize to the same canonical record?
8. Which defaults are selected, inspectable, overridable, and identity-affecting?
9. What is the complete non-accelerated behavior?
10. What exact evidence permits an accelerated variant to be recommended?

Reject the design if it embeds a model/training/search domain, requires CUDA-JS internals, hides device ownership, makes tensor acceleration mandatory, or adds hot-path selection work that can be resolved before execution.
