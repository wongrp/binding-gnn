# Y Series: Ablation Controls for Paper

Missing controls needed to complete the ablation table for JACS submission.

Parent architecture: d2 (384 scalars, 5 layers, persistent edges + feedback, reembed_cg onsite).

The ablation in the paper has two parts:
1. **Mechanism ladder** (w7 → b2 → c7 → d1 → d2 → u44): progressive addition of state interaction mechanisms.
2. **Angular descriptor factorial** (d2/u1 × none/atom SH/dual SH): how angular information enters the model.

Two controls are missing from the ladder:

| Config | Description | What it tests |
|--------|-------------|---------------|
| y1 | 2-copy, persist + feedback, onsite=none | Does the edge plumbing help without comparing bound vs unbound? Isolates persistent edges + feedback from onsite comparison. |
| y2 | 1-copy, persist + feedback, reembed_cg onsite | Is the 2-copy architecture necessary? Tests whether a single-copy model with the same mechanisms can match d2. |
| y3 | Config A + dual-scale SH (atom + residue) | Missing cell in the angular factorial. Does residue SH help Config A the same way it helps Config D? |

If y1 ≈ w7 (no improvement from persist+feedback without onsite), it confirms that the edge mechanisms serve the state comparison, not general representation learning.
If y2 << d2, it confirms that maintaining separate bound/unbound representations is essential.
