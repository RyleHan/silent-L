# Stage 13 Protocol: Does the Language Shift Predict Obedience?

Exploratory. Written on 2026-09-28 after Stage 12 and before any Stage 13
rollout or probe read. The analysis below is fixed in advance, but the
hypothesis was suggested by Stage 12's exploratory result, so a positive
outcome is reported as exploratory evidence, not as a confirmatory finding.

## Motivation

Stage 12 found that, at the first policy query, swapping the instruction moves
the decoded goal towards the named object in every wrong-object rollout, but
only 0.40–0.70 of the way. Stage 11 measured swapped-instruction compliance in
six pair-by-scene conditions that range from 6% to 100%. If language acts as
one input that competes with a scene prior, the distance the instruction moves
the internal goal should track how often the robot obeys.

## Conditions

All six Stage 11 conditions, replayed under the Stage 12 read-only capture
(official sampler, same initial states, prompts, and query-indexed noise):

| Condition | Pair | Scene side | Anchor task | Stage 11 swapped compliance |
|---|---:|---:|---:|---:|
| 0A | 0 alphabet soup / cream cheese | A | 0 | 6% |
| 1A | 1 salad dressing / ketchup | A | 2 | 98% |
| 2A | 2 bbq sauce / chocolate pudding | A | 3 | 100% |
| 3A | 3 tomato sauce / butter | A | 5 | 46% |
| 0B | 0 | B | 1 | 100% |
| 3B | 3 | B | 6 | 98% |

Conditions 0A and 3A repeat Stage 12 with videos and the agent-view camera
matrix added; their first-query reads must equal Stage 12's.

## Probe

Leave-anchor-scene-out (LASO) expert block-13 target-XYZ probe for each
condition's anchor task, three-seed mean, fitted exactly as in Stage 12
Amendment 1. New fits: anchor tasks 1, 2, 3, 6. Robustness: the Stage 10
leave-pair-out probe (the same probe serves both scenes of a pair).

## Quantities

For each initial state, the decoded first-query target is projected onto the
axis from the scene's native object (0) to the swapped-instruction object (1).

- `shift` = projection under the swapped instruction minus projection under
  the native instruction.
- Condition-level `mean shift` = mean over the 50 states.
- Compliance = Stage 12/13 swapped-instruction first-grasp compliance, which
  must reproduce Stage 11 exactly.

## Fixed analysis

1. **Primary:** Spearman rank correlation across the six conditions between
   mean shift and swapped compliance, with an exact permutation p-value over
   all 720 orderings (one-sided, positive).
2. **Secondary:** per-rollout AUC of `shift` for predicting commanded first
   grasp under the swapped instruction, pooled across conditions, with a
   condition-cluster bootstrap. Within-condition AUC is reported only where
   each outcome has at least 5 rollouts.
3. **Robustness:** repeat 1 with the leave-pair-out probe.
4. **Validity floor per condition:** aligned compliant rollouts must decode
   the native object in at least 90% of determinate reads; conditions that fail
   are reported and excluded from 1–3.

With six conditions the primary test has little power; the exact
permutation p-value is reported whatever its size, and no condition is
dropped or added after reading results.

## Result (2026-09-28)

Jobs: rollout array `1825079` (RTX 3080, six conditions), LASO probe array
`1825080` (anchor tasks 1, 2, 3, 6). Artifacts:
`artifacts/pi05_stage13_language_shift/`.

### Audits

- All 600 rollouts reproduce their Stage 11 outcomes exactly; hooked versus
  unhooked first-query action difference `0.0`.
- Conditions 0A and 3A reproduce the Stage 12 first-query reads (200 reads,
  maximum difference `5.2e-7`, float16 storage).
- Every LASO probe passes its validity floor (0A 49/49, 1A 42/43, 2A 33/35,
  3A 50/50, 0B 38/39, 3B 42/45). The leave-pair-out probe passes only for 0A,
  3A and 3B.

### Primary: not supported

| Condition | Swapped compliance | Native-prompt position | Swapped-prompt position | Mean shift [95% bootstrap] |
|---|---:|---:|---:|---:|
| 0A alphabet soup scene | 6% | −0.09 | 0.60 | 0.69 [0.66, 0.72] |
| 3A tomato sauce scene | 46% | −0.04 | 0.58 | 0.61 [0.58, 0.64] |
| 3B butter scene | 98% | 0.25 | 0.81 | 0.56 [0.52, 0.59] |
| 1A salad dressing scene | 98% | 0.24 | 0.75 | 0.51 [0.45, 0.57] |
| 0B cream cheese scene | 100% | 0.33 | 0.75 | 0.43 [0.39, 0.47] |
| 2A bbq sauce scene | 100% | 0.33 | 0.60 | 0.26 [0.21, 0.31] |

Positions use the native object as 0 and the swapped-instruction object as 1.

Spearman ρ between mean shift and compliance is **−0.97** (one-sided exact
p for a positive relation = 1.0). The prespecified hypothesis, that a larger
instruction-induced goal shift goes with more obedience, is not supported;
the ordering is reversed. Pooled rollout AUC is 0.23 (condition-cluster 95%
[0.09, 0.51]). The only condition with at least five rollouts of each outcome,
3A, has within-condition AUC 0.80.

Robustness with the leave-pair-out probe (three conditions pass its floor):
ρ = 0.5, p = 0.5. Pair 3 is the only pair whose two scenes both pass that
probe's floor, and there, with one probe shared by both scenes, the obeying
scene shows the larger shift (3B 0.61 vs 3A 0.11), the opposite of the LASO
ordering. The sign of any relation is therefore not stable across probes.

Conclusion: the size of the decoded goal shift does not predict whether π0.5
obeys. Because each condition uses its own LASO probe, cross-condition
magnitudes also carry probe-calibration differences; the negative ρ is not
interpreted as an inverse relation.

### Post-hoc observation (not tested)

The swapped-prompt position is similar across conditions (0.58–0.81). What
separates the two failing conditions is the native-prompt baseline: there the
decoded goal sits on the scene's own object (−0.09, −0.04), while in the four
obeying conditions it already sits 0.24–0.33 of the way to the other object.
A scene prior that locks the goal onto one object is a hypothesis for a future
preregistered test, not a finding.
