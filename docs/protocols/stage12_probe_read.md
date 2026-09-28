# Stage 12 Protocol Draft: Probe Reading On Natural Compliance Failures

Draft only. Not executed. Written to be pasted into `README.md` after Stage 11
if the gate is opened.

## Bounded Question

Stage 11 produced closed-loop states in which the unmodified pi0.5 policy
grasps the object that the instruction did not name, with no hook, patch, or
intervention of any kind. Stage 12 asks one read-only question about those
states:

> At the first policy query, before any arm motion, does the fitted
> action-expert goal-state probe decode the object named by the instruction,
> or the object the policy subsequently grasps?

This is not a reopening of the Stage 11 language-conflict branch, which
remains closed. Stage 11 asked whether a behavioral asymmetry generalizes
across object pairs and answered no. Stage 12 asks whether an already-fitted
representation readout agrees with behavior on the specific states where
behavior is known to be wrong. No steering, layer search, or intervention is
authorized by this stage regardless of outcome.

## Why This Stage Is Worth Running

Every causal result in Stages 5 through 9 depends on an intervention operator,
and therefore on the objection that the operator itself broke the computation.
Stage 8 answers that objection with a positive control but cannot remove it.
Stage 12 requires no operator: the policy runs untouched under the official
sampler, and the only added computation is a forward hook that stores
residuals. Whatever it finds is free of operator-validity concerns.

## Read Location And Primary Endpoint

The probe is read at the **first policy query of the rollout**, before any
action is executed.

This is a preregistered constraint, not a convenience. The fitted probes
decode target-relative quantities. Reading them near the grasp is circular:
once the gripper has approached object A, `target-to-EEF` necessarily points at
object A regardless of what the instruction-conditioned representation
contains. At the first query the physical state, image, and robot state are
identical between the aligned and swapped prompt conditions, so the only
difference is the instruction.

For the same reason the primary probe is **target XYZ**, the absolute position
of the instructed object. At the first query the end effector is at its home
pose in both conditions, so target XYZ is the readout least contaminated by
arm position. `target-to-basket XYZ` is a preregistered secondary. Probes whose
definition depends on the current EEF pose or on rollout progress
(`target-to-EEF`, `target displacement`, `coarse phase`) are **excluded from the
primary endpoint** and reported as descriptive secondaries only.

## Decision Variable

For each rollout, decode target XYZ from the action-expert block-13 token-mean
readout at the first policy query, then assign a discrete probe target:

```text
probe_target = argmin over {A, B} of || decoded_xyz - ground_truth_xyz(object) ||
```

Ground-truth object positions at the initial state are already recorded by the
Stage 11 harness. A rollout is scored `probe_indeterminate` if the two
distances differ by less than 2 cm; indeterminate rollouts are reported and
excluded from the primary rate.

Rollouts are then cross-tabulated:

|  | Behavior grasps instructed object | Behavior grasps other object |
|---|---|---|
| Probe points at instructed object | consistent-correct | **dissociation** |
| Probe points at grasped object | — | consistent-wrong |

The primary quantity is the **dissociation rate**: among rollouts whose first
grasp is the object the instruction did not name, the fraction whose probe
points at the instructed object.

## States

Primary analysis set: pair 0 (`alphabet_soup_vs_cream_cheese`), side-A anchor
scene, swapped prompt B. Stage 11 recorded 80% first-grasp A, 6% first-grasp
B, 14% no grasp over 50 initial states, giving approximately 40 wrong-object
rollouts. A Wilson interval on a proportion at `n=40` separates 0.70 from 0.30
comfortably, so this set is adequate for the decision rule below.

Paired reference set: the same 50 initial states under the aligned prompt A,
where Stage 11 measured 98% compliance. These supply the probe-validity floor.

Secondary set: pair 3 (`tomato_sauce_vs_butter`), side-A anchor scene, swapped
prompt B. Its dominant failure is omission (50% grasped neither object) rather
than commission, so it answers a different question and is descriptive only.
Rollouts with no grasp are reported separately in all sets and never enter the
primary rate.

Rollouts are regenerated from the Stage 11 initial-state ids and the same
state- and query-indexed deterministic noise, so behavior reproduces exactly.
Reproduction is audited: any rollout whose first-grasp outcome differs from its
Stage 11 record invalidates the run.

## Probe Provenance And Leakage Control

Two probes are read, both already fitted, neither refitted for this stage:

1. **Primary, pooled probe.** The Stage 4 / Stage 10 action-expert block-13
   target-XYZ probe. Before any result is interpreted, the Stage 11 initial
   state ids must be verified disjoint from the episodes used to fit and
   validate this probe. If disjointness cannot be established from the cached
   split metadata, the primary read is declared invalid and the stage reports
   only the LOPO probe.
2. **Secondary, LOPO probe.** The leave-pair-0-out probe from Stage 10, which
   has never seen this object pair. Stage 10 measured its cross-pair
   target-XYZ transfer at `+0.049` pair-macro, so it is weak, and it is used
   only as a robustness check on the direction of the primary result.

If the two probes disagree in direction, the stage reports inconclusive.

## Preregistered Decision Rule

**Validity floor, evaluated first.** On the aligned prompt A reference set,
the probe must point at the instructed object in at least 90% of compliant
rollouts. This confirms the probe transfers to these unseen initial states at
all. If the floor fails, the stage stops and reports a probe-transfer failure.
No dissociation number is interpreted.

**Primary rule**, evaluated only if the floor passes, on the wrong-object
rollouts of pair 0, with a 95% Wilson interval on the dissociation rate:

- interval lower bound above 0.70: **dissociation**. The decoded goal state
  identifies the instructed object while the action goes elsewhere. Decodable
  goal state is not the quantity driving object selection at this decision
  point.
- interval upper bound below 0.30: **probe tracks behavior**. The decoded goal
  state already names the object that will be grasped, locating the failure
  upstream of the action expert, in prompt integration rather than in the
  action pathway.
- any other interval: **inconclusive**. Reported as such, with no mechanistic
  claim.

The thresholds, the read location, the primary probe, the analysis set, and
the validity floor are fixed before execution and are not revised after seeing
results.

## Secondary Descriptive Outputs

Not gated, reported for completeness:

- probe target assignment at every policy query along the trajectory, to show
  whether an initially correct assignment decays or is wrong from the first
  query;
- the same first-query read at PaliGemma prefix prompt end, which localizes
  any dissociation to the prefix-expert interface;
- pair 3 first-query reads, including the no-grasp rollouts;
- decoded-vs-ground-truth XYZ error magnitudes, not only the discrete
  assignment.

## Stop Condition

Stage 12 terminates at the decision rule under every outcome. A dissociation
result does not authorize a new intervention search; it is observational
evidence bearing on the Stage 5 semantic-attribution result. A
probe-tracks-behavior result does not authorize prefix steering. If the
validity floor fails, the stage reports that the existing probes do not
transfer to fresh official initial states, which is itself a bounded and
reportable limitation of Stages 2 through 10.

## Implementation Notes

No new probe training, no layer selection, no intervention.

| Need | Existing entry point |
|---|---|
| Rollout with residual capture, no patching | `vla_coordinates/pi05_runtime.py:783` `sample_actions_capture_residuals` |
| Expert block token pooling | `vla_coordinates/pi05_runtime.py:459` `extract_layer_tokens` |
| Prefix read for the secondary | `vla_coordinates/pi05_runtime.py:75` |
| Stage 11 states, noise, first-grasp records | `scripts/eval_pi05_instruction_compliance.py` |
| Object ground-truth positions at init | already logged by the Stage 11 harness |
| Fitted probe weights | Stage 4 / Stage 10 atlas artifacts |

Expected new code is one evaluation script that replays Stage 11 rollouts with
capture enabled and applies stored probe weights, plus one summary script.
Both mirror existing Stage 11 scripts. Compute is 100 primary rollouts plus 50
secondary, with no training.

The audits below are recorded in the run summary and are pass/fail:

- `patching=false` and only capture hooks registered;
- first-grasp outcomes reproduce the Stage 11 records exactly;
- initial-state disjointness from probe fitting and validation episodes;
- probe weights loaded from stored artifacts, with checksums, and not refitted.

## Amendment 1 (2026-09-28, before execution)

Written after the smoke run (job `1824888`, one paired state, used only to
check reproduction and hooks) and before any probe was applied to a Stage 12
residual.

### Disjointness audit result

The pooled Stage 10 probes fail the preregistered disjointness check. The
Stage 10 v2 trajectories used all 50 official initial states of every task,
split 35/7/8 by episode. For both analysis anchors, 42 of the 50 Stage 11
initial-state ids appear in probe training or validation episodes:

| Pair | Anchor task | Stage 11 ids | In probe train/valid | Disjoint (Stage 10 test) |
|---|---:|---:|---:|---:|
| 0 | 0 | 50 | 42 | 8 |
| 3 | 5 | 50 | 42 | 8 |

Per the rule above, the pooled primary read is invalid. It is reported only as
a descriptive, contaminated readout.

### Replacement primary probe: leave-anchor-scene-out (LASO)

The primary probe is refit with the Stage 10 pooled recipe unchanged
(`train_openvla_world_state_atlas.py`, paired prompt mode, identical
hyperparameters, seeds 42, 43, 44), except that every episode of the anchor
task is removed from training and validation (`--exclude-task-ids 0` for pair
0, `--exclude-task-ids 5` for pair 3). The pair's objects remain seen through
the opposite-side task scene; none of the 50 anchor initial states is seen.
The refit's test split is the anchor task's 8 locked Stage 10 test episodes and
serves only as a descriptive transfer check on cached data.

Fixed read, not revised after seeing results:

- readout: action-expert block 13, ten-token mean, first denoising evaluation
  (`t=1`), first policy query;
- decoded target XYZ: mean of the three seed probes' decoded XYZ; per-seed
  assignments are reported as a sensitivity check;
- assignment, 2 cm indeterminate margin, Wilson intervals, and the 0.70 / 0.30
  decision thresholds: unchanged.

The v2 validation-selected expert layer is 16, not 13. Block 13 stays primary
because it was fixed by this protocol; block 16 is reported descriptively.

### Secondary probes

- LOPO (unchanged): the Stage 10 leave-pair-out block-13 probes, three-seed
  mean. The direction veto ("if the two probes disagree in direction, report
  inconclusive") applies only if LOPO passes the same validity floor; a LOPO
  probe that fails the floor is reported as a transfer failure and has no
  direction to compare.
- PaliGemma prefix: LASO refit at prefix block 12 (the Stage 7A block),
  descriptive only.
- Pooled: descriptive only, labelled contaminated.

### Validity floor clarification

The floor is computed over determinate compliant aligned rollouts. If more
than half of the compliant aligned rollouts are indeterminate, the floor fails.

### Rollout set clarification

Pair 3 is replayed under both prompts, not only prompt B, so that the rollout
loop is identical to Stage 11. The 50 added pair-3 aligned rollouts are
descriptive only.

## Result (2026-09-28)

Jobs: formal rollout array `1824918` (RTX 3080, pairs 0 and 3), LASO probe
array `1824924` (A100, 12 fits), summary run on the locked outputs. Artifacts:
`artifacts/pi05_stage12_probe_read/`.

### Audits

- All 200 rollouts reproduce their Stage 11 outcome exactly (first-grasp side
  and step, executed steps, policy queries, grasp and basket flags, BDDL
  success).
- Hooked versus unhooked official sampler at the first query: maximum
  absolute action difference `0.0`.
- Every query fired one prefix hook and ten expert hooks per block; no patching.
- Probe weights were loaded from stored files with SHA256 recorded in
  `summary.json`; nothing was refit on Stage 12 data.

Pre-read transfer check on cached Stage 10 data (descriptive, per Amendment 1):
the LASO block-13 probes have negative within-task target-XYZ R² on the
anchor task's test episodes (task 0: −0.48 to −0.51; task 5: −0.34 to −0.40).
Within one scene the target varies over a few centimetres, so R² penalises any
constant offset heavily; the two-object assignment is the operative read.

### Preregistered decision: inconclusive

Pair 0, side-A scene, first policy query:

| Probe | Validity floor (aligned, compliant) | Wrong-object rollouts: probe at instructed object | Rate, 95% Wilson |
|---|---:|---:|---:|
| **LASO expert b13 (primary)** | 49/49, passed | 23/29 determinate (11 indeterminate) | 0.79 [0.62, 0.90] |
| LOPO expert b13 | 49/49, passed | 10/30 (10 indeterminate) | 0.33 [0.19, 0.51] |

The primary lower bound (0.62) does not exceed 0.70. LOPO passed its floor and
points in the opposite direction, which triggers the direction veto. Per-seed
LASO rates are unstable (0.73, 0.40, 0.89). No mechanistic claim is made.

Descriptive readouts (not gated): LASO expert b16 29/35 (0.83); LASO prefix
b12 22/37 (0.59); pooled b13 39/40 (0.97), contaminated by fitting on these
initial states and shown only to illustrate the size of that bias. Pair 3 has
only 2 wrong-object rollouts; its dominant failure is omission (25/50 no
grasp), so its dissociation rate is undefined in practice.

### Exploratory, not preregistered

For each of the 50 pair-0 states, the first-query decoded target is projected
on the object A (0) to object B (1) axis under both prompts. Swapping the
instruction from A to B moves the decoded target towards B in **40/40**
wrong-object states for every unseen-state probe:

| Probe | Mean shift, wrong-object states [95% bootstrap] | All 50 states |
|---|---:|---:|
| LASO expert b13 | 0.68 [0.65, 0.72] | 0.69 |
| LOPO expert b13 | 0.40 [0.37, 0.43] | 0.41 |
| LASO prefix b12 | 0.67 [0.56, 0.78] (39/40 positive) | 0.69 |
| LASO expert b16 | 0.70 [0.66, 0.74] | 0.71 |

The shift is as large in states where pi0.5 then grasps the wrong object as
across all states. The instruction therefore moves the internal goal estimate
most of the way towards the named object before any motion, and this movement
does not distinguish rollouts that obey from rollouts that do not. Because the
shift ends near the midpoint rather than at object B, a discrete two-object
read is inherently fragile here, which is consistent with the inconclusive
preregistered outcome. This supports a partial "heard but not obeyed" reading
as a hypothesis for a future preregistered test, not as a finding.
