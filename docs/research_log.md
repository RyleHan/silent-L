# Research Log: VLA Coordinate Systems and World-State Atlas

> **Chronological research log.** This is the original stage-by-stage lab
> notebook (Stages 1–11), kept verbatim apart from relative link fixes. The
> project summary is in the top-level [README](../README.md); the Stage 12
> preregistration is in [protocols/stage12_probe_read.md](protocols/stage12_probe_read.md).
> Cluster paths below refer to the NHR@FAU TinyGPU setup used for all runs.

Independent research project testing whether VLA residual streams represent task
state more linearly in an instruction-relative coordinate system than in an
absolute object-identity coordinate system.

## Stage 1 Question

For the same LIBERO-Object image and two swapped instructions, compare matched
three-class linear probes:

- Absolute: `OTHER / ALPHABET_SOUP / CREAM_CHEESE`
- Relative: `OTHER / TARGET / ALTERNATIVE`

The relative labels swap classes 1 and 2 when the prompt changes. The absolute
labels do not. Both probes use the same images, masks, splits, architecture,
optimizer, and number of classes.

The main OpenVLA readout is the residual at the final prompt token (and later
action positions), decoded from `[4096]` to `[16, 16, 3]`. Visual patch residuals
are a negative architectural control because causal attention places the prompt
after the image tokens.

## Cluster Layout

```text
HOME=<cluster home>
WORK=<cluster work filesystem>

$HOME/code/vla_coordinates                  # this project
$HOME/code/vla_coordinates_vendor/openvla
$HOME/code/vla_coordinates_vendor/mech_int_othelloGPT
$HOME/code/vla_coordinates_vendor/othello_world
$HOME/code/vla_coordinates_vendor/emergent_openvla
$HOME/code/vla_coordinates_vendor/mechanistic-steering-vlas
$HOME/code/vla_coordinates_vendor/openpi
$HOME/code/LIBERO                           # shared, read-only

$WORK/vla_coordinates/envs/openvla-object
$WORK/vla_coordinates/envs/othello-ref
$WORK/vla_coordinates/envs/pi05-atlas
$WORK/vla_coordinates/envs/mechanistic-openvla
$WORK/vla_coordinates/checkpoints/pi05_libero_pytorch
$WORK/vla_coordinates/cache
$WORK/vla_coordinates/datasets
$WORK/vla_coordinates/reference/othello
$WORK/vla_coordinates/environment-locks
$WORK/vla_coordinates/runs
$WORK/vla_coordinates/logs
```

Pinned source revisions:

```text
OpenVLA              c8f03f48af692657d3060c19588038c7220e9af9
LIBERO               8f1084e3132a39270c3a13ebe37270a43ece2a01
mech_int_othelloGPT  58aa116a18076e040c8b828b81a31e8799bced4c
othello_world         f23bb5696cf30b93bd8af8a391ee33fc3aac417e
emergent_openvla     5bcf77240dda9bf4fa84af240ab3764185662e65
mechanistic-steering 559c0f25a3cc5a20fc8b804774a88415404e0d22
openpi                15a9616a00943ada6c20a0f158e3adb39df2ccac
```

## OpenVLA Runtime

```bash
source $HOME/code/vla_coordinates/cluster/env_openvla.sh
```

Checkpoint:

```text
openvla/openvla-7b-finetuned-libero-object
revision 287d6cfdf12d07b1449505f66d9bf3550257e9b3
```

The 14.049 GiB snapshot is cached locally. GPU jobs set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

V100 inference uses FP16 and eager attention. The project runtime explicitly
appends OpenVLA's empty action token (`29871`) and extends the attention mask at
the same time. The upstream `predict_action()` appends the token without
extending the mask, which only becomes an error outside FlashAttention 2.

Verified baseline:

- Real LIBERO-Object task 0, initial state 0
- Success at action step 146
- Mean inference after warm-up: 0.288 seconds/action
- Peak allocated CUDA memory: 14.41 GiB
- Result: `$WORK/vla_coordinates/runs/openvla-object-rollout-1766464.json`

## pi0.5 Runtime

The pi0.5 experiments use a separate environment and cache; the OpenVLA and
thesis environments are not modified:

```bash
bash cluster/setup_pi05_env.sh
source cluster/env_pi05.sh
```

The environment uses Python 3.11.15, Torch 2.7.1+cu126, Transformers 4.53.2,
and JAX 0.5.3. Exact locks are under `environment-locks/pi05-atlas/`. The
official `pi05_libero` JAX checkpoint is downloaded with per-object GCS MD5
verification and converted with OpenPI's official conversion code. The final
PyTorch checkpoint is:

```text
$WORK/vla_coordinates/checkpoints/pi05_libero_pytorch/model.safetensors
SHA256 047b5d0f437abd77f93e22bdf2e731877f0bae155fb5093b2fe907ef342ee609
```

## Othello Reference

```bash
source $HOME/code/vla_coordinates/cluster/env_othello.sh
```

The reference environment uses Torch 2.0.1+cu118, TransformerLens 1.6.0, and
Transformers 4.32.1. Exact locks are under `environment-locks/`.

The synthetic OthelloGPT checkpoint has SHA256:

```text
aef97afb1e7ec9dbef1d9007174458830a5c428d62f1e2338128879b179bd5c4
```

Matched linear-probe reproduction on 10,000 train and 1,000 validation games:

| Labels | Best layer | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| `EMPTY / BLACK / WHITE` | 0 | 0.7522 | 0.6891 |
| `EMPTY / MINE / YOURS` | 6 | 0.9919 | 0.9902 |

Metrics:
`$WORK/vla_coordinates/reference/othello/probe_runs/job-1766491/metrics.json`

## Static Reset-State Pilot

Object pair:

```text
A = alphabet_soup_1
B = cream_cheese_1
```

The pair co-occurs in LIBERO-Object task layouts 0 through 4 and occupies varied
target and distractor slots.

Layout-held-out split:

```text
train = task layouts 0, 2, 4
valid = task layout 1
test  = task layout 3
```

Each simulator state stores one center-cropped RGB image, instance masks for A
and B, the full simulator state, and both prompts as HDF5 metadata. Prompt pairs
are always kept in the same split because they are derived from the same image.

Run:

```bash
sbatch.tinygpu cluster/jobs/generate_libero_object_pairs.sbatch
```

Output:

```text
$WORK/vla_coordinates/datasets/libero_object_pairs_v1/paired_states.h5
```

This reset-state dataset is retained as a pilot, not as the main coordinate
experiment. Although all 1,000 RGB frames are unique, each task layout yields
only one to four distinct 16-by-16 patch-label maps. Holding out a whole layout
therefore withholds positive training examples for its output cells, and the
non-shared OthelloGPT-style grid decoder cannot identify them. Results under
that split must not be interpreted as evidence for or against either coordinate
system.

## Dynamic Trajectory Data

The main experiment uses OpenVLA rollouts from the two LIBERO-Object tasks that
move the selected object:

```text
task 0 = alphabet soup -> basket
task 1 = cream cheese -> basket
```

Frames and instance masks are sampled every four action steps. Train,
validation, and test are split by complete rollout episode, so no same-trajectory
frames cross a split. The paired counterfactual prompts remain identical to the
static pilot.

Run:

```bash
sbatch.tinygpu cluster/jobs/generate_libero_object_trajectories_smoke.sbatch
sbatch.tinygpu cluster/jobs/generate_libero_object_trajectories.sbatch
```

Full output:

```text
$WORK/vla_coordinates/datasets/libero_object_trajectories_v1/paired_trajectories.h5
```

The 12-episode smoke run produced 478 frames and 117 distinct patch-label maps.
The full run produced 1,946 frames from 50 episodes, with 37 successful
rollouts and train/validation/test frame counts of 1,392/223/331. The moving
alphabet-soup and cream-cheese masks produced 146 and 102 distinct patch-label
maps. Only one test cream-cheese cell lacked a positive training example.

## OpenVLA Residual Probes

Residual extraction uses forward hooks on all 32 Llama decoder blocks, matching
Emergent OpenVLA's post-block `residual_output` convention. It stores two text
readouts for each image/prompt pair:

```text
prompt_end       = final prompt token before the empty action token
action_boundary  = empty action token used to predict the first action token
```

The mean over the 256 visual-token residuals is stored once per image as a
causal negative control. The first image is evaluated under both prompts to
verify that visual residuals are invariant up to floating-point kernel error
while text readouts differ substantially.

Run:

```bash
sbatch.tinygpu cluster/jobs/extract_openvla_residuals_smoke.sbatch
sbatch.tinygpu cluster/jobs/extract_openvla_residuals.sbatch
```

Static-pilot output:

```text
$WORK/vla_coordinates/datasets/libero_object_pairs_v1/openvla_residuals.h5
```

The main dynamic residual cache is written to:

```text
$WORK/vla_coordinates/datasets/libero_object_trajectories_v1/openvla_residuals.h5
```

The linear decoder follows the OthelloGPT probe shape with no bias:

```text
[4096] -> [16, 16, 3]
```

Each 14-by-14 image patch is labeled from the transformed instance masks using
a five-pixel minimum. Absolute and target/alternative probes share the same
initial weights, optimizer, batches, foreground balancing, capacity, and
data split. The main experiment holds out complete episodes. Layers are selected
on validation data only; the held-out test episodes are reported after selection.

Run all three readouts (`prompt_end`, `action_boundary`, and `visual_mean`):

```bash
sbatch.tinygpu cluster/jobs/train_openvla_coordinate_probes.sbatch
sbatch.tinygpu cluster/jobs/train_openvla_trajectory_coordinate_probes.sbatch
```

### Dynamic Probe Result

Layers below were selected by the validation difference between relative and
absolute foreground macro-F1. Test results were then read once at that fixed
layer. Confidence intervals use 10,000 bootstrap resamples of the eight complete
held-out test episodes.

| Readout | Layer | Absolute | Target/alternative | Difference | 95% CI |
|---|---:|---:|---:|---:|---:|
| Prompt end | 23 | 0.9482 | 0.9469 | -0.0013 | [-0.0055, 0.0046] |
| Action boundary | 22 | 0.9494 | 0.9484 | -0.0010 | [-0.0073, 0.0052] |
| Visual mean control | 2 | 0.9333 | 0.4625 | -0.4708 | [-0.4753, -0.4667] |

Both coordinate systems are strongly linearly decodable after the instruction,
but this experiment does not reproduce OthelloGPT's large relative-coordinate
advantage. The prompt-blind visual control decodes absolute identity while
failing on the contradictory target/alternative assignment, confirming that
the paired-prompt test is sensitive to instruction-relative information.

Results:

```text
$WORK/vla_coordinates/runs/openvla_trajectory_coordinate_probes/summary/
artifacts/openvla_trajectory_results/
```

## Stage 2: OpenVLA World-State Atlas

Stage 1 did not reproduce OthelloGPT's relative-coordinate advantage at the
16-by-16 patch-label level. Stage 2 asks the broader question supported by both
OthelloGPT and Emergent OpenVLA: which global world-state factors are linearly
readable across OpenVLA layers, tokens, and time, and which factors are actually
conditioned on the instruction?

### Simulator-Grounded Labels

The existing 1,946 trajectory states are replayed with LIBERO's
`set_init_state()`. This updates MuJoCo, observables, object states, and task
predicates without rerunning OpenVLA. The label file includes:

- Absolute geometry: alphabet-soup, cream-cheese, and end-effector positions.
- Robot state: seven joint positions and gripper width.
- Goal-relative geometry: target/alternative positions, target-to-EEF and
  target-to-basket vectors and distances.
- Dynamics: per-episode target displacement and finite-difference speed.
- Simulator state: robosuite grasp checks for both objects.
- Explicit heuristics: near-target, near-basket, moved, lifted, and a coarse
  `approach / pregrasp / manipulation` phase.

Heuristic thresholds and definitions are stored in HDF5 metadata. The source
rollout stops immediately after a successful action and does not cache the
post-success state. Consequently, `in_basket` and `task_success` have zero
positive cached frames and are not probed. This atlas makes no claim about
decoding the completed phase.

Run:

```bash
sbatch.tinygpu cluster/jobs/inspect_libero_world_state.sbatch
sbatch.tinygpu cluster/jobs/extract_libero_world_state_labels_smoke.sbatch
sbatch.tinygpu cluster/jobs/extract_libero_world_state_labels.sbatch
```

Output:

```text
$WORK/vla_coordinates/runs/openvla_world_state_atlas/world_state_labels.h5
$WORK/vla_coordinates/runs/openvla_world_state_atlas/world_state_labels_summary.json
```

### Probe Protocols

Each layer receives matched linear heads for standardized continuous factors,
binary state factors, and the three-class coarse phase. Feature and target
normalization use training episodes only. Optimization and capacity are fixed
across all 32 layers. Layers and binary thresholds are selected on validation
episodes, followed by one report on the locked test episodes.

Two complementary protocols are retained:

1. `actual`: one residual per image under the instruction used for the rollout.
2. `paired`: every image appears under both swapped prompts. Absolute labels
   repeat, while target/alternative labels swap within the pair. One shared
   linear probe must fit both assignments. The prompt-invariant visual mean is
   duplicated as an exact negative control.

Run:

```bash
sbatch.tinygpu cluster/jobs/train_openvla_world_state_atlas.sbatch
sbatch.tinygpu cluster/jobs/train_openvla_world_state_atlas_paired.sbatch
sbatch.tinygpu cluster/jobs/bootstrap_openvla_world_state_atlas.sbatch
```

### Actual-Instruction Atlas

Validation-selected test results show that global physical state is strongly
linearly decodable at all three readouts:

| Factor | Prompt end | Action boundary | Visual mean |
|---|---:|---:|---:|
| Target XYZ, R2 | 0.967 | 0.970 | 0.967 |
| EEF XYZ, R2 | 0.973 | 0.968 | 0.976 |
| Target-to-EEF XYZ, R2 | 0.906 | 0.865 | 0.913 |
| Target displacement XYZ, R2 | 0.949 | 0.955 | 0.947 |
| Target grasped, AUROC | 0.992 | 0.970 | 0.989 |
| Coarse phase, macro-F1 | 0.931 | 0.902 | 0.928 |

These results establish decodability, but actual-target labels alone are not a
clean test of language conditioning: the moving object and rollout progress
are visible in the image.

### Paired-Prompt Atlas

The paired protocol separates physical-state visibility from instruction-based
target remapping:

| Factor | Prompt end | Action boundary | Visual mean control |
|---|---:|---:|---:|
| Absolute object A XYZ, R2 | 0.951 | 0.958 | 0.968 |
| EEF XYZ, R2 | 0.971 | 0.976 | 0.978 |
| Target XYZ, R2 | 0.895 | 0.858 | 0.297 |
| Target-to-EEF XYZ, R2 | 0.899 | 0.866 | 0.432 |
| Target-to-basket XYZ, R2 | 0.908 | 0.873 | 0.284 |
| Target displacement XYZ, R2 | 0.880 | 0.847 | 0.333 |
| Coarse phase, macro-F1 | 0.866 | 0.880 | 0.611 |

Absolute object and EEF factors are already within 90% of their validation peak
after decoder block 0. Prompt-aware target position, target-to-EEF, target
displacement, and coarse phase reach 90% of their own peak by blocks 2 through
4, then continue to refine toward factor-specific optima in later layers. The
visual control also peaks early, but at a much lower target-relative ceiling.

Confidence intervals use 10,000 cluster-bootstrap resamples of the eight held-out
test episodes. Prompt-end minus visual-control differences are:

| Factor | Test difference | 95% CI |
|---|---:|---:|
| Target XYZ, R2 | +0.598 | [0.550, 0.668] |
| Target-to-EEF XYZ, R2 | +0.466 | [0.427, 0.545] |
| Target-to-basket XYZ, R2 | +0.624 | [0.575, 0.696] |
| Target displacement XYZ, R2 | +0.547 | [0.429, 0.620] |
| Coarse phase, macro-F1 | +0.255 | [0.190, 0.311] |

The absolute object and robot state remains highly decodable from the visual
stream, while target-centric state requires the prompt-aware residual. This is
positive evidence that OpenVLA contains a linearly accessible,
instruction-conditioned, goal-centric world-state representation. It does not
yet establish that the representation is causally used, that intervening on it
is safe, or that extracting it can improve policy success.

Results:

```text
$WORK/vla_coordinates/runs/openvla_world_state_atlas/
artifacts/openvla_world_state_atlas/
```

Presentation figures are available as 300-DPI PNG and vector PDF files:

```text
artifacts/openvla_world_state_atlas/figures/paired_headline_scores.*
artifacts/openvla_world_state_atlas/figures/paired_layer_curves.*
artifacts/openvla_world_state_atlas/figures/paired_layer_heatmap.*
artifacts/openvla_world_state_atlas/figures/paired_episode_bootstrap.*
```

## Stage 3: pi0.5 Paired-Prompt Transfer

Stage 3 transfers the Stage 2 paired-prompt atlas to the official
`pi05_libero` policy. This is an architecture-aware transfer rather than a raw
copy of the OpenVLA hook locations:

- pi0.5 uses an 18-block PaliGemma prefix model with hidden size 2048 and has no
  OpenVLA-style action-boundary token.
- The prompt-aware readout is the final language token after each PaliGemma
  decoder block and before the final norm.
- PaliGemma image tokens can attend to language after entering the decoder, so
  a post-block visual mean would not be prompt blind. The negative control is
  instead the mean valid projected SigLIP image token before any language
  decoder block. Its features are asserted to be exactly identical for the two
  prompts paired with each physical state.

### Input Reconstruction

The same 1,946 MuJoCo trajectory states are replayed with the official pi0.5
LIBERO input contract: base image, wrist image, and eight-dimensional robot
state. This produces 3,892 examples after pairing each state with the two
swapped instructions. Complete episodes remain in the same train, validation,
or test split; frame counts remain 1,392/223/331.

```bash
sbatch.tinygpu cluster/jobs/prepare_pi05_checkpoint.sbatch
sbatch.tinygpu cluster/jobs/prepare_pi05_libero_inputs_smoke.sbatch
sbatch.tinygpu cluster/jobs/prepare_pi05_libero_inputs.sbatch
sbatch.tinygpu cluster/jobs/extract_pi05_prefix_residuals_smoke.sbatch
sbatch.tinygpu cluster/jobs/extract_pi05_prefix_residuals.sbatch
```

The full residual cache has shape `[1946, 2, 18, 2048]`. A replay audit on 16
samples found a mean absolute base-image difference of 0.139 on the 0--255
pixel scale, confirming alignment with the original physical states.

### Probe Protocol And Result

The transfer reuses the Stage 2 labels, episode split, linear heads,
optimization, validation-only layer selection, and locked test evaluation.
Only the model-specific readouts change. Best layers below are zero-indexed
PaliGemma decoder blocks selected on validation data.

| Factor | Prompt end | Layer | SigLIP control | Difference |
|---|---:|---:|---:|---:|
| Absolute object A XYZ, R2 | 0.923 | 0 | 0.923 | +0.000 |
| EEF XYZ, R2 | 0.954 | 0 | 0.955 | -0.001 |
| Target XYZ, R2 | 0.659 | 12 | 0.267 | +0.392 |
| Target-to-EEF XYZ, R2 | 0.703 | 8 | 0.423 | +0.279 |
| Target-to-basket XYZ, R2 | 0.661 | 13 | 0.285 | +0.376 |
| Target displacement XYZ, R2 | 0.686 | 12 | 0.259 | +0.427 |
| Coarse phase, macro-F1 | 0.798 | 12 | 0.629 | +0.169 |

Confidence intervals use 10,000 cluster-bootstrap resamples of the eight
held-out test episodes:

| Factor | Test difference | 95% CI | P(difference > 0) |
|---|---:|---:|---:|
| Target XYZ, R2 | +0.392 | [0.242, 0.552] | 1.0000 |
| Target-to-EEF XYZ, R2 | +0.279 | [0.185, 0.387] | 1.0000 |
| Target-to-basket XYZ, R2 | +0.376 | [0.209, 0.529] | 0.9997 |
| Target displacement XYZ, R2 | +0.427 | [0.370, 0.509] | 1.0000 |
| Coarse phase, macro-F1 | +0.169 | [0.125, 0.218] | 1.0000 |

The sanity controls behave as intended: absolute object and EEF geometry are
equally readable with or without language, while target-relative factors gain
substantially after the prompt is integrated. The strongest goal-conditioned
readouts concentrate in middle-to-late blocks 8--13.

This reproduces the Stage 2 qualitative result in a second, substantially
different VLA architecture: pi0.5 also contains a linearly accessible,
instruction-conditioned, goal-centric representation of the current task
state. It does not rescue the Stage 1 claim that an OthelloGPT-style relative
patch coordinate system is superior, and it does not yet establish learned
dynamics, causal use by the action policy, or a policy-performance benefit.
Raw scores should not be treated as an OpenVLA-versus-pi0.5 leaderboard because
the architecture-correct prompt-blind controls occur at different locations.

Results and presentation figures:

```text
$WORK/vla_coordinates/runs/pi05_world_state_atlas/
artifacts/pi05_world_state_atlas/
artifacts/pi05_world_state_atlas/figures/paired_headline_scores.*
artifacts/pi05_world_state_atlas/figures/paired_layer_curves.*
artifacts/pi05_world_state_atlas/figures/paired_layer_heatmap.*
artifacts/pi05_world_state_atlas/figures/paired_episode_bootstrap.*
```

The next experimental gate is the pi0.5 action expert at a fixed denoising
time. Stage 4 below implements that gate.

## Stage 4: pi0.5 Action-Expert Transfer

Stage 4 tests whether the goal-conditioned state found in the PaliGemma prefix
survives into pi0.5's actual action computation. The official PyTorch path has
an important architecture detail: with `pi05=True`, the suffix contains ten
noisy-action tokens and no separate state token. Time conditions the 18-block,
1024-wide Gemma expert through AdaRMS. Although the LIBERO policy interface
accepts an eight-dimensional robot state, the official `pi05_libero` config
uses `discrete_state_input=False` and does not embed that state as a pi0.5
suffix token.

### Deterministic Protocol

The main protocol evaluates the first inference vector field at `t=1.0`. This
choice is both on-path and clean: at the start of flow matching, `x_t` is pure
Gaussian noise, so no demonstration action is supplied. Every physical state
receives a deterministic state-indexed noise sample that is exactly shared by
its two swapped prompts. Two readouts are cached after every expert block:

- `expert_action_mean`: mean residual over all ten action tokens.
- `expert_action_first`: residual at the first action token.

The smoke test repeated the complete forward pass and obtained exact equality
for both residual readouts and the predicted velocity. Swapping only the prompt
changed the expert-mean residual by 0.592 mean absolute units and changed the
first-step velocity field by 0.014 mean absolute units on the first audited
state. The full cache contains 1,946/1,946 states:

```text
expert_action_mean_residuals   [1946, 2, 18, 1024]
expert_action_first_residuals  [1946, 2, 18, 1024]
expert_velocity                [1946, 2, 10, 32]
```

Run:

```bash
sbatch.tinygpu cluster/jobs/extract_pi05_action_expert_residuals_smoke.sbatch
sbatch.tinygpu cluster/jobs/extract_pi05_action_expert_residuals.sbatch
sbatch.tinygpu cluster/jobs/train_pi05_action_expert_atlas_paired.sbatch
sbatch.tinygpu cluster/jobs/bootstrap_pi05_action_expert_atlas.sbatch
sbatch.tinygpu cluster/jobs/compare_pi05_prefix_action_expert.sbatch
sbatch.tinygpu cluster/jobs/plot_pi05_action_expert_atlas.sbatch
```

### Transfer Result

The same paired labels, episode splits, linear probes, validation selection,
locked test, and visual-input control are reused without modification:

| Factor | PaliGemma prompt end | Expert token mean | Expert first token | Visual control |
|---|---:|---:|---:|---:|
| Target XYZ, R2 | 0.659 | 0.644 | 0.602 | 0.267 |
| Target-to-EEF XYZ, R2 | 0.703 | 0.664 | 0.630 | 0.423 |
| Target-to-basket XYZ, R2 | 0.661 | 0.646 | 0.612 | 0.285 |
| Target displacement XYZ, R2 | 0.686 | 0.598 | 0.587 | 0.259 |
| Coarse phase, macro-F1 | 0.798 | 0.797 | 0.722 | 0.629 |

Expert-token mean minus visual-control differences remain positive under 10,000
cluster-bootstrap resamples of the eight held-out episodes:

| Factor | Test difference | 95% CI |
|---|---:|---:|
| Target XYZ, R2 | +0.377 | [0.303, 0.479] |
| Target-to-EEF XYZ, R2 | +0.240 | [0.167, 0.323] |
| Target-to-basket XYZ, R2 | +0.361 | [0.276, 0.465] |
| Target displacement XYZ, R2 | +0.339 | [0.303, 0.400] |
| Coarse phase, macro-F1 | +0.168 | [0.115, 0.248] |

A second paired bootstrap directly compares validation-selected expert and
prefix probes on the same held-out episodes. Expert-token mean minus PaliGemma
prompt end is:

| Factor | Test difference | 95% CI | Interpretation |
|---|---:|---:|---|
| Target XYZ, R2 | -0.015 | [-0.101, 0.080] | No resolved change |
| Target-to-EEF XYZ, R2 | -0.039 | [-0.090, 0.018] | No resolved change |
| Target-to-basket XYZ, R2 | -0.015 | [-0.093, 0.095] | No resolved change |
| Target displacement XYZ, R2 | -0.088 | [-0.149, -0.031] | Lower in expert |
| Coarse phase, macro-F1 | -0.001 | [-0.045, 0.058] | No resolved change |

The first action token is weaker than the ten-token mean. Relative to prefix,
its target-to-EEF R2 decreases by 0.073 with CI [-0.138, -0.003], and phase
macro-F1 decreases by 0.076 with CI [-0.098, -0.048]. This is consistent with
goal state being distributed across the action chunk rather than concentrated
at its first token.

The strict conclusion is that instruction-conditioned goal state reaches the
action-expert residual stream and is robustly decodable there. It is mostly
preserved, not sharpened, across the prefix-to-expert interface at the first
denoising step. This rejects the initial hypothesis that pi0.5's lower prefix
score is simply explained by the state becoming much more explicit only in the
expert. It does not prove that the decoded variables are causally used, because
the current experiment observes activations and prompt-dependent velocity but
does not intervene on a state feature. It also makes no claim about later
denoising times.

Results and figures:

```text
$WORK/vla_coordinates/runs/pi05_action_expert_atlas/
artifacts/pi05_action_expert_atlas/
artifacts/pi05_action_expert_atlas/figures/prefix_expert_headline_scores.*
artifacts/pi05_action_expert_atlas/figures/prefix_expert_layer_curves.*
artifacts/pi05_action_expert_atlas/figures/expert_minus_prefix_bootstrap.*
artifacts/pi05_action_expert_atlas/figures/expert_action_mean_heatmap.*
```

If a later sparse decomposition is warranted, the action-token mean atlas makes
expert block 13 the primary SAE candidate for target position,
target-to-basket, displacement, and phase; block 9 is the factor-specific
target-to-EEF candidate. SAE is not a prerequisite for the causal gate below.

## Stage 5: Natural Counterfactual Action-Expert Patching

Stage 5 replaces the earlier plan to make SAE replication a prerequisite. Its
bounded question is whether the same-state, prompt-induced action-expert state
causally mediates the corresponding change in pi0.5's action vector field.
For each physical state, prompts A and B use exactly the same deterministic
Gaussian noise at `t=1.0`. The base branch is patched at one expert block with
the natural paired residual difference from the other branch:

```text
h_base' = h_base + alpha * (h_source - h_base)
```

The patch is applied separately at all ten action-token positions. The source
branch stays in the same batch as an untouched determinism control. Both A to B
and B to A directions are evaluated. The preregistered coarse sweep is:

```text
expert blocks = 0, 5, 9, 13, 17
alpha         = 0, 0.25, 0.5, 0.75, 1
```

Block 17 is an endpoint positive control and is excluded from layer selection:
at `alpha=1`, replacing its residual with the source residual is followed only
by the shared linear action projection, so near-perfect action recovery is
algebraically expected. Layer and alpha selection among blocks 0, 5, 9, and 13
uses validation episodes only, minimizing normalized error to the natural
source-prompt velocity. Test episodes remain locked.

Action recovery is the projection of the intervention-induced velocity change
onto the natural prompt-swap velocity change:

```text
R_action = <v_patch - v_base, v_source - v_base>
           / ||v_source - v_base||^2
```

The analysis also reports cosine alignment, intervention norm, orthogonal
distortion, and normalized source error. Episode-cluster bootstrap confidence
intervals use complete held-out test episodes.

Smoke tests on one state in both directions passed exact isolation audits:

```text
alpha=0 base-velocity max absolute difference       0.0
untouched source-velocity max absolute difference  0.0
peak allocated CUDA memory                         7.26 GiB
```

Run:

```bash
sbatch.tinygpu cluster/jobs/extract_pi05_paired_patching_smoke.sbatch
sbatch.tinygpu cluster/jobs/extract_pi05_paired_patching.sbatch
sbatch.tinygpu cluster/jobs/summarize_pi05_paired_patching.sbatch
```

The full paired patch is a causal mediation test for the complete
prompt-conditioned residual difference, not yet a semantic attribution to the
decoded goal-state subspace. Goal-subspace, nullspace, shuffled, and random
controls are admitted only if an intermediate expert block passes this first
offline action-recovery gate.

### Offline Causal Result

Both directions independently selected expert block 13 and `alpha=1` on
validation episodes. On the 331 locked test states from eight complete
episodes, the same-state natural patch recovered a positive fraction of the
natural prompt-swap action-vector-field difference:

| Direction | Action recovery | 95% episode-bootstrap CI | Direction cosine |
|---|---:|---:|---:|
| A to B | 0.212 | [0.155, 0.287] | 0.631 |
| B to A | 0.225 | [0.194, 0.270] | 0.673 |

The effect is dose-responsive on validation data. Blocks 0 and 5 have almost
no effect, block 9 has a small effect, and block 13 has the strongest
nontrivial recovery. Block 17 reaches one by construction and remains an
endpoint control rather than evidence for intermediate causal mediation.

The selected block-13 patch then received exactly one fixed semantic-control
evaluation. The probe-defined goal subspace is the rank-30 row space of the
effective paired-atlas decoders for target/alternative geometry, target
dynamics, target binary state, and coarse phase. Per-token deltas are projected
onto that subspace or its orthogonal complement. Shuffled controls use a
different episode from the same task and split at matched trajectory progress;
random controls preserve each action token's delta norm.

| Patch at block 13 | A to B recovery | B to A recovery |
|---|---:|---:|
| Same-state paired | 0.212 | 0.225 |
| Probe goal subspace | 0.004 | 0.007 |
| Probe nullspace | 0.211 | 0.222 |
| Episode-shuffled | 0.076 | 0.077 |
| Norm-matched random | 0.007 | 0.005 |

Same-state paired minus shuffled recovery is +0.136 with CI [0.084, 0.212]
for A to B and +0.148 with CI [0.112, 0.205] for B to A. The effect is
therefore substantially state-specific and cannot be explained by residual
norm or an arbitrary perturbation. However, removing the rank-30 probe goal
subspace leaves nearly all recovery: paired minus null is +0.001 with CI
[-0.001, 0.004] for A to B and +0.003 with CI [0.001, 0.006] for B to A.

The strict conclusion is narrower than the initial causal hypothesis. A
natural, state-specific, prompt-conditioned residual difference at expert
block 13 causally shifts the first pi0.5 vector field toward the natural
counterfactual action. The causal effect is not carried by the low-dimensional
row space selected by the existing supervised goal-state probes. Consequently,
this stage does not establish that the specifically decoded goal variables are
the mechanism used by the policy. It establishes causal controllability of the
full natural counterfactual residual and a negative semantic attribution result
for this probe-defined subspace.

Results and figures:

```text
$WORK/vla_coordinates/runs/pi05_paired_patching/
artifacts/pi05_paired_patching/base-{0,1}/
artifacts/pi05_paired_patching/control-base-{0,1}/
artifacts/pi05_paired_patching/figures/validation_dose_response.*
artifacts/pi05_paired_patching/figures/locked_test_controls.*
```

The next bounded gate is closed-loop target switching with the already selected
block-13 natural patch. Its claim, if positive, is behavioral controllability
of the natural counterfactual state, not direct causal use of the probe-defined
goal subspace and not policy-performance improvement.

### Closed-Loop Behavioral Gate

The closed-loop test fixes expert block 13 and `alpha=1` from the offline
validation result. No closed-loop episode is used for layer, strength, or
metric selection. Each LIBERO-Object direction uses 20 initial states that were
not used by the trajectory atlas. Every initial state is replayed under four
conditions:

| Condition | Prompt supplied to base branch | Expert residual intervention |
|---|---|---|
| Correct | Environment target | None |
| Mismatch | Alternative object | None |
| Paired | Alternative object | Same-state natural delta toward target prompt |
| Random | Alternative object | Per-token norm-matched random delta |

The paired and random patches are applied at all ten flow-matching steps. A
state-, query-, and task-indexed Gaussian noise tensor is shared across prompt
branches and conditions. The policy replans every five actions for at most 280
environment actions. The official sampler audit is exact in all four
conditions: unpatched branches have a maximum absolute action-chunk difference
of zero.

The pre-specified primary comparisons are paired versus mismatch (behavioral
effect) and paired versus random (direction specificity). Success, target
grasp, target placement, alternative grasp, minimum target-to-EEF distance, and
target displacement are reported separately for both directions with 95%
paired-initial-state bootstrap confidence intervals. Correct-prompt behavior is
a natural reference, not an intervention baseline.

Run:

```bash
sbatch.tinygpu cluster/jobs/eval_pi05_paired_patch_closed_loop_smoke.sbatch
sbatch.tinygpu cluster/jobs/eval_pi05_paired_patch_closed_loop.sbatch
sbatch.tinygpu cluster/jobs/analyze_pi05_paired_patch_closed_loop.sbatch
```

#### Closed-Loop Result

All 160 episodes completed: 20 paired initial states, two target directions, and
four conditions per state. Correct-prompt success was 20/20 in both tasks,
confirming that the held-out states are solvable by the unmodified policy. All
four sampler audits were exactly zero in both directions.

| Target direction and condition | Success | Target grasp | Target in basket | Target min EEF distance |
|---|---:|---:|---:|---:|
| B to A, correct | 1.00 | 1.00 | 1.00 | 0.018 m |
| B to A, mismatch | 0.80 | 0.85 | 0.80 | 0.048 m |
| B to A, paired patch | 0.20 | 0.45 | 0.20 | 0.098 m |
| B to A, random patch | 0.80 | 0.85 | 0.80 | 0.046 m |
| A to B, correct | 1.00 | 1.00 | 1.00 | 0.020 m |
| A to B, mismatch | 0.00 | 0.00 | 0.00 | 0.204 m |
| A to B, paired patch | 0.00 | 0.00 | 0.00 | 0.225 m |
| A to B, random patch | 0.00 | 0.00 | 0.00 | 0.215 m |

For B to A, paired minus mismatch success is -0.60 with 95% paired-bootstrap
CI [-0.80, -0.40]; target grasp is -0.40 [-0.60, -0.20]. Paired minus random
gives the same differences. For A to B, paired patching changes none of the
three binary target outcomes relative to mismatch or random. It instead
increases target distance: the favorable paired-minus-mismatch distance
difference is -0.021 m [-0.037, -0.010], and paired minus random is -0.010 m
[-0.015, -0.006].

The target-switching result is therefore negative in both directions, with
asymmetric failure modes. In B to A, the mismatched policy already retains a
strong alphabet-soup preference, while the natural patch substantially damages
otherwise successful behavior. In A to B, the mismatched policy reliably
selects the alternative alphabet soup and neither natural nor random patching
redirects it to cream cheese. The offline block-13 effect is thus real at the
first action vector field but does not support behavioral target control when
reapplied throughout closed-loop denoising. Decodability and local causal action
steering do not imply useful closed-loop control or policy improvement.

This triggers the pre-specified stop condition. The project does not tune a
closed-loop layer or strength after observing these results and does not expand
into SAE replication. The probe-defined goal subspace already failed semantic
mediation offline, and the stronger full natural residual failed the behavioral
gate.

Results and figures:

```text
$WORK/vla_coordinates/runs/pi05_paired_patching/closed_loop/
artifacts/pi05_paired_patching/closed-loop/summary/summary.json
artifacts/pi05_paired_patching/closed-loop/summary/metrics.csv
artifacts/pi05_paired_patching/closed-loop/summary/figures/closed_loop_outcomes.*
artifacts/pi05_paired_patching/closed-loop/summary/figures/closed_loop_primary_effects.*
```

## Stage 6: Write-Native Steering Calibration

Stage 5 is closed as a negative result for dense, state-specific action-expert
residual patching. Its layer, coefficient, and test conclusions remain locked;
Stage 6 does not reopen that sweep. The new bounded question is whether the
published FFN intervention from *Mechanistic Interpretability for Steering
Vision-Language-Action Models* produces its intended semantic behavioral effect
on this cluster stack.

This calibration is necessary because the two interventions write into the
model differently:

```text
Stage 5: add a dense paired residual at pi0.5 expert block 13
Stage 6: overwrite selected OpenVLA FFN activations, then let their native
         down-projection value vectors write into the residual stream
```

The official repository is pinned at
`559c0f25a3cc5a20fc8b804774a88415404e0d22`. It publishes `up_10`, `up_20`,
and `careful_10_full` clusters, but not the paper's `fast` and `slow` OpenVLA
clusters. Therefore the first calibration uses the released `up_10` cluster
and does not reconstruct an unpublished cluster.

### Stage 6A Preregistered Positive Control

- Model: `openvla/openvla-7b-finetuned-libero-10`, revision
  `80970322773f81baa2e22fe495d0487b93a05cfa`.
- Environment: LIBERO-Long (`libero_10`).
- Conditions: unmodified baseline, official `up_10`, and a deterministic
  layer-histogram-matched random 10-neuron control.
- Intervention: official `down_proj` forward hook with coefficient `4.0`, as
  provided by the repository example configuration.
- Matching: conditions replay the same task, initial simulator state, seed,
  preprocessing, and deterministic action decoding.
- Primary behavioral readout: signed per-step EEF Y displacement, matching the
  paper's reported directional readout. XYZ displacement, path length, raw and
  executed actions, task success, hook counts, and videos are retained as
  audits.

The official implementation assumes H100 BF16 inference. The school V100
compatibility layer changes only numerical execution to FP16 with eager
attention, using the already verified OpenVLA runtime. Semantic neuron indices,
activation overwrite semantics, coefficient, prompts, and environment behavior
remain unchanged.

The execution gate is deliberately staged:

1. One fixed task/state smoke test checks model loading, exact state replay,
   hook execution, finite actions, trajectory logging, and a nonzero action
   change under intervention. A single trajectory is not an effect estimate.
2. A ten-task matched pilot compares baseline, `up_10`, and random on one fixed
   initial state per task.
3. Only if the released semantic intervention is distinguishable from the
   matched random control is a locked multi-state confirmation run admitted.

The pilot gate is fixed before execution: the mean `up_10 - baseline` signed
per-step EEF Y effect must be positive, and the two-sided 95% task-bootstrap CI
for `up_10 - random` must lie strictly above zero. All task-level replay and
hook audits must also pass.

Interpretation and stop rules:

- Positive semantic control: the infrastructure is calibrated; Stage 5 then
  supports a basis/operator failure for dense residual patching, not a general
  impossibility of VLA steering.
- Action changes without the predicted directional behavioral effect: the hook
  is active, but the published effect is not reproduced on this hardware and
  software stack. Stop before target-object transfer.
- No action change: treat the setup as invalid and debug only implementation
  fidelity; do not interpret it scientifically.
- Target-object steering is Stage 6B and is admitted only after Stage 6A. SAE
  remains optional and is not a prerequisite.

### Stage 6A Smoke Result

The infrastructure smoke completed on A100 job `1774102` because the V100 queue
was fully allocated. Numerical execution remained the preregistered FP16/eager
path. All three conditions replayed the same initial EEF state exactly.

```text
up_10 selected-layer hook counts                 all positive (280 each)
random selected-layer hook counts                all positive (280 each)
up_10 raw actions differing from baseline        40 / 40
random raw actions differing from baseline       37 / 40
up_10 mean raw-action L2 from baseline            0.121
random mean raw-action L2 from baseline           0.136
peak allocated CUDA memory                        14.41 GiB
```

The single trajectory is not a semantic effect estimate. It establishes that
the official intervention is active, the released semantic cluster changes
actions, the matched random control is comparably perturbative, and the pilot
must discriminate direction rather than merely detect action change.

Run the bounded pilot and its locked summary with:

```bash
sbatch.tinygpu cluster/jobs/eval_openvla_ffn_steering_pilot.sbatch
sbatch.tinygpu --dependency=afterok:<array-job-id> \
  cluster/jobs/summarize_openvla_ffn_steering_pilot.sbatch
```

Smoke artifacts:

```text
$WORK/vla_coordinates/runs/mechanistic_steering_openvla/job-1774102-task-0-state-0/
artifacts/mechanistic_steering_openvla/smoke/
```

## Stage 7: Language Dependence and Native Pathway Gate

Stage 6 did not reproduce the released `up_10` directional effect and is not
expanded further. Stage 7 asks a narrower question that directly resolves an
ambiguity in Stage 5: did paired residual patching fail because target language
is not behaviorally used, or because the intervention bypassed the model's
native language-to-action computation?

### Stage 7A: Correct, Swapped, and Target-Underspecified Language

The two named instructions remain fixed. The third condition is the grammatical
but target-underspecified instruction:

```text
pick up an object and place it in the basket
```

For every one of the existing 1,946 states, image, wrist image, robot state,
noise seed, and fixed denoising time `t=1` remain identical. Only the prompt is
changed. The null condition caches PaliGemma prompt-end residuals, action-expert
mean and first-token residuals, and the action-expert vector field in one
forward path. No probe is retrained: the paired-atlas weights, normalization,
factor slices, validation-selected layers, and test split are locked.

The primary offline readout is target XYZ on the locked test split. A null
prediction is projected onto the line from the object-A-prompt prediction
(`0`) to the object-B-prompt prediction (`1`). Target-to-EEF and phase are
secondary readouts. Initial frames are reported separately because they remove
trajectory-state evidence about which object the expert demonstration was
already manipulating.

Closed-loop evaluation adds only the null condition on the same 20 unused
initial states per task used by Stage 5. Correct and swapped results are reused
from the locked Stage 5 rollouts, producing 40 new episodes rather than
rerunning 80 old ones. Success, target and alternative grasp, placement,
minimum EEF distance, and object displacement are paired by initial-state ID.

Stage 7B is admitted only if Stage 7A establishes both of the following:

1. Named target language creates a nontrivial A-versus-B separation in the
   locked probe and vector-field endpoints.
2. Null language reveals a stable default preference that agrees between an
   internal action readout and closed-loop object selection, while the opposite
   named prompt can overturn that preference on the matched states.

If admitted, Stage 7B contains one fixed native-pathway replay experiment, not
a layer or coefficient sweep. It will use the independently selected
PaliGemma target-XYZ layer and Action Atlas's full-output overwrite principle,
then allow all downstream pi0.5 computation to proceed normally. If either
Stage 7A criterion fails, the project stops with the current decodability,
local action causality, and closed-loop intervention boundary.

The Action Atlas reference was audited at commit
`b8b0db331df18fc30a3fd92c45ec721d35d3ee52`. Its public cross-task injection
code sequentially overwrites complete layer outputs with source-run
activations; it does not publish a separate same-scene pi0.5 target-switching
protocol.

Smoke jobs `1774737` and `1774738` passed on RTX 3080 GPUs. The combined null
extractor captured prefix, expert, and vector-field outputs with exact repeated
inference; both one-state closed-loop tasks completed and the official sampler
audit was exactly zero.

### Stage 7A Result and Stop Decision

Formal jobs `1774823` through `1774826` completed successfully. The offline
analysis used all 1,946 cached states and reports the locked 331-state test
split; the closed-loop analysis adds 20 null-prompt episodes per task and reuses
the paired Stage 5 correct and swapped baselines.

On the A-to-B interpolation axis, `0` is the object-A-prompt prediction and `1`
is the object-B-prompt prediction. The target-unspecified instruction lands on
the B side at every internal readout:

| Locked test readout | Null interpolation, mean [95% episode-bootstrap CI] | Null closer to A |
|---|---:|---:|
| PaliGemma target-XYZ probe, block 12 | 0.684 [0.634, 0.727] | 0.139 |
| Action-expert target-XYZ probe, block 13 | 0.648 [0.620, 0.680] | 0.082 |
| Action-expert vector field at `t=1` | 0.562 [0.528, 0.607] | 0.417 |

The preference persists when trajectory-state evidence is removed. Across all
50 initial frames, the corresponding null interpolation means are 0.912,
0.588, and 0.737. These initial-frame results are descriptive because the
locked test split contains only eight initial frames. R-squared values are not
reported for constant target dimensions; RMSE and the interpolation statistics
remain defined.

Closed-loop behavior does not follow a single B-default target variable:

| Physical target | Correct success | Swapped success | Null success | Null target grasp | Null alternative grasp |
|---|---:|---:|---:|---:|---:|
| A, alphabet soup | 1.00 | 0.80 | 0.85 | 0.80 | 0.00 |
| B, cream cheese | 1.00 | 0.00 | 0.80 | 0.80 | 0.15 |

Thus Stage 7A criterion 1 passes: named instructions produce separable target
representations through the backbone, action expert, and first vector-field
evaluation. Criterion 2 fails: the internal null preference is B-like, but the
same null instruction selects the task-associated object in both task-specific
state distributions. Language control is also asymmetric: the A instruction
overturns B-task behavior, whereas the B instruction usually does not overturn
A-task behavior.

Stage 7B is therefore not admitted. A native-pathway overwrite would no longer
test a clean, preregistered hypothesis connecting one internal target variable
to object selection. The bounded project stops with the stronger conclusion
that pi0.5 contains an instruction-sensitive, linearly decodable goal state
that reaches its action expert and changes its local vector field, while
closed-loop target selection combines that state with scene or task priors in
an asymmetric way. Decodability, pathway propagation, and local action
sensitivity do not establish a single causally sufficient goal-state control
axis.

Results and figures:

```text
$WORK/vla_coordinates/runs/pi05_language_dependence/
artifacts/pi05_language_dependence/offline/summary.json
artifacts/pi05_language_dependence/offline/offline_language_dependence.*
artifacts/pi05_language_dependence/closed_loop/summary/summary.json
artifacts/pi05_language_dependence/closed_loop/summary/metrics.csv
artifacts/pi05_language_dependence/closed_loop/summary/closed_loop_language_dependence.*
```

Run:

```bash
sbatch.tinygpu cluster/jobs/extract_pi05_language_null_residuals.sbatch
sbatch.tinygpu cluster/jobs/analyze_pi05_language_dependence.sbatch
sbatch.tinygpu cluster/jobs/eval_pi05_language_null_closed_loop.sbatch
sbatch.tinygpu cluster/jobs/summarize_pi05_language_dependence_closed_loop.sbatch
```

## Stage 8: Full Expert-Pathway Replay Positive Control

Stage 8 is a fixed operator-validity control added after Stage 7, not a reopening
of the rejected Stage 7B target-axis hypothesis. It addresses the remaining
criticism of Stage 5: replacing only block 13 creates a hybrid state in which a
natural source residual is processed under the base prompt's remaining
context. Closed-loop failure could therefore reflect the intervention operator
rather than the absence of causally useful internal state.

The protocol is fixed before execution:

- Model and environment: the existing pi0.5 LIBERO-Object checkpoint and two
  paired object tasks.
- Base branch: the mismatched prompt for the current physical target.
- Source branch: the correct prompt on the identical image, wrist image, robot
  state, and deterministic flow-matching noise.
- Intervention: overwrite the base branch's complete output at all 18
  action-expert blocks with the corresponding source-branch output at every one
  of the 10 denoising steps.
- Scope: one full-expert layer group, coefficient implicitly fixed to complete
  replacement, with no layer or strength sweep.
- Closed loop: the same 20 held-out initial states per task used in Stage 5.

This follows Action Atlas's full-output replay principle but uses a stricter
same-state paired-prompt source. It is intentionally a positive control: full
replay is expected to reproduce the source policy, not discover a new steering
direction.

Implementation validity requires all 18 hooks to fire exactly once per layer
per denoising step and the untouched source branch to match the official
sampler exactly. The positive control passes if replayed velocity and action
chunks match the natural source within maximum absolute error `1e-6`, and
closed-loop success matches the correct-prompt baseline in both directions.

Interpretation is fixed:

- Pass: full natural pathway replay is a valid steering operator; the Stage 5
  negative result is specific to single-block intervention and its hybrid
  downstream computation. Remaining direction asymmetry belongs to the native
  policy and task/scene priors.
- Fail with valid hooks: the action expert is not a causally closed replay
  boundary under prompt changes; prefix/VLM context must participate in a valid
  intervention.
- Invalid source or hook audits: debug implementation only and draw no
  scientific conclusion.

Run:

```bash
sbatch.tinygpu cluster/jobs/eval_pi05_full_pathway_replay_smoke.sbatch
sbatch.tinygpu cluster/jobs/eval_pi05_full_pathway_replay.sbatch
sbatch.tinygpu cluster/jobs/summarize_pi05_full_pathway_replay.sbatch
```

### Stage 8 Result

Smoke job `1774900`, formal array job `1774910`, and dependent summary job
`1774911` completed successfully. The preregistered positive-control gate
passed without a numerical tolerance edge case:

- maximum replay-versus-source action error: `0.0`;
- maximum replay-versus-source velocity error: `0.0`;
- untouched source-versus-official-sampler error: `0.0`;
- replay-versus-official-source error: `0.0`;
- every one of the 18 action-expert blocks fired exactly 10 hooks per policy
  query, one at each denoising step.

Closed-loop results on 20 held-out initial states per target were:

| Physical target | Correct | Mismatch | Single block 13 | Full pathway |
|---|---:|---:|---:|---:|
| A, alphabet soup | 1.00 | 0.80 | 0.20 | 1.00 |
| B, cream cheese | 1.00 | 0.00 | 0.00 | 1.00 |

Full pathway replay also achieved target grasp and placement on all 40 trials
and never grasped the alternative object. Its success, target grasp,
placement, and wrong-object outcomes exactly match the natural correct-prompt
baseline. Continuous trajectory diagnostics differ slightly from the earlier,
independently executed correct-prompt rollouts because closed-loop simulator
states can accumulate small numerical differences; this does not affect the
online operator audit, which compares replay and source actions on the same
current observation and is exactly zero throughout.

This result resolves the intervention-method objection in a bounded way.
Single-block replacement is not a faithful proxy for replaying the prompt's
natural action computation: it creates a mixed pathway and can severely damage
behavior even when its offline action moves toward the source. Replacing the
complete action-expert pathway removes that incompatibility and recovers the
source policy exactly. Therefore the Stage 5 closed-loop failure cannot support
the broad claim that the decoded goal state is unusable; it supports only the
narrow claim that one dense residual at one block is not causally sufficient
under the tested replacement operator.

Full-pathway replay is deliberately not claimed as a compact steering method or
a policy improvement. It copies the entire source action computation and thus
serves as an operator positive control. The remaining open problem is to find a
smaller, independently specified intervention that preserves pathway
compatibility while changing target selection. That problem is outside the
fixed scope of this RA project unless a new study is opened explicitly.

Results and figures:

```text
$WORK/vla_coordinates/runs/pi05_full_pathway_replay/
artifacts/pi05_full_pathway_replay/summary/summary.json
artifacts/pi05_full_pathway_replay/summary/metrics.csv
artifacts/pi05_full_pathway_replay/summary/full_pathway_positive_control.*
artifacts/pi05_full_pathway_replay/task-{0,1}/
```

## Stage 9: Bounded COAST Reimplementation Pilot

Stage 9 opens one explicitly bounded follow-up to test a softer replacement
operator. COAST applies a multiplicative conceptor gate to the action expert
instead of overwriting a residual with an activation from another prompt. The
VLA implementation was not available as a conventional official source
repository when this stage was opened, so this is a paper-guided independent
reimplementation, not a code-level reproduction claim. The paper, step-2,000
checkpoint, and pre-collected pi0.5 activation dataset are public.

The protocol is fixed before execution:

- Model: the public `brandonyang/openpi-libero-2000` pi0.5 checkpoint converted
  with the existing OpenPI-to-PyTorch converter. Its local JAX directory name
  intentionally includes `pi05`, because OpenPI revision
  `15a9616a00943ada6c20a0f158e3adb39df2ccac` selects the adaptive-normalization
  conversion branch by checking the path string rather than its `pi05` model
  configuration flag.
- Task: LIBERO-10 KS3, `turn on the stove and put the moka pot on it` (local
  suite index 2).
- Fitting data: the 15 public on-policy episodes for KS3, audited as 8 success
  and 7 failure episodes. Every policy-query residual contains all 10 denoising
  steps at expert layers 0, 5, 11, and 17.
- Representation: mean-pool the 10 action tokens separately at each denoising
  step, then pool all query/denoising vectors within the success or failure
  class.
- Operator: global `C_success AND NOT C_failure` at action-expert layer 5,
  aperture `0.5`, and gate strength `0.1`. These are the paper's reported
  task-specific parameters; there is no local layer, aperture, or strength
  sweep.
- Inference: apply `M = (1-beta) I + beta C` to every action token at every one
  of the 10 denoising steps. The gate matrix is converted to the model working
  dtype immediately before multiplication.
- Evaluation: 30 held-out LIBERO initial states 15 through 44, disjoint from
  the 15 fitting rollouts, with 520 maximum environment steps and replanning
  every 5 actions.
- Conditions: unmodified baseline, COAST, and a stricter random control that
  preserves the COAST conceptor's complete eigenvalue spectrum while randomly
  rotating its eigenvectors.
- Pairing: all conditions use identical initial states and deterministic
  flow-matching noise for each policy query.

Operator validity requires the `beta=0` sampler to match the official sampler
exactly, the hook to fire exactly 10 times per steered query, and the fitted
matrix to satisfy the conceptor symmetry and spectrum audits. The primary
replication criterion is a positive paired-bootstrap 95% lower bound for
COAST-minus-baseline success. Beating the matched-spectrum random control by
the same criterion establishes the stronger direction-specific result. A
positive point estimate whose interval includes zero is reported as
underpowered; a non-positive estimate is a non-replication. The paper reports
KS3 global success `0.53 -> 0.93` on its 30 held-out rollouts, but exact rate
matching is descriptive rather than a hard gate because the released data do
not identify the paper's held-out simulator seeds.

Run:

```bash
# Login node, network I/O only (task-scoped and resumable):
source cluster/env_pi05.sh
HF_HUB_DOWNLOAD_TIMEOUT=120 \
  $PI05_ENV/bin/python scripts/download_coast_public_subset.py \
  --checkpoint-output "$PI05_COAST_JAX_CHECKPOINT" \
  --activation-output "$PI05_COAST_PUBLIC_ROOT" \
  --task-name "$PI05_COAST_TASK_NAME" \
  --max-workers 1

# Slurm jobs, beginning with checkpoint conversion:
sbatch.tinygpu cluster/jobs/prepare_pi05_coast_assets.sbatch
sbatch.tinygpu cluster/jobs/fit_pi05_coast_conceptor.sbatch
sbatch.tinygpu cluster/jobs/eval_pi05_coast_smoke.sbatch
sbatch.tinygpu cluster/jobs/eval_pi05_coast_pilot.sbatch
sbatch.tinygpu cluster/jobs/summarize_pi05_coast_pilot.sbatch
```

### Stage 9 Result

Public-resource preparation and fitting job `1775104`, checkpoint conversion
job `1775166`, smoke job `1775173`, formal evaluation job `1775176`, and
dependent summary job `1775177` completed successfully. The public cache audit
found exactly 15 fitting episodes (8 success, 7 failure) and 1,141 policy-query
residual files. Mean-pooling all action tokens while retaining each denoising
step yielded 4,130 success and 7,280 failure activation vectors.

The operator-validity gate passed:

- `beta=0` matched the official sampler with maximum absolute action error
  `0.0`;
- the action-expert layer-5 hook fired exactly 10 times for every steered
  policy query;
- the fitted AND-NOT conceptor is symmetric and positive semidefinite, with
  eigenvalues in `[2.88e-6, 0.356]` and quota `0.0105`;
- COAST and the matched-spectrum random control changed residuals by similar
  amounts (mean relative changes `0.0953` and `0.0977` in the smoke run), so
  the control does not win merely by being a weaker perturbation.

Closed-loop results on held-out initial-state IDs 15 through 44 were:

| Condition | Success | 95% episode-bootstrap CI |
|---|---:|---:|
| Unmodified baseline | 12/30 = 0.400 | [0.233, 0.567] |
| COAST | 16/30 = 0.533 | [0.367, 0.700] |
| Matched-spectrum random | 13/30 = 0.433 | [0.267, 0.600] |

The paired COAST-minus-baseline gain is `+0.133`, with 95% CI
`[-0.033, 0.300]`. COAST-minus-random is `+0.100`, with CI
`[-0.033, 0.233]`. Relative to baseline, COAST rescued 6 failed states and
harmed 2 successful states (net +4; two-sided exact sign-test `p=0.289`).
Relative to the random control, it rescued 4 and harmed 1 (net +3;
`p=0.375`). The smoke state's baseline-fail / COAST-success / random-fail
pattern reproduced exactly in the formal run.

The preregistered decision is therefore `positive_but_underpowered`. This
paper-guided implementation recovers the predicted ordering and a meaningful
positive point estimate. Unlike the earlier hard single-residual replacement,
this soft subspace gate produces positive closed-loop evidence; because the
task and checkpoint differ, this is not treated as a head-to-head operator
comparison. It does not reproduce the paper's strong KS3 result
(`0.53 -> 0.93`) or establish a statistically resolved gain at 30 states.
Remaining non-code-matched ambiguities include the authors' unreleased VLA
steering implementation, exact held-out seed/reset mapping, and
simulator/runtime versions. The result supports COAST as a promising answer to
the replacement-operator concern, but does not justify a stronger efficacy
claim or a post-hoc parameter sweep.

Results and figures:

```text
$WORK/vla_coordinates/runs/pi05_coast_pilot/
artifacts/pi05_coast_pilot/fit/summary.json
artifacts/pi05_coast_pilot/heldout/episodes.jsonl
artifacts/pi05_coast_pilot/summary/summary.json
artifacts/pi05_coast_pilot/summary/metrics.csv
artifacts/pi05_coast_pilot/summary/coast_pilot.*
artifacts/pi05_coast_pilot/smoke/videos/
```

## Stage 10: Final Multi-Pair Confirmation

Stage 10 is the final scale-up before project closure. It changes statistical
scale and object coverage, not the research question or model architecture. No
new steering direction, layer, or coefficient is selected from its test set.

The LIBERO Python benchmark API was audited at runtime because its task order
does not match the order in `tasks_info.txt`. Each retained pair satisfies two
conditions: both objects occur in both task scenes, and the two task languages
select opposite members of the pair.

| Pair | Side A task | Side B task | Objects |
|---:|---:|---:|---|
| 0 | 0 | 1 | alphabet soup / cream cheese |
| 1 | 2 | 4 | salad dressing / ketchup |
| 2 | 3 | 8 | bbq sauce / chocolate pudding |
| 3 | 5 | 6 | tomato sauce / butter |

Task IDs 7 and 9 are reserved rather than reused in a second pair. The frozen
configuration is `configs/libero_object_pairs_v2.json`; the runtime audit is
written to `$WORK/vla_coordinates/runs/libero_object_pair_audit_v2.json`.

Data protocol:

- 50 deterministic initial states per task, 8 tasks, 400 rollouts total;
- up to 180 policy actions per rollout, caching every fourth action;
- per-task episode split of 35 train, 7 validation, and 8 test episodes;
- aggregate split of 280 train, 56 validation, and 64 test episodes;
- both prompts are evaluated on every cached image and always remain in the
  same episode split;
- all task shards are generated independently and merged only after every
  shard passes its completion and schema checks.

The state count is intentionally not fixed: successful rollouts terminate
early, and frames with an empty object mask are excluded. Episode counts and
all exclusions are reported from metadata rather than inferred from frames.

Confirmatory representation analyses:

1. OpenVLA Othello-style coordinate probes compare absolute object labels with
   pair-local `TARGET / ALTERNATIVE` labels.
2. Goal-state atlases compare prompt-aware residuals with prompt-blind visual
   controls in OpenVLA, the pi0.5 PaliGemma prefix, and the pi0.5 action expert.
3. The principal continuous endpoint is target XYZ R2. Target-to-EEF,
   target-to-basket, displacement, and coarse phase are secondary endpoints.
4. pi0.5 action-expert minus prefix performance is evaluated on exactly the
   same held-out episodes.
5. Leave-one-object-pair-out fits train on three pairs and test on the fourth,
   providing the cross-object generalization test missing from v1.

Every probe family uses seeds 42, 43, and 44. Layers are selected on validation
episodes only. Test uncertainty uses 10,000 episode bootstrap resamples within
each pair, followed by an equal-weight pair macro average, so a pair with more
cached frames cannot dominate the result. The test split remains locked.

The causal conclusion remains bounded by the already calibrated intervention
hierarchy: single-block dense replacement is not sufficient; complete native
action-expert pathway replay is an exact positive control; and a soft COAST
subspace gate gives positive but underpowered closed-loop evidence. Stage 10
does not open a new post-hoc operator search. Its purpose is to determine
whether the representation findings survive broader object coverage and
cross-pair evaluation.

Validation status on 2026-08-13:

- runtime pair audit job `1776078`: passed all 8 configured tasks;
- data/label smoke job `1776080`: 8/8 tasks and 4/4 pairs passed;
- pi0.5 prefix/action-expert smoke job `1776102`: 8/8 states passed;
- formal rollout array job `1776091`: 8/8 tasks completed;
- merged dataset: 400 episodes, 301 successes, and 15,275 cached states;
- split audit: 280/56/64 train/validation/test episodes, with every prompt
  pair kept inside one episode split;
- residual caches: OpenVLA `[15275, 2, 32, 2, 4096]`, pi0.5 prefix
  `[15275, 2, 18, 2048]`, and pi0.5 action expert
  `[15275, 2, 18, 1024]` completed;
- all pooled probes, 3-seed episode bootstraps, and 72 LOPO fits completed;
- coordinate probe repair excluded 10/15,275 frames (0.07%) where object B
  had no positive 16-by-16 patch at the preregistered five-pixel threshold.
  The threshold was not changed, and all excluded source indices are recorded.

### Stage 10 Results

The Othello-style coordinate hypothesis remains a negative result after the
multi-pair scale-up. Values below are the three-seed mean difference in held-out
foreground macro-F1 (`TARGET/ALTERNATIVE - absolute identity`); brackets give
the envelope of the seed-specific 95% episode-bootstrap intervals.

| OpenVLA readout | Relative minus absolute F1 |
|---|---:|
| prompt end | -0.0033 `[-0.0084, 0.0022]` |
| action boundary | -0.0062 `[-0.0122, 0.0023]` |
| visual mean | -0.4696 `[-0.4816, -0.4610]` |

The prompt-aware readouts therefore recover task-relative labels almost as well
as absolute identities, while the prompt-blind visual control cannot know which
object the instruction calls `TARGET`. This establishes instruction-dependent
relabelling, but not an OthelloGPT-like preference for the relative coordinate
system.

In the episode-held-out pooled analysis, target XYZ is consistently more
linearly decodable from prompt-aware state than from the matched visual control:

| Representation | Mean prompt-aware minus visual target-XYZ R2 |
|---|---:|
| OpenVLA prompt end | +0.543 |
| OpenVLA action boundary | +0.495 |
| pi0.5 prefix prompt end | +0.327 |
| pi0.5 action-expert mean | +0.366 |
| pi0.5 action-expert first token | +0.311 |

Every row has a positive point estimate in all three seeds, and every object
pair has a positive pooled gain for the principal prompt-aware readout of each
model. The action-expert mean exceeds the pi0.5 prefix by only +0.039 R2 on
average; one seed interval crosses zero. The first action token does not exceed
the prefix (`-0.016` mean), so propagation into the expert is clear but extra
decodability is modest and readout-dependent.

LOPO changes the interpretation. OpenVLA and the pi0.5 prefix fail to transfer
their target-XYZ linear map to a completely unseen object pair (prompt-aware
minus visual: `-0.348` and `-0.308`). The pi0.5 action-expert mean is stronger:
target-XYZ transfer is positive on three of four held-out pairs (`+0.049`
pair-macro), and target-displacement transfer is positive on all four
(`+0.110` pair-macro). The defensible conclusion is therefore not a universal
world coordinate system: pooled goal-state information contains substantial
pair-specific coding, while the action expert carries the strongest evidence
for a shared, goal-relative state abstraction.

The representation result and the earlier causal tests close different parts
of the question. Full 18-block native-pathway replay exactly recovers the source
policy, proving that internal action-expert state can causally determine the
action when the intervention preserves the computational pathway. A hard
single-block replacement fails closed loop. The soft COAST gate gives a positive
held-out point estimate (`40% -> 53%`, 30 states) but its paired confidence
interval crosses zero. Operator compatibility matters; performance improvement
is promising rather than established.

![Stage 10 multi-pair confirmation and causal boundaries](../artifacts/stage10_v2/figures/stage10_multi_pair_confirmation.png)

Final local artifacts:

```text
artifacts/stage10_v2/summary.json
artifacts/stage10_v2/seed_metrics.csv
artifacts/stage10_v2/figures/stage10_multi_pair_confirmation.{png,pdf,svg}
```

This is the planned stopping point for the RA project. It provides a reproduced
OthelloGPT reference, a controlled negative transfer result, cross-model and
cross-module representation evidence, a strict cross-pair boundary test, and a
calibrated causal intervention hierarchy without opening another post-hoc
steering search.

## Stage 11: Multi-Pair Instruction-Compliance Gate

Stage 11 is a separate, minimal follow-up gate. It does not train a probe,
select a layer, or apply an intervention. Its sole question is whether the
language-conflict asymmetry previously observed for alphabet soup and cream
cheese recurs across the four object pairs.

The fixed 400-rollout protocol is:

- model: the unmodified pi0.5 LIBERO policy;
- one anchor task per pair, fixed to side A in
  `configs/libero_object_pairs_v2.json`;
- all 50 official initial states from that anchor task;
- each state replayed once with the side-A instruction and once with the
  side-B instruction;
- identical state- and query-indexed Gaussian noise across the two prompts;
- official `model.sample_actions`, with no hooks or patching;
- 4 pairs x 50 paired states x 2 prompts = 400 closed-loop rollouts.

The primary endpoint is first-grasp command compliance: whether the first
object grasped is the object named by the instruction. `ever_grasped`, object
placement, BDDL success, minimum EEF distance, displacement, and videos in the
smoke run are retained as secondary audits. First grasp is used because the
original BDDL success predicate still refers to the anchor task target under a
swapped instruction.

Uncertainty is reported with Wilson intervals for each prompt-specific rate
and paired-initial-state bootstrap intervals for aligned-minus-swapped
compliance. The four pairs receive equal weight in the aggregate. Evidence for
a multi-pair phenomenon requires a positive pair-macro interval and a positive
aligned-minus-swapped point difference in at least three of four pairs. If the
legacy pair is the only strong asymmetry, the language-conflict direction
stops. Mixed results admit only a reciprocal-scene confirmation on the affected
pairs, not steering-method development.

This anchor-scene survey deliberately does not claim invariance to task layout.
The opposite scene side is reserved for a reciprocal confirmation on affected
pairs only, so a mixed result does not trigger another full four-pair survey.

Implementation and jobs:

```text
scripts/eval_pi05_instruction_compliance.py
scripts/summarize_pi05_instruction_compliance.py
cluster/jobs/eval_pi05_instruction_compliance_smoke.sbatch
cluster/jobs/eval_pi05_instruction_compliance.sbatch
cluster/jobs/summarize_pi05_instruction_compliance.sbatch
cluster/jobs/eval_pi05_instruction_compliance_reciprocal.sbatch
cluster/jobs/summarize_pi05_instruction_compliance_reciprocal.sbatch
```

Smoke array `1800546` completed all four pairs with one paired state per pair.
It produced eight rollouts, two videos per pair, valid first-grasp records, and
explicit `patching=false` and `hooks_registered=false` audits. The smoke is an
execution check only and is not an effect estimate. Formal array `1800552` and
dependent summary job `1800555` completed on 2026-09-02.

### Stage 11 Results

The side-A anchor survey completed all 400 rollouts. Prompt A is the native
instruction for each anchor scene; Prompt B is the counterfactual target swap.
Rates are first-grasp command compliance over 50 initial states per condition.

| Pair | Prompt A | Prompt B | Prompt-B first-grasp distribution |
|---|---:|---:|---|
| alphabet soup / cream cheese | 98% | 6% | 80% A, 6% B, 14% none |
| salad dressing / ketchup | 98% | 98% | 98% B, 2% none |
| bbq sauce / chocolate pudding | 100% | 100% | 100% B |
| tomato sauce / butter | 100% | 46% | 4% A, 46% B, 50% none |

The equal-weight pair macro is 99.0% for Prompt A and 62.5% for Prompt B. The
aligned-minus-swapped contrast is +36.5 percentage points with a paired
bootstrap 95% interval of `[+32.5, +40.5]`. This aggregate is heterogeneous:
the pair-level contrast is positive for only two of four pairs, so the
preregistered requirement of at least three positive pairs is not met. Pair 0
shows a wrong-object failure, whereas Pair 3 primarily shows failure to grasp
either object. The other two pairs switch targets almost perfectly.

This mixed result activated the reciprocal confirmation gate for Pairs 0 and
3. Array `1800613` evaluated the opposite, side-B anchor scenes with the same
50-by-2 paired protocol (200 additional rollouts); dependent summary job
`1800616` completed successfully.

| Pair, side-B anchor scene | Counterfactual Prompt A | Native Prompt B |
|---|---:|---:|
| alphabet soup / cream cheese | 100% | 100% |
| tomato sauce / butter | 98% | 100% |

The reciprocal pair-macro aligned-minus-swapped contrast is +1.0 percentage
point with a 95% interval of `[0.0, +3.0]`. Both affected pairs therefore
recover near-perfect command compliance in the opposite scene. The original
failures do not follow object identity, do not show that pi0.5 cannot interpret
the cream-cheese or butter instructions, and do not recur as a reversed anchor
preference.

The bounded Stage 11 conclusion is that pi0.5 instruction following is usually
strong but can fail sharply for particular counterfactual instruction-scene
combinations. The evidence does not support a universal language-ignoring or
visual-object-preference claim. Because the two task sides use different
official initial-state distributions, these experiments do not separate a
learned scene shortcut from target geometry or reachability. The most precise
current description is a scene-distribution-dependent counterfactual
compositional-generalization failure. Per the preregistered stopping rule, this
branch closes without steering-method development.

Results and figures:

```text
$WORK/vla_coordinates/runs/pi05_instruction_compliance_v1/summary/
$WORK/vla_coordinates/runs/pi05_instruction_compliance_reciprocal_v1/summary/
artifacts/instruction_compliance_v1.png
artifacts/instruction_compliance_reciprocal_v1.png
```
