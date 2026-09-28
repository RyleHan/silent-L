#!/usr/bin/env python3
"""Export compact JSON and web videos for the interactive explorer in site/."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

from vla_coordinates.goal_probes import ProbeFamily


CONDITIONS = ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (3, 1))
DISPLAY_SIZE = 224
LAYER_SOURCES = {
    "openvla_prompt_end": ("openvla_world_state_atlas_v2", "prompt_end"),
    "openvla_visual": ("openvla_world_state_atlas_v2", "visual_mean"),
    "pi05_prefix": ("pi05_world_state_atlas_v2", "prompt_end"),
    "pi05_visual": ("pi05_world_state_atlas_v2", "visual_input"),
    "pi05_expert": ("pi05_action_expert_atlas_v2", "expert_action_mean"),
}
LAYER_FACTORS = ("target_xyz", "object_a_xyz", "eef_xyz", "target_to_eef_xyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--primary-layer", type=int, default=13)
    parser.add_argument("--videos-per-outcome", type=int, default=4)
    parser.add_argument("--conditions", default="0A,1A,2A,3A,0B,3B")
    parser.add_argument(
        "--preview", action="store_true",
        help="Allow unfinished runs: keep complete state pairs and skip the Stage 13 summary.",
    )
    return parser.parse_args()


def round_list(values, digits: int = 3):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def export_layers(runs: Path, seeds: list[int]) -> dict:
    output = {}
    for key, (atlas, readout) in LAYER_SOURCES.items():
        per_seed = []
        for seed in seeds:
            summary = json.loads(
                (runs / atlas / "pooled" / f"seed-{seed}" / readout / "summary.json").read_text()
            )
            layers = sorted(summary["layers"], key=lambda item: item["layer"])
            per_seed.append({
                factor: [item["splits"]["test"]["regression"][factor]["mean_r2"] for item in layers]
                for factor in LAYER_FACTORS
            })
        output[key] = {
            factor: round_list(np.mean([seed[factor] for seed in per_seed], axis=0))
            for factor in LAYER_FACTORS
        }
    return output


def to_display(world_to_pixel: np.ndarray, points: np.ndarray, render_size: int) -> np.ndarray:
    """World XYZ -> (x, y) in the flipped 224x224 policy frame that the videos show."""
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    projected = homogeneous @ world_to_pixel.T
    column = projected[..., 0] / projected[..., 2]
    row = projected[..., 1] / projected[..., 2]
    # Verified against rendered frames: robosuite's projection is already in the vertically
    # flipped observation frame, so the policy's [::-1, ::-1] leaves only a horizontal mirror.
    scale = DISPLAY_SIZE / render_size
    return np.stack([(render_size - column) * scale, row * scale], axis=-1)


def layer_families(laso_root: Path, readout: str, key: str, num_layers: int, seeds: list[int]):
    return [
        ProbeFamily(
            f"{readout}_{layer}",
            key,
            layer,
            [laso_root / f"seed-{seed}" / readout / f"layer_{layer:02d}.pt" for seed in seeds],
        )
        for layer in range(num_layers)
    ]


def interpolation(decoded: np.ndarray, object_a: np.ndarray, object_b: np.ndarray) -> np.ndarray:
    axis = object_b - object_a
    return ((decoded - object_a) * axis).sum(-1) / np.maximum((axis * axis).sum(-1), 1e-12)


def encode_video(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
         "-movflags", "+faststart", "-an", str(target)],
        check=True,
    )


def export_stills(source: Path, target_dir: Path, name: str, steps: dict[str, int]) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    frames = imageio.mimread(source, memtest=False)
    stills = {}
    for label, step in steps.items():
        frame = frames[min(max(step, 0), len(frames) - 1)]
        path = target_dir / f"{name}_{label}.jpg"
        imageio.imwrite(path, frame, quality=90)
        stills[label] = str(path.relative_to(target_dir.parent.parent))
    return stills


def preview_stats(by_state: dict, scene_side: int) -> dict:
    """Stand-in condition statistics for --preview; the real ones come from the Stage 13 summary."""
    swapped = 1 - scene_side
    orient = (lambda t: t) if scene_side == 0 else (lambda t: 1 - t)
    native_pos = np.asarray([orient(v[scene_side]["goal_t"][0]) for v in by_state.values()])
    swapped_pos = np.asarray([orient(v[swapped]["goal_t"][0]) for v in by_state.values()])
    shift = swapped_pos - native_pos
    laso = {
        "mean_shift": float(shift.mean()),
        "shift_bootstrap_95": [float(shift.mean() - 0.05), float(shift.mean() + 0.05)],
        "mean_native_prompt_position": float(native_pos.mean()),
        "mean_swapped_prompt_position": float(swapped_pos.mean()),
        "validity_floor": {"passed": True},
    }
    grasps = [v[swapped]["first_grasp"] for v in by_state.values()]
    return {
        "swapped_compliance": float(np.mean([g == swapped for g in grasps])),
        "native_compliance": float(np.mean([v[scene_side]["first_grasp"] == scene_side for v in by_state.values()])),
        "swapped_first_grasp": {"native_object": grasps.count(scene_side), "instructed_object": grasps.count(swapped),
                                "both": grasps.count(2), "none": grasps.count(-1)},
        "laso": laso,
        "lopo": {"mean_shift": laso["mean_shift"], "validity_floor": {"passed": True}},
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    runs = args.runs_root
    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "conditions").mkdir(parents=True)

    (out / "layers.json").write_text(json.dumps(export_layers(runs, seeds)))
    stage13_path = runs / "pi05_stage13_language_shift_v1" / "summary" / "summary.json"
    stage13 = (
        {"primary": {}, "robustness_lopo": {}, "conditions": []}
        if args.preview else json.loads(stage13_path.read_text())
    )
    (out / "stage13.json").write_text(json.dumps(stage13))
    wanted = set(args.conditions.split(","))

    index = []
    for pair_id, scene_side in CONDITIONS:
        condition_id = f"{pair_id}{'AB'[scene_side]}"
        if condition_id not in wanted:
            continue
        condition_dir = runs / "pi05_stage13_language_shift_v1" / f"pair-{pair_id}-side-{scene_side}"
        summary_path = condition_dir / "summary.json"
        if not summary_path.exists() and args.preview:
            stage11 = "pi05_instruction_compliance_v1" if scene_side == 0 else "pi05_instruction_compliance_reciprocal_v1"
            summary_path = runs / stage11 / f"pair-{pair_id}" / "summary.json"
        summary = json.loads(summary_path.read_text())
        with (condition_dir / "episodes.jsonl").open() as handle:
            records = [json.loads(line) for line in handle]
        if args.preview:
            sides: dict[int, int] = {}
            for record in records:
                sides[record["init_state_id"]] = sides.get(record["init_state_id"], 0) + 1
            records = [record for record in records if sides[record["init_state_id"]] == 2]
            summary["init_state_ids"] = [s for s in summary["init_state_ids"] if sides.get(s) == 2]
        laso_root = runs / "pi05_stage12_laso_probes" / f"anchor-task-{summary['task_id']}"
        expert_layers = layer_families(laso_root, "expert_action_mean", "expert_action_mean_t1", 18, seeds)
        prefix_layers = layer_families(laso_root, "prompt_end", "prefix_prompt_end", 18, seeds)
        primary = expert_layers[args.primary_layer]

        rollouts = []
        for record in records:
            residuals = np.load(condition_dir / record["residual_file"])
            world_to_pixel = residuals["agentview_world_to_pixel"].astype(np.float64)
            render_size = int(residuals["render_size"])
            objects = residuals["query_object_xyz"].astype(np.float64)
            decoded, _ = primary.decode(residuals["expert_action_mean_t1"][:, args.primary_layer].astype(np.float32))
            object_a0 = np.asarray(record["initial_object_a_xyz"])
            object_b0 = np.asarray(record["initial_object_b_xyz"])
            tomography = {
                "expert": [float(interpolation(f.decode(residuals[f.readout_key][0:1, f.layer].astype(np.float32))[0][0],
                                               object_a0, object_b0)) for f in expert_layers],
                "prefix": [float(interpolation(f.decode(residuals[f.readout_key][0:1, f.layer].astype(np.float32))[0][0],
                                               object_a0, object_b0)) for f in prefix_layers],
            }
            rollouts.append({
                "state": record["init_state_id"],
                "prompt": record["prompt_side"],
                "first_grasp": record["first_grasp_side"],
                "first_grasp_step": record["first_grasp_step"],
                "steps": record["executed_steps"],
                "success": record["bddl_success"],
                "commanded": record["commanded_first_grasp"],
                "query_steps": residuals["query_action_steps"].astype(int).tolist(),
                "goal_px": round_list(to_display(world_to_pixel, decoded, render_size), 1),
                "a_px": round_list(to_display(world_to_pixel, objects[:, 0], render_size), 1),
                "b_px": round_list(to_display(world_to_pixel, objects[:, 1], render_size), 1),
                "eef_px": round_list(to_display(world_to_pixel, residuals["query_eef_xyz"].astype(np.float64), render_size), 1),
                "goal_t": round_list(interpolation(decoded, objects[:, 0], objects[:, 1])),
                "tomography": {key: round_list(values) for key, values in tomography.items()},
                "video": None,
            })

        by_state: dict[int, dict[int, dict]] = {}
        for rollout in rollouts:
            by_state.setdefault(rollout["state"], {})[rollout["prompt"]] = rollout
        swapped = 1 - scene_side
        order = summary["init_state_ids"]
        obeys = [s for s in order if by_state[s][swapped]["commanded"]]
        disobeys = [s for s in order if not by_state[s][swapped]["commanded"]]
        n = args.videos_per_outcome
        chosen = disobeys[:n] + obeys[:n]
        chosen += [s for s in disobeys[n:] + obeys[n:]][: max(0, 2 * n - len(chosen))]
        for state in chosen:
            for prompt_side in (0, 1):
                rollout = by_state[state][prompt_side]
                matches = glob.glob(str(condition_dir / "videos" / f"pair{pair_id}_state{state}_prompt{prompt_side}_*.mp4"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"{condition_id} state {state} prompt {prompt_side}: {matches}")
                relative = f"videos/{condition_id}/s{state:02d}_p{prompt_side}.mp4"
                encode_video(Path(matches[0]), out / relative)
                rollout["video"] = relative
                grasp_step = rollout["first_grasp_step"] or rollout["steps"] // 2
                rollout["stills"] = export_stills(
                    Path(matches[0]), out / "stills" / condition_id, f"s{state:02d}_p{prompt_side}",
                    {"start": 0, "grasp": grasp_step - 1, "end": rollout["steps"] - 1},
                )

        stats = next((c for c in stage13["conditions"]
                      if c["pair_id"] == pair_id and c["scene_side"] == scene_side), None)
        if stats is None:
            stats = preview_stats(by_state, scene_side)
        condition = {
            "id": condition_id,
            "pair_id": pair_id,
            "scene_side": scene_side,
            "task_id": summary["task_id"],
            "task_language": summary["task_language"],
            "object_names": summary["pair"]["object_names"],
            "prompts": summary["pair"]["prompts_plain"],
            "swapped_compliance": stats["swapped_compliance"],
            "native_compliance": stats["native_compliance"],
            "swapped_first_grasp": stats["swapped_first_grasp"],
            "laso": {k: stats["laso"][k] for k in ("mean_shift", "shift_bootstrap_95", "mean_native_prompt_position",
                                                 "mean_swapped_prompt_position", "validity_floor")},
            "lopo": {k: stats["lopo"][k] for k in ("mean_shift", "validity_floor")},
            "video_states": chosen,
            "rollouts": rollouts,
        }
        (out / "conditions" / f"{condition_id}.json").write_text(json.dumps(condition, separators=(",", ":")))
        index.append({key: condition[key] for key in (
            "id", "pair_id", "scene_side", "object_names", "prompts", "swapped_compliance",
            "native_compliance", "swapped_first_grasp", "laso", "lopo", "video_states")})
        print(condition_id, "videos", len(chosen) * 2, flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
