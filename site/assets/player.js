// Two synchronised rollouts of the same initial state under two instructions, with the
// action-expert goal probe drawn on top of the agent-view frames.

import { el, svg, orient, outcomeOf, load } from "./core.js";

const FPS = 10;
const TRAIL = 6;

function queryAt(rollout, frame) {
  let q = 0;
  while (q + 1 < rollout.query_steps.length && rollout.query_steps[q + 1] <= frame) q += 1;
  return q;
}

function sayInstruction(condition, promptSide) {
  const o = orient(condition);
  const cls = promptSide === o.native ? "obj-native" : "obj-other";
  const name = condition.object_names[promptSide];
  return el("span", { class: "say" }, "“pick up the ", el("span", { class: cls }, name), "”");
}

function buildMeter(condition) {
  const o = orient(condition);
  const root = svg("svg", { viewBox: "0 0 400 46", class: "meter", preserveAspectRatio: "none" });
  const x = (t) => 40 + Math.max(-0.35, Math.min(1.35, t)) * 320;
  root.append(
    svg("line", { x1: x(0), x2: x(1), y1: 20, y2: 20, stroke: "var(--line-2)", "stroke-width": 2 }),
    svg("line", { x1: x(0.5), x2: x(0.5), y1: 13, y2: 27, stroke: "var(--faint)", "stroke-width": 1, "stroke-dasharray": "2 2" }),
    svg("circle", { cx: x(0), cy: 20, r: 5, fill: "var(--native)" }),
    svg("circle", { cx: x(1), cy: 20, r: 5, fill: "var(--other)" }),
  );
  const labelA = svg("text", { x: x(0) - 6, y: 42, "text-anchor": "start", "font-size": 10, fill: "var(--native)", "font-family": "JetBrains Mono" });
  labelA.textContent = o.nativeName;
  const labelB = svg("text", { x: x(1) + 6, y: 42, "text-anchor": "end", "font-size": 10, fill: "var(--other)", "font-family": "JetBrains Mono" });
  labelB.textContent = o.otherName;
  const ghosts = svg("g");
  const marker = svg("g");
  marker.append(
    svg("line", { x1: 0, x2: 0, y1: 8, y2: 32, stroke: "var(--goal)", "stroke-width": 2 }),
    svg("circle", { cx: 0, cy: 20, r: 4, fill: "var(--goal)" }),
  );
  root.append(labelA, labelB, ghosts, marker);
  return {
    node: root,
    update(values) {
      ghosts.replaceChildren(
        ...values.slice(0, -1).map((t) => svg("circle", { cx: x(o.t(t)), cy: 20, r: 2.2, fill: "var(--goal)", opacity: 0.18 })),
      );
      marker.setAttribute("transform", `translate(${x(o.t(values[values.length - 1]))},0)`);
    },
  };
}

function buildOverlay(condition) {
  const root = svg("svg", { viewBox: "0 0 224 224" });
  const defs = svg("defs");
  const glow = svg("filter", { id: `glow-${Math.random().toString(36).slice(2)}`, x: "-50%", y: "-50%", width: "200%", height: "200%" });
  glow.append(svg("feGaussianBlur", { stdDeviation: 2.4, result: "b" }));
  const merge = svg("feMerge");
  merge.append(svg("feMergeNode", { in: "b" }), svg("feMergeNode", { in: "SourceGraphic" }));
  glow.append(merge);
  defs.append(glow);
  const layer = svg("g");
  root.append(defs, layer);
  const o = orient(condition);

  function ring(point, color, label, anchorUp) {
    const group = svg("g", { transform: `translate(${point[0]},${point[1]})` });
    group.append(
      svg("circle", { r: 10, fill: "none", stroke: color, "stroke-width": 1.4, opacity: 0.95 }),
      svg("circle", { r: 1.6, fill: color }),
    );
    const text = svg("text", {
      y: anchorUp ? -15 : 22, "text-anchor": "middle", "font-size": 7.5, "font-family": "JetBrains Mono",
      fill: color, "paint-order": "stroke", stroke: "rgba(0,0,0,0.75)", "stroke-width": 2.4,
    });
    text.textContent = label;
    group.append(text);
    return group;
  }

  return {
    node: root,
    update(rollout, q, grasped) {
      const native = o.native === 0 ? rollout.a_px[q] : rollout.b_px[q];
      const other = o.native === 0 ? rollout.b_px[q] : rollout.a_px[q];
      const upNative = native[1] <= other[1];
      const trail = rollout.goal_px.slice(Math.max(0, q - TRAIL), q + 1);
      const goal = rollout.goal_px[q];
      const children = [
        ring(native, "var(--native)", o.nativeName, upNative),
        ring(other, "var(--other)", o.otherName, !upNative),
      ];
      if (trail.length > 1) {
        children.push(svg("polyline", {
          points: trail.map((p) => p.join(",")).join(" "), fill: "none", stroke: "#fff",
          "stroke-width": 1, opacity: 0.45, "stroke-dasharray": "2 2",
        }));
      }
      const g = svg("g", { transform: `translate(${goal[0]},${goal[1]})`, filter: `url(#${glow.id})` });
      g.append(
        svg("circle", { r: 7, fill: "none", stroke: "#fff", "stroke-width": 1, opacity: 0.55 }),
        svg("line", { x1: -12, x2: -8, y1: 0, y2: 0, stroke: "#fff", "stroke-width": 1 }),
        svg("line", { x1: 8, x2: 12, y1: 0, y2: 0, stroke: "#fff", "stroke-width": 1 }),
        svg("line", { x1: 0, x2: 0, y1: -12, y2: -8, stroke: "#fff", "stroke-width": 1 }),
        svg("line", { x1: 0, x2: 0, y1: 8, y2: 12, stroke: "#fff", "stroke-width": 1 }),
        svg("circle", { r: 2.6, fill: "#fff" }),
      );
      children.push(g);
      if (grasped !== null) {
        const target = grasped === o.native ? native : other;
        children.push(svg("circle", {
          cx: target[0], cy: target[1], r: 15, fill: "none",
          stroke: grasped === rollout.prompt ? "var(--pass)" : "var(--fail)", "stroke-width": 2,
        }));
      }
      layer.replaceChildren(...children);
    },
  };
}

function buildPane(condition, rollout, { still, stillLabel }) {
  const o = orient(condition);
  const head = el("div", { class: "pane-head" },
    el("span", { class: "tag" }, rollout.prompt === o.native ? "native instruction" : "swapped instruction"),
  );
  head.append(sayInstruction(condition, rollout.prompt));
  const screen = el("div", { class: "screen" });
  let media;
  if (still) {
    media = el("img", { src: `data/${rollout.stills[stillLabel]}`, alt: "" });
  } else {
    media = el("video", { src: `data/${rollout.video}`, muted: "", playsinline: "", preload: "auto", poster: `data/${rollout.stills.start}` });
    media.muted = true;
  }
  const overlay = buildOverlay(condition);
  const hud = el("div", { class: "hud" });
  screen.append(media, overlay.node, hud);
  const meter = buildMeter(condition);
  const chip = el("span", { class: "chip" }, "…");
  const foot = el("div", { class: "pane-foot" }, chip);
  const node = el("div", { class: "pane" }, head, screen, meter.node, foot);
  const outcome = outcomeOf(rollout, condition);
  const duration = rollout.steps / FPS;

  function render(frame) {
    const f = Math.max(0, Math.min(rollout.steps - 1, frame));
    const q = queryAt(rollout, f);
    const graspedNow = rollout.first_grasp_step !== null && f + 1 >= rollout.first_grasp_step;
    overlay.update(rollout, q, graspedNow && rollout.first_grasp !== -1 && rollout.first_grasp !== 2 ? rollout.first_grasp : null);
    meter.update(rollout.goal_t.slice(0, q + 1));
    hud.innerHTML = `step ${String(f).padStart(3, "0")} · query ${String(q).padStart(2, "0")}<br>goal probe · expert b13`;
    const done = f >= rollout.steps - 1 || graspedNow;
    chip.className = `chip ${done ? outcome.cls : ""}`;
    chip.textContent = done ? outcome.text : "first grasp …";
  }

  return { node, media, render, duration };
}

export async function createPlayer(root, { index, conditionId, state, still = false, stillLabel = "grasp", onState }) {
  root.replaceChildren();
  const condition = await load(`conditions/${conditionId}.json`);
  const states = condition.video_states;
  let current = state ?? states[0];
  const o = orient(condition);

  const bar = el("div", { class: "player-bar" });
  const screens = el("div", { class: "screens" });
  const transport = el("div", { class: "transport" });
  root.append(...(still ? [screens] : [bar, screens, transport]));

  let panes = [];
  let playing = false;
  let t0 = 0;
  let offset = 0;
  let raf = 0;
  let maxDuration = 1;

  const playIcon = '<svg viewBox="0 0 14 14"><path d="M3 1.5v11l9-5.5z" fill="currentColor"/></svg>';
  const pauseIcon = '<svg viewBox="0 0 14 14"><path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor"/></svg>';
  const button = el("button", { class: "btn", "aria-label": "Play" });
  const scrub = el("input", { type: "range", class: "scrub", min: 0, max: 1000, value: 0, "aria-label": "Time" });
  const clock = el("span", { class: "clock" });
  transport.append(button, scrub, clock);

  const now = () => (playing ? offset + (performance.now() - t0) / 1000 : offset);

  function tick() {
    const time = Math.min(now(), maxDuration);
    for (const pane of panes) {
      const local = Math.min(time, pane.duration - 0.05);
      if (playing && time < pane.duration) {
        if (pane.media.paused) pane.media.play().catch(() => {});
        if (Math.abs(pane.media.currentTime - local) > 0.15) pane.media.currentTime = local;
      } else if (!pane.media.paused) {
        pane.media.pause();
      }
      if (!playing && Math.abs(pane.media.currentTime - local) > 0.03) pane.media.currentTime = local;
      pane.render(Math.floor(time * FPS));
    }
    scrub.value = Math.round((time / maxDuration) * 1000);
    clock.textContent = `${time.toFixed(1)}s / ${maxDuration.toFixed(1)}s`;
    if (playing && time >= maxDuration) {
      offset = maxDuration;
      setPlaying(false);
    }
    if (playing) raf = requestAnimationFrame(tick);
  }

  function setPlaying(value) {
    if (value && now() >= maxDuration - 0.05) offset = 0;
    playing = value;
    if (value) {
      t0 = performance.now();
      raf = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(raf);
      panes.forEach((pane) => pane.media.pause());
    }
    button.innerHTML = value ? pauseIcon : playIcon;
  }

  button.innerHTML = playIcon;
  button.addEventListener("click", () => {
    if (playing) {
      offset = now();
      setPlaying(false);
      tick();
    } else setPlaying(true);
  });
  scrub.addEventListener("input", () => {
    offset = (scrub.value / 1000) * maxDuration;
    t0 = performance.now();
    tick();
  });

  function mount(stateId) {
    current = stateId;
    const pair = condition.rollouts.filter((r) => r.state === stateId);
    const ordered = [pair.find((r) => r.prompt === o.native), pair.find((r) => r.prompt === o.other)];
    panes = ordered.map((rollout) => buildPane(condition, rollout, { still, stillLabel }));
    screens.replaceChildren(...panes.map((pane) => pane.node));
    maxDuration = Math.max(...panes.map((pane) => pane.duration));
    offset = 0;
    if (still) {
      const [left, right] = ordered;
      const frame = (rollout) => {
        const label = stillLabel === "start" ? 0 : stillLabel === "end" ? rollout.steps - 1 : (rollout.first_grasp_step || Math.floor(rollout.steps / 2)) - 1;
        return label;
      };
      panes[0].render(frame(left));
      panes[1].render(frame(right));
      return;
    }
    tick();
    onState?.(condition, stateId);
  }

  if (!still) {
    const picker = el("select", { class: "pick", "aria-label": "Initial state" });
    for (const stateId of states) {
      const swapped = condition.rollouts.find((r) => r.state === stateId && r.prompt === o.other);
      const label = swapped.commanded ? "obeys" : swapped.first_grasp === -1 ? "no grasp" : "disobeys";
      picker.append(el("option", { value: stateId }, `initial state ${String(stateId).padStart(2, "0")} · swapped ${label}`));
    }
    picker.value = current;
    picker.addEventListener("change", () => {
      setPlaying(false);
      mount(Number(picker.value));
    });
    const conditionPicker = el("div", { class: "seg", role: "group", "aria-label": "Condition" });
    for (const item of index) {
      const io = orient(item);
      const buttonNode = el("button", {
        "aria-pressed": String(item.id === conditionId),
        title: `${io.nativeName} scene, swapped instruction names ${io.otherName}`,
        onclick: () => {
          setPlaying(false);
          createPlayer(root, { index, conditionId: item.id, onState });
        },
      }, item.id);
      conditionPicker.append(buttonNode);
    }
    const compliance = el("span", { class: `chip ${condition.swapped_compliance > 0.9 ? "pass" : "fail"}` },
      `swapped instruction obeyed ${Math.round(condition.swapped_compliance * 100)}% of ${new Set(condition.rollouts.map((r) => r.state)).size}`);
    bar.append(conditionPicker, picker, el("span", { class: "spacer" }), compliance);
  }

  mount(current);
  return { condition, get state() { return current; } };
}
