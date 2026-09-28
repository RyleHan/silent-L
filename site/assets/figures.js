// Fixed-size renderings of the explorer components for the README (see scripts/render_figures.sh).

import { load, el, orient } from "./core.js";
import { createPlayer } from "./player.js";
import { layerCurves, conditionRows, dumbbells, tomography, ladder } from "./charts.js";

const params = new URLSearchParams(location.search);
document.documentElement.dataset.theme = params.get("theme") || "dark";
const root = document.getElementById("fig");

function header(title, sub) {
  return [el("h2", { class: "fig-title", html: title }), el("p", { class: "fig-sub", html: sub })];
}

function footer(left, right = "github.com/RyleHan/silent-L") {
  return el("div", { class: "fig-foot" }, el("span", {}, left), el("span", {}, right));
}

async function featured() {
  const condition = await load(`conditions/${params.get("condition") || "0A"}.json`);
  const o = orient(condition);
  const stateParam = params.get("state");
  const state = stateParam !== null ? Number(stateParam) : condition.video_states.find((s) =>
    condition.rollouts.some((r) => r.state === s && r.prompt === o.other && !r.commanded && r.first_grasp !== -1));
  return { condition, state };
}

const FIGURES = {
  async hero() {
    const index = await load("index.json");
    const stage13 = await load("stage13.json");
    const { condition, state } = await featured();
    const left = el("div", {},
      el("div", { class: "eyebrow" }, el("span", { class: "dot" }), "VLA interpretability · OpenVLA & ", el("span", { class: "nc" }, "π"), "0.5"),
      el("h1", { html: "The Silent <em>L</em>?" }),
      el("p", { class: "lede", html: window.HERO_LEDE || "Same scene, two instructions. π0.5 hears the swapped instruction, and its internal goal moves toward the named object. Then it grabs the scene's usual object anyway." }),
      el("div", { class: "stat-row" },
        ...(window.HERO_STATS || [["heard", "+0.37 R²"], ["heard ≠ obeyed", "40 / 40"], ["writable", "0.0 error"]])
          .map(([k, v]) => el("div", {}, el("div", { class: "k" }, k), el("div", { class: "v" }, v)))),
    );
    const right = el("div", { class: "panel player" });
    root.append(el("div", { class: "hero-grid" }, left, right));
    await createPlayer(right, { index, conditionId: condition.id, state, still: true, stillLabel: params.get("still") || "grasp" });
    void stage13;
  },

  async heard() {
    const layers = await load("layers.json");
    root.append(...header("Language creates a goal the image <em>cannot</em>.",
      "Held-out R² of linear probes after every block. The target's position becomes readable only once the instruction is integrated; a prompt-blind control that sees the same image cannot recover it. Absolute object geometry needs no language."));
    const body = el("div", { class: "panel panel-pad" });
    root.append(body);
    layerCurves(body, layers, { height: 300 });
    root.append(footer("400 rollouts · 15,275 states · 4 object pairs · 3 probe seeds · episode-held-out test"));
  },

  async obeyed() {
    const [index, stage13] = await Promise.all([load("index.json"), load("stage13.json")]);
    const { condition, state } = await featured();
    const o = orient(condition);
    root.append(...header(window.OBEYED_TITLE || "What it hears is <em>not</em> what it does.",
      window.OBEYED_SUB || "Left: six scene × instruction conditions. How far the swapped instruction moves π0.5's decoded goal before the arm moves does not predict how often the robot obeys. Right: in the condition that almost never obeys, the goal still moves toward the named object in all 50 states."));
    const scatter = el("div");
    const bells = el("div");
    const grid = el("div", { class: "grid-2", style: "grid-template-columns:1.25fr 1fr" },
      el("div", { class: "panel panel-pad" }, scatter),
      el("div", { class: "panel panel-pad" },
        el("div", { class: "mono", style: "font-size:12px;color:var(--muted)" }, `${condition.id} · ${o.nativeName} scene, “pick up the ${o.otherName}”`),
        bells));
    root.append(grid);
    conditionRows(scatter, index, stage13, { selected: condition.id, width: 820, height: 470 });
    dumbbells(bells, condition, { width: 560, height: 470, highlight: state });
    root.append(footer("action expert block 13 · leave-scene-out probe · first policy query · exploratory (Stage 13)"));
  },

  async tomography() {
    const { condition, state } = await featured();
    const o = orient(condition);
    root.append(...header("The instruction arrives <em>layer by layer</em>.",
      `Decoded goal after every block of π0.5 for one initial state, before any motion. Blue: the ${o.nativeName}; amber: the ${o.otherName}. Under “pick up the ${o.otherName}” the goal leaves the scene's object but never fully arrives.`));
    const body = el("div", { class: "panel panel-pad" });
    root.append(body);
    tomography(body, condition, state, { width: 1440, height: 250 });
    root.append(footer(`condition ${condition.id} · initial state ${state} · outlined: expert block 13, the preregistered readout`));
  },

  async writable() {
    root.append(...header("Readable is not <em>writable</em>.",
      "Four rungs of intervention on π0.5 with same-state counterfactual prompts. Only an operator that preserves the model's own computation path controls it."));
    const body = el("div");
    root.append(body);
    ladder(body);
    root.append(footer("LIBERO-Object · 20 held-out states per target · official sampler audits exact"));
  },
};

(async () => {
  const name = params.get("fig") || "hero";
  await FIGURES[name]();
  await document.fonts.ready;
  await Promise.all([...document.images].map((img) => (img.complete ? null : new Promise((r) => { img.onload = img.onerror = r; }))));
  document.body.dataset.ready = "1";
})();
