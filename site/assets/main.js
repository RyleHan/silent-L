import { load, el, orient } from "./core.js";
import { createPlayer } from "./player.js";
import { layerCurves, conditionRows, dumbbells, tomography, ladder, ledger } from "./charts.js";

const CLAIMS = [
  { k: "heard", v: "+0.37 R²", d: "The instruction creates a goal state that a prompt-blind readout cannot see, and it reaches π0.5's action expert." },
  { k: "obeyed", v: "6 → 100%", d: "The same policy obeys a swapped instruction almost never or always, depending on the scene." },
  { k: "heard ≠ obeyed", v: "40 / 40", d: "Wrong-object rollouts in which the decoded goal had still moved toward the object the robot was told to pick." },
  { k: "writable", v: "0.0 error", d: "Control works only through the model's own pathway; a one-block patch breaks the rollout." },
];

async function main() {
  const [index, stage13, layers, rows] = await Promise.all([
    load("index.json"), load("stage13.json"), load("layers.json"), load("ledger.json"),
  ]);

  document.getElementById("claims").append(...CLAIMS.map((c) => el("div", { class: "claim" },
    el("div", { class: "k" }, c.k), el("div", { class: "v" }, c.v), el("div", { class: "d" }, c.d))));

  const playerRoot = document.getElementById("player");
  let selected = "0A";

  async function selectCondition(id, state) {
    selected = id;
    await createPlayer(playerRoot, { index, conditionId: id, state, onState: onState });
  }

  async function onState(condition, state) {
    tomography(document.getElementById("tomography"), condition, state);
    const o = orient(condition);
    document.getElementById("dumbbell-title").textContent =
      `${condition.id} · ${o.nativeName} scene, “pick up the ${o.otherName}” · 50 initial states`;
    dumbbells(document.getElementById("dumbbells"), condition, { highlight: state });
    conditionRows(document.getElementById("scatter"), index, stage13, {
      selected: condition.id,
      onSelect: (id) => selectCondition(id).then(() => playerRoot.scrollIntoView({ behavior: "smooth", block: "center" })),
    });
  }

  const first = await load(`conditions/${selected}.json`);
  const o = orient(first);
  const disobeying = first.video_states.find((s) => first.rollouts.some((r) => r.state === s && r.prompt === o.other && !r.commanded && r.first_grasp !== -1));
  await selectCondition(selected, disobeying);

  layerCurves(document.getElementById("layers"), layers);
  const bib = document.getElementById("bib");
  const bibtex = bib.textContent.trim();
  const copy = el("button", { class: "copy", type: "button" }, "copy");
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(bibtex);
    copy.textContent = "copied";
    setTimeout(() => { copy.textContent = "copy"; }, 1500);
  });
  bib.append(copy);
  ladder(document.getElementById("ladder"));
  ledger(document.getElementById("ledger-rows"), rows, window.STAGE13_RESULT);
}

main().catch((error) => {
  console.error(error);
  document.getElementById("player").append(el("p", { class: "caption" }, `Could not load data: ${error.message}`));
});
