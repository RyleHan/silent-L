// D3 charts shared by the explorer and the static README figures.

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import { el, orient, tooltip, fmt, pct, REPO } from "./core.js";

const MONO = "JetBrains Mono, ui-monospace, monospace";

function frame(root, width, height, margin) {
  root.replaceChildren();
  const svgNode = d3.select(root).append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("class", "chart")
    .style("width", "100%")
    .style("height", "auto")
    .style("display", "block");
  const g = svgNode.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  return { svg: svgNode, g, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
}

function styleAxis(selection) {
  selection.selectAll("path").attr("stroke", "var(--line-2)");
  selection.selectAll("line").attr("stroke", "var(--line-2)");
  selection.selectAll("text").attr("fill", "var(--faint)").style("font-family", MONO).style("font-size", "10px");
}

// ---------------------------------------------------------------- Heard: layer curves
const STREAMS = [
  { key: "openvla_prompt_end", control: "openvla_visual", title: "OpenVLA · Llama residual", sub: "prompt-end token · 32 blocks" },
  { key: "pi05_prefix", control: "pi05_visual", title: "π0.5 · PaliGemma prefix", sub: "prompt-end token · 18 blocks" },
  { key: "pi05_expert", control: "pi05_visual", title: "π0.5 · action expert", sub: "10 action tokens · 18 blocks" },
];

export function layerCurves(root, layers, { height = 300 } = {}) {
  root.replaceChildren();
  const tip = tooltip();
  const grid = el("div", { class: "grid-3" });
  root.append(grid);
  for (const stream of STREAMS) {
    const cell = el("div");
    cell.append(
      el("div", { class: "mono", style: "font-size:12px;color:var(--text);margin-bottom:2px" }, stream.title),
      el("div", { class: "mono", style: "font-size:11px;color:var(--faint);margin-bottom:8px" }, stream.sub),
    );
    const plot = el("div");
    cell.append(plot);
    grid.append(cell);

    const aware = layers[stream.key].target_xyz;
    const absolute = layers[stream.key].object_a_xyz;
    const controlSeries = layers[stream.control].target_xyz;
    const control = aware.map((_, i) => (controlSeries.length === aware.length ? controlSeries[i] : controlSeries[0]));
    const { g, w, h } = frame(plot, 360, height, { top: 10, right: 12, bottom: 30, left: 34 });
    const x = d3.scaleLinear().domain([0, aware.length - 1]).range([0, w]);
    const y = d3.scaleLinear().domain([-0.1, 1]).range([h, 0]).clamp(true);

    g.append("g").selectAll("line").data([0, 0.25, 0.5, 0.75, 1]).join("line")
      .attr("x1", 0).attr("x2", w).attr("y1", (d) => y(d)).attr("y2", (d) => y(d))
      .attr("stroke", "var(--line)");
    g.append("g").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(aware.length > 20 ? 8 : 6).tickSize(3)).call(styleAxis);
    g.append("g").call(d3.axisLeft(y).ticks(5).tickSize(3)).call(styleAxis);

    const area = d3.area().x((_, i) => x(i)).y0((_, i) => y(control[i])).y1((d) => y(d)).curve(d3.curveMonotoneX);
    g.append("path").datum(aware).attr("d", area).attr("fill", "var(--other-soft)");
    const line = d3.line().x((_, i) => x(i)).y((d) => y(d)).curve(d3.curveMonotoneX);
    g.append("path").datum(absolute).attr("d", line).attr("fill", "none").attr("stroke", "var(--faint)").attr("stroke-width", 1.2);
    g.append("path").datum(control).attr("d", line).attr("fill", "none").attr("stroke", "var(--native)").attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
    g.append("path").datum(aware).attr("d", line).attr("fill", "none").attr("stroke", "var(--other)").attr("stroke-width", 2.2);

    const peak = d3.maxIndex(aware);
    g.append("circle").attr("cx", x(peak)).attr("cy", y(aware[peak])).attr("r", 3.5).attr("fill", "var(--other)");
    g.append("text").attr("x", x(peak)).attr("y", y(aware[peak]) - 9).attr("text-anchor", peak > aware.length * 0.75 ? "end" : "middle")
      .style("font-family", MONO).style("font-size", "10px").attr("fill", "var(--other)")
      .text(`+${(aware[peak] - control[peak]).toFixed(2)}`);

    const cursor = g.append("line").attr("y1", 0).attr("y2", h).attr("stroke", "var(--line-2)").style("opacity", 0);
    g.append("rect").attr("width", w).attr("height", h).attr("fill", "transparent")
      .on("mousemove", (event) => {
        const i = Math.max(0, Math.min(aware.length - 1, Math.round(x.invert(d3.pointer(event)[0]))));
        cursor.attr("x1", x(i)).attr("x2", x(i)).style("opacity", 1);
        tip.show(`block ${i}<br><span style="color:var(--other)">target XYZ, with language ${aware[i].toFixed(3)}</span><br>`
          + `<span style="color:var(--native)">target XYZ, prompt-blind ${control[i].toFixed(3)}</span><br>`
          + `<span style="color:var(--muted)">absolute object XYZ ${absolute[i].toFixed(3)}</span>`, event);
      })
      .on("mouseleave", () => { cursor.style("opacity", 0); tip.hide(); });
  }
  root.append(el("div", { class: "legend" },
    el("span", {}, el("i", { style: "background:var(--other)" }), "target XYZ R², language integrated"),
    el("span", {}, el("i", { style: "background:var(--native)" }), "target XYZ R², prompt-blind control"),
    el("span", {}, el("i", { style: "background:var(--faint)" }), "absolute object XYZ R²"),
  ));
}

// ---------------------------------------------------------------- Obeyed: goal shift by condition
export function conditionRows(root, index, stage13, { selected, onSelect, height = 430, width = 760 } = {}) {
  const tip = tooltip();
  const rows = [...index].sort((a, b) => a.swapped_compliance - b.swapped_compliance
    || a.laso.mean_shift - b.laso.mean_shift);
  const { g, w, h } = frame(root, width, height, { top: 34, right: 18, bottom: 64, left: 210 });
  const barW = 130;
  const plotW = w - barW - 36;
  const x = d3.scaleLinear().domain([-0.2, 1]).range([0, plotW]);
  const y = d3.scaleBand().domain(rows.map((c) => c.id)).range([0, h]).padding(0.34);
  const statusColor = (c) => (c.swapped_compliance > 0.9 ? "var(--pass)" : c.swapped_compliance > 0.3 ? "var(--other)" : "var(--fail)");

  for (const [value, label] of [[0, "scene's own object"], [1, "named object"]]) {
    g.append("line").attr("x1", x(value)).attr("x2", x(value)).attr("y1", -8).attr("y2", h)
      .attr("stroke", value === 0 ? "var(--native)" : "var(--other)").attr("opacity", 0.35);
    g.append("text").attr("x", x(value)).attr("y", -16).attr("text-anchor", value === 0 ? "middle" : "end").style("font-family", MONO)
      .style("font-size", "10px").attr("fill", value === 0 ? "var(--native)" : "var(--other)").text(label);
  }
  g.append("line").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", 0).attr("y2", h).attr("stroke", "var(--line-2)").attr("stroke-dasharray", "2 3");
  g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(6).tickSize(3)).call(styleAxis);
  g.append("text").attr("x", plotW / 2).attr("y", h + 34).attr("text-anchor", "middle").style("font-family", MONO)
    .style("font-size", "10.5px").attr("fill", "var(--muted)").text("decoded goal at the first query: native instruction → swapped instruction");
  g.append("text").attr("x", plotW + 36).attr("y", -16).style("font-family", MONO).style("font-size", "10px")
    .attr("fill", "var(--muted)").text("robot obeys swap");

  const row = g.append("g").selectAll("g").data(rows).join("g")
    .attr("transform", (c) => `translate(0,${y(c.id) + y.bandwidth() / 2})`)
    .style("cursor", onSelect ? "pointer" : "default")
    .on("click", (_, c) => onSelect?.(c.id))
    .on("mousemove", (event, c) => {
      const o = orient(c);
      tip.show(`<b>${c.id}</b> · ${o.nativeName} scene<br>swapped: “pick up the ${o.otherName}”<br>`
        + `goal ${c.laso.mean_native_prompt_position.toFixed(2)} → ${c.laso.mean_swapped_prompt_position.toFixed(2)} `
        + `(shift ${c.laso.mean_shift.toFixed(2)})<br>obeyed ${pct(c.swapped_compliance)} of 50`, event);
    })
    .on("mouseleave", () => tip.hide());
  row.append("rect").attr("x", -204).attr("y", -y.bandwidth() / 2 - 6).attr("width", w + 204).attr("height", y.bandwidth() + 12)
    .attr("rx", 8).attr("fill", (c) => (c.id === selected ? "var(--surface-2)" : "transparent"));
  row.append("text").attr("x", -196).attr("y", -3).style("font-family", MONO).style("font-size", "11px").attr("fill", "var(--text)")
    .text((c) => `${c.id} · ${orient(c).nativeName} scene`);
  row.append("text").attr("x", -196).attr("y", 12).style("font-family", MONO).style("font-size", "10px").attr("fill", "var(--faint)")
    .text((c) => `“pick up the ${orient(c).otherName}”`);
  row.append("line").attr("x1", (c) => x(c.laso.mean_native_prompt_position)).attr("x2", (c) => x(c.laso.mean_swapped_prompt_position) - 7)
    .attr("stroke", "var(--line-2)").attr("stroke-width", 2);
  row.append("path").attr("d", "M-7,-4 L0,0 L-7,4").attr("fill", "none").attr("stroke", "var(--muted)").attr("stroke-width", 1.4)
    .attr("transform", (c) => `translate(${x(c.laso.mean_swapped_prompt_position) - 7},0)`);
  row.append("circle").attr("cx", (c) => x(c.laso.mean_native_prompt_position)).attr("r", 5).attr("fill", "var(--native)");
  row.append("circle").attr("cx", (c) => x(c.laso.mean_swapped_prompt_position)).attr("r", 6.5).attr("fill", statusColor)
    .attr("stroke", "var(--bg)").attr("stroke-width", 2);
  const bars = row.append("g").attr("transform", `translate(${plotW + 36},0)`);
  bars.append("rect").attr("y", -5).attr("width", barW - 44).attr("height", 10).attr("rx", 5).attr("fill", "var(--line)");
  bars.append("rect").attr("y", -5).attr("width", (c) => Math.max(3, (barW - 44) * c.swapped_compliance)).attr("height", 10).attr("rx", 5).attr("fill", statusColor);
  bars.append("text").attr("x", barW - 38).attr("y", 4).style("font-family", MONO).style("font-size", "11px").attr("fill", "var(--text)")
    .text((c) => pct(c.swapped_compliance));

  const primary = stage13.primary;
  if (primary.spearman_rho !== undefined) {
    g.append("text").attr("x", plotW / 2).attr("y", h + 52).attr("text-anchor", "middle").style("font-family", MONO)
      .style("font-size", "10px").attr("fill", "var(--faint)")
      .text(`Spearman rho(shift, obedience) = ${fmt(primary.spearman_rho)} across ${primary.included_conditions.length} conditions: the predicted positive relation is absent`);
  }
}

// ---------------------------------------------------------------- per-state dumbbells
export function dumbbells(root, condition, { height = 420, width = 520, highlight } = {}) {
  const tip = tooltip();
  const o = orient(condition);
  const byState = d3.group(condition.rollouts, (r) => r.state);
  const rows = [...byState].map(([state, pair]) => {
    const native = pair.find((r) => r.prompt === o.native);
    const swapped = pair.find((r) => r.prompt === o.other);
    return { state, a: o.t(native.goal_t[0]), b: o.t(swapped.goal_t[0]), swapped };
  }).sort((p, q) => (q.b - q.a) - (p.b - p.a));
  const { g, w, h } = frame(root, width, height, { top: 30, right: 16, bottom: 48, left: 16 });
  const x = d3.scaleLinear().domain([-0.4, 1.3]).range([0, w]).clamp(true);
  const y = d3.scaleBand().domain(rows.map((r) => r.state)).range([0, h]).padding(0.3);
  for (const [value, color, label] of [[0, "var(--native)", o.nativeName], [1, "var(--other)", o.otherName]]) {
    g.append("line").attr("x1", x(value)).attr("x2", x(value)).attr("y1", -6).attr("y2", h).attr("stroke", color).attr("opacity", 0.35);
    g.append("text").attr("x", x(value)).attr("y", -12).attr("text-anchor", "middle").style("font-family", MONO).style("font-size", "10px").attr("fill", color).text(label);
  }
  g.append("line").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", 0).attr("y2", h).attr("stroke", "var(--line-2)").attr("stroke-dasharray", "2 3");
  const color = (r) => (r.swapped.commanded ? "var(--pass)" : r.swapped.first_grasp === -1 ? "var(--faint)" : "var(--fail)");
  const row = g.append("g").selectAll("g").data(rows).join("g").attr("transform", (r) => `translate(0,${y(r.state) + y.bandwidth() / 2})`)
    .on("mousemove", (event, r) => tip.show(`initial state ${r.state}<br>native prompt ${r.a.toFixed(2)} → swapped ${r.b.toFixed(2)}<br>${r.swapped.commanded ? "obeyed" : r.swapped.first_grasp === -1 ? "no grasp" : "grasped the scene's object"}`, event))
    .on("mouseleave", () => tip.hide());
  row.append("line").attr("x1", (r) => x(r.a)).attr("x2", (r) => x(r.b)).attr("stroke", color).attr("stroke-width", 1.4).attr("opacity", 0.8);
  row.append("circle").attr("cx", (r) => x(r.a)).attr("r", 2.4).attr("fill", "var(--native)");
  row.append("circle").attr("cx", (r) => x(r.b)).attr("r", (r) => (r.state === highlight ? 4.5 : 3)).attr("fill", color);
  const legend = g.append("g").attr("transform", `translate(0,${h + 24})`);
  [["var(--pass)", "obeyed"], ["var(--fail)", "grasped the scene's object"], ["var(--faint)", "no grasp"]].forEach(([fill, label], i) => {
    const item = legend.append("g").attr("transform", `translate(${[0, 90, 300][i]},0)`);
    item.append("circle").attr("r", 3.5).attr("cy", -3).attr("fill", fill);
    item.append("text").attr("x", 8).style("font-family", MONO).style("font-size", "10px").attr("fill", "var(--muted)").text(label);
  });
  legend.append("text").attr("y", 16).style("font-family", MONO).style("font-size", "10px")
    .attr("fill", "var(--faint)").text("blue: native instruction · coloured: swapped instruction");
}

// ---------------------------------------------------------------- layer tomography
export function tomography(root, condition, state, { width = 640, height = 190 } = {}) {
  const tip = tooltip();
  const o = orient(condition);
  const pair = condition.rollouts.filter((r) => r.state === state);
  const native = pair.find((r) => r.prompt === o.native);
  const swapped = pair.find((r) => r.prompt === o.other);
  const rows = [
    { label: "prefix · native", values: native.tomography.prefix },
    { label: "prefix · swapped", values: swapped.tomography.prefix },
    { label: "expert · native", values: native.tomography.expert },
    { label: "expert · swapped", values: swapped.tomography.expert },
  ];
  const { g, w, h } = frame(root, width, height, { top: 8, right: 8, bottom: 26, left: 118 });
  const x = d3.scaleBand().domain(d3.range(18)).range([0, w]).padding(0.08);
  const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, h]).padding(0.14);
  const scale = (value) => {
    const t = Math.max(0, Math.min(1, o.t(value)));
    return t < 0.5
      ? d3.interpolateRgb(getComputedStyle(document.documentElement).getPropertyValue("--native").trim(), "#3a3f48")(t * 2)
      : d3.interpolateRgb("#3a3f48", getComputedStyle(document.documentElement).getPropertyValue("--other").trim())((t - 0.5) * 2);
  };
  for (const row of rows) {
    g.append("text").attr("x", -10).attr("y", y(row.label) + y.bandwidth() / 2 + 4).attr("text-anchor", "end")
      .style("font-family", MONO).style("font-size", "10px").attr("fill", "var(--muted)").text(row.label);
    g.append("g").selectAll("rect").data(row.values).join("rect")
      .attr("x", (_, i) => x(i)).attr("y", y(row.label)).attr("width", x.bandwidth()).attr("height", y.bandwidth()).attr("rx", 3)
      .attr("fill", (v) => scale(v))
      .on("mousemove", (event, v) => tip.show(`${row.label} · block ${row.values.indexOf(v)}<br>decoded goal ${o.t(v).toFixed(2)} (0 = ${o.nativeName}, 1 = ${o.otherName})`, event))
      .on("mouseleave", () => tip.hide());
  }
  g.append("rect").attr("x", x(13) - 2).attr("y", y("expert · native") - 2).attr("width", x.bandwidth() + 4)
    .attr("height", y("expert · swapped") + y.bandwidth() - y("expert · native") + 4).attr("fill", "none").attr("stroke", "var(--text)").attr("rx", 4);
  g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).tickValues([0, 3, 6, 9, 13, 17]).tickSize(0)).call(styleAxis)
    .call((s) => s.select("path").remove());
}

// ---------------------------------------------------------------- Writable: intervention ladder
export const LADDER = [
  { n: "rung 1", title: "Decode", big: "+0.37", cls: "", text: "Target-XYZ R² over the prompt-blind control inside π0.5's action expert (OpenVLA +0.54).", audits: ["locked test", "3 seeds", "4 pairs"] },
  { n: "rung 2", title: "Patch one block", big: "22%", cls: "warn", text: "of the prompt-induced action change recovered offline. Through the probe's own subspace: under 1%.", audits: ["shuffled 8%", "random < 1%"] },
  { n: "rung 3", title: "Close the loop", big: "80→20%", cls: "fail", text: "Writing that residual back in closed loop breaks behaviour the swapped prompt still had.", audits: ["20 held-out states", "sampler Δ = 0"] },
  { n: "rung 4", title: "Replay the path", big: "40/40", cls: "pass", text: "Replaying all 18 expert blocks at every denoising step reproduces the correct-prompt policy exactly.", audits: ["max |Δa| = 0.0", "18 × 10 hooks"] },
];

export function ladder(root) {
  root.replaceChildren();
  const box = el("div", { class: "ladder" });
  for (const rung of LADDER) {
    box.append(el("div", { class: "rung" },
      el("div", { class: "n" }, rung.n.toUpperCase()),
      el("h3", {}, rung.title),
      el("div", { class: `big ${rung.cls}` }, rung.big),
      el("p", {}, rung.text),
      el("div", { class: "audits" }, ...rung.audits.map((a) => el("span", { class: "chip" }, a))),
    ));
  }
  root.append(box);
}

// ---------------------------------------------------------------- the ledger
export function ledger(root, rows, stage13Result) {
  root.replaceChildren();
  const box = el("div", { class: "ledger" });
  for (const row of rows) {
    const result = row.stage === "13" && stage13Result ? stage13Result : row.result;
    box.append(el("div", { class: "row" },
      el("span", { class: "s" }, `S${row.stage.padStart(2, "0")}`),
      el("a", { class: "t", href: `${REPO}/blob/main/${row.doc}`, target: "_blank", rel: "noopener" }, row.title),
      el("span", { class: "q" }, row.question),
      el("span", { class: "r" }, result),
      el("span", { class: `chip decision ${row.decision}` }, row.decision),
    ));
  }
  root.append(box);
}
