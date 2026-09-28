// Shared helpers: data loading, theme tokens, small DOM utilities.

export const REPO = "https://github.com/RyleHan/silent-L";

const cache = new Map();
export function load(path) {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`data/${path}`).then((response) => {
        if (!response.ok) throw new Error(`Failed to load ${path}`);
        return response.json();
      }),
    );
  }
  return cache.get(path);
}

export const token = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

export const SVG_NS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

export const pct = (value) => `${Math.round(value * 100)}%`;
export const fmt = (value, digits = 2) => (value >= 0 ? "" : "−") + Math.abs(value).toFixed(digits);

// Condition geometry: the scene's native object is always drawn blue, the other object amber.
export function orient(condition) {
  const native = condition.scene_side;
  const other = 1 - native;
  return {
    native,
    other,
    nativeName: condition.object_names[native],
    otherName: condition.object_names[other],
    // goal_t is stored with object A = 0, object B = 1; display runs native (0) -> other (1).
    t: (value) => (native === 0 ? value : 1 - value),
  };
}

export function conditionLabel(condition) {
  const o = orient(condition);
  return `${o.nativeName} scene · “pick up the ${o.otherName}”`;
}

export function outcomeOf(rollout, condition) {
  const o = orient(condition);
  const target = rollout.prompt;
  if (rollout.first_grasp === -1) return { cls: "none", text: "no grasp" };
  if (rollout.first_grasp === 2) return { cls: "fail", text: "grasped both" };
  const name = rollout.first_grasp === o.native ? o.nativeName : o.otherName;
  return rollout.first_grasp === target
    ? { cls: "pass", text: `grasped ${name} · obeyed` }
    : { cls: "fail", text: `grasped ${name} · disobeyed` };
}

export function tooltip() {
  let node = document.querySelector(".tooltip");
  if (!node) {
    node = el("div", { class: "tooltip" });
    document.body.append(node);
  }
  return {
    show(html, event) {
      node.innerHTML = html;
      node.style.opacity = 1;
      const x = Math.min(event.clientX + 14, window.innerWidth - node.offsetWidth - 12);
      node.style.left = `${x}px`;
      node.style.top = `${event.clientY + 14}px`;
    },
    hide() {
      node.style.opacity = 0;
    },
  };
}
