// Render the README figures from site/figures.html with the local Chrome, light and dark.
// Usage: python3 scripts/serve_site.py &  then  npm run figures [-- fig1,fig2]

import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import puppeteer from "puppeteer-core";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "assets", "figures");
const BASE = process.env.SITE_URL || "http://127.0.0.1:8765";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FIGURES = (process.argv[2] || "hero,heard,obeyed,tomography,writable").split(",");

await mkdir(OUT, { recursive: true });
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ["--hide-scrollbars"] });
try {
  for (const fig of FIGURES) {
    for (const theme of ["dark", "light"]) {
      const page = await browser.newPage();
      await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 });
      await page.goto(`${BASE}/figures.html?fig=${fig}&theme=${theme}`, { waitUntil: "networkidle0" });
      await page.waitForSelector("body[data-ready='1']", { timeout: 60000 });
      const element = await page.$("#fig");
      const file = path.join(OUT, `${fig}_${theme}.png`);
      await element.screenshot({ path: file });
      console.log(`wrote ${path.relative(ROOT, file)}`);
      await page.close();
    }
  }
} finally {
  await browser.close();
}
