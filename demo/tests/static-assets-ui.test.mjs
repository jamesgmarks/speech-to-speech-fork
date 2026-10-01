import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import path from "node:path";
import { chromium } from "@playwright/test";

test("upgrading a warm browser refreshes cached imports and styles without a hard refresh", async t => {
  const socket = createServer();
  await new Promise(resolve => socket.listen(0, "127.0.0.1", resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  const root = path.resolve(import.meta.dirname, "../..");
  const child = spawn(process.env.S2S_PYTHON || path.join(root, ".venv/bin/python"),
    [path.join(import.meta.dirname, "static-assets-server.py"), String(port)], { stdio: ["ignore", "ignore", "pipe"] });
  let log = "";
  child.stderr.on("data", data => { log += data; });
  t.after(() => child.kill("SIGTERM"));
  const url = `http://127.0.0.1:${port}`;
  let ready = false;
  for (let i = 0; i < 100; i++) {
    try { ready = (await fetch(url)).ok; } catch {}
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  assert.ok(ready, log);
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.goto(url);
  await page.waitForFunction(() => document.body.dataset.value === "old");
  await page.reload(); // prime and reuse the legacy cache
  await page.waitForFunction(() => document.body.dataset.value === "old");
  assert.equal(await page.evaluate(() => getComputedStyle(document.body).color), "rgb(255, 0, 0)");
  assert.equal((await fetch(url + "/deploy", { method: "POST" })).status, 200);
  await page.reload(); // ordinary refresh must replace the whole module graph
  await page.waitForFunction(() => document.body.dataset.value === "new");
  assert.equal(await page.evaluate(() => getComputedStyle(document.body).color), "rgb(0, 0, 255)");
});
