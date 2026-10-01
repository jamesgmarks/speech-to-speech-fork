import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
const catalog = { voices: [
  { id: "custom:james", name: "James", kind: "custom" },
  { id: "custom:pepper", name: "Pepper", kind: "custom" },
  { id: "custom:pepper-upbeat", name: "Pepper (upbeat)", kind: "custom" },
], default: "custom:james" };

test("settings discover custom voices, migrate an unsupported preset, and persist the selection", async (t) => {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    if (name.startsWith("/api/")) {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify(name === "/api/voices" ? catalog : name === "/api/config"
        ? { s2sUrl: "ws://localhost:8765/v1/realtime", clientTools: false } : { enabled: false }));
      return;
    }
    const file = path.resolve(root, `.${name === "/" ? "/index.html" : name}`);
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      let content = await readFile(file, "utf8");
      if (file.endsWith(".html")) content = content.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "");
      res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html");
      res.end(content);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  for (const width of [1280, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 850 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("https://**/*", route => route.abort());
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(() => localStorage.setItem("s2s.ws.voice", "Aiden"));
    await page.evaluate(() => import("/main.js"));
    await page.waitForFunction(() => document.querySelectorAll("#voice option").length === 3);
    await page.locator("#settings-btn").click();
    assert.equal(await page.locator("#voice").inputValue(), "custom:james");
    assert.deepEqual(await page.locator("#voice option").allTextContents(), ["James", "Pepper", "Pepper (upbeat)"]);
    await page.locator("#voice").selectOption("custom:pepper-upbeat");
    await page.locator("#settings-save").click();
    assert.equal(await page.evaluate(() => localStorage.getItem("s2s.ws.voice")), "custom:pepper-upbeat");
    await page.reload();
    await page.evaluate(() => import("/main.js"));
    await page.waitForFunction(() => document.querySelectorAll("#voice option").length === 3);
    await page.locator("#settings-btn").click();
    assert.equal(await page.locator("#voice").inputValue(), "custom:pepper-upbeat");
    assert.deepEqual(errors, []);
    await page.close();
  }
  const page = await browser.newPage();
  await page.route("https://**/*", route => route.abort());
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const initialize = async () => page.evaluate(async () => {
    const { ChatView } = await import("/ui/chat.js");
    window.chat = new ChatView();
  });
  await initialize();
  const key = await page.evaluate(() => {
    chat.onConversationRestored({ key: chat.conversationKey, backend: "claude-agent-sdk", reset: false });
    chat.onTranscript({ role: "user", text: "Remember Amir prefers tea.", itemId: "msg_1", partial: false });
    chat.onTranscript({ role: "assistant", text: "I'll remember.", responseId: "resp_1", partial: false });
    chat.reset({ dismiss: true }); // disconnect / reconnect clears only stream bookkeeping
    return chat.conversationKey;
  });
  await page.waitForFunction(() => JSON.parse(localStorage.getItem("s2s.conversation")).messages.length === 2);
  await page.reload();
  await initialize();
  assert.equal(await page.evaluate(() => chat.conversationKey), key);
  assert.deepEqual(await page.locator(".hist-body").allTextContents(), ["Remember Amir prefers tea.", "I'll remember."]);
  await page.evaluate(() => chat.onConversationRestored({ key: chat.conversationKey, backend: "claude-agent-sdk", reset: false }));
  assert.equal(await page.locator(".hist-msg").count(), 2);
  await page.evaluate(() => chat.onConversationRestored({ key: chat.conversationKey, backend: "responses-api", reset: true }));
  assert.equal(await page.locator(".hist-msg").count(), 0);
  await page.reload();
  await initialize();
  assert.equal(await page.evaluate(() => chat.conversationBackend), "responses-api");
  assert.equal(await page.locator(".hist-msg").count(), 0);
  await page.evaluate(() => chat.onConversationRestored({
    key: chat.conversationKey, backend: "responses-api", reset: false,
    history: [{ role: "user", text: "Recovered from durable context." }],
  }));
  assert.equal(await page.locator(".hist-body").textContent(), "Recovered from durable context.");
  await page.close();
});
