import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

test("new browsers and different origins share the transcript; only the fresh action rotates its key", async t => {
  const root = path.resolve(import.meta.dirname, "..");
  let resets = 0;
  let conversation = {
    key: "88edaf87-91f2-4fc3-a153-7b5355fdd7d5", backend: "claude-agent-sdk", reset: false, resumed: true,
    history: [{ role: "user", text: "Amir prefers tea." }, { role: "assistant", text: "I will remember that." }],
  };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname.startsWith("/api/")) {
      let data = { enabled: false };
      if (url.pathname === "/api/config") data = { allowDirect: true, s2sUrl: "ws://127.0.0.1:1/v1/realtime", sharedConversation: true, clientTools: false, rtc: false, startupGreeting: "" };
      if (url.pathname === "/api/voices") data = { voices: [{ id: "custom:james", name: "James" }], default: "custom:james" };
      if (url.pathname === "/api/conversation/new") {
        assert.equal(req.method, "POST");
        assert.equal(url.searchParams.get("expected_key"), conversation.key);
        resets++;
        conversation = { ...conversation, key: "a353e5f4-53dd-403c-bb2c-30469a6e0797", history: [], resumed: false };
      }
      if (url.pathname.startsWith("/api/conversation")) data = conversation;
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify(data));
      return;
    }
    const file = path.resolve(root, `.${url.pathname === "/" ? "/index.html" : url.pathname}`);
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
  const pages = [];
  for (const host of ["127.0.0.1", "localhost"]) {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.route("https://**/*", route => route.abort());
    await page.goto(`http://${host}:${server.address().port}/`);
    await page.evaluate(() => {
      localStorage.setItem("s2s.conversation", JSON.stringify({ key: crypto.randomUUID(), backend: "claude-agent-sdk", messages: [{ role: "user", text: "A stale browser conversation." }] }));
      navigator.mediaDevices.getUserMedia = async () => { throw new Error("Test intentionally stops before microphone startup"); };
    });
    await page.evaluate(() => import("/main.js"));
    await page.waitForFunction(() => document.querySelector("#chat-history").textContent.includes("Amir prefers tea."));
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem("s2s.conversation")).key), conversation.key);
    assert.equal(await page.locator("#chat-history").getByText("A stale browser conversation.").count(), 0);
    pages.push(page);
  }
  // A new window/context with no browser storage also resumes the server key.
  await pages[1].context().close();
  const context = await browser.newContext();
  const reopened = await context.newPage();
  await reopened.route("https://**/*", route => route.abort());
  await reopened.goto(`http://localhost:${server.address().port}/`);
  await reopened.evaluate(() => import("/main.js"));
  await reopened.waitForFunction(() => document.querySelector("#chat-history").textContent.includes("I will remember that."));
  assert.equal(await reopened.evaluate(() => JSON.parse(localStorage.getItem("s2s.conversation")).key), conversation.key);
  const page = pages[0];
  await page.locator("#settings-btn").click();
  await page.locator("#settings-save").click();
  assert.equal(resets, 0);
  await page.locator("#settings-btn").click();
  await page.locator("#restart-conversation").click();
  await page.waitForFunction(() => document.querySelector("#circle-caption").textContent.includes("Test intentionally"));
  assert.equal(resets, 0);
  assert.match(await page.locator("#chat-history").textContent(), /Amir prefers tea/);
  await page.locator("#settings-btn").click();
  await page.locator("#new-conversation").click();
  await page.waitForFunction(() => document.querySelectorAll("#chat-history .hist-msg").length === 0);
  assert.equal(resets, 1);
  await reopened.waitForFunction(() => document.querySelectorAll("#chat-history .hist-msg").length === 0);
  assert.equal(await reopened.evaluate(() => JSON.parse(localStorage.getItem("s2s.conversation")).key), conversation.key);
});
