import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");

test("sessions remain nonmodal, show real inventory and route messages through the voice client", async (t) => {
  let fail = false;
  let permissionMode = "default";
  let permissionFail = false;
  const changes = [];
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    if (name === "/api/agent-sessions/peer_1/permission-mode") {
      const parts = [];
      for await (const part of req) parts.push(part);
      const change = JSON.parse(Buffer.concat(parts));
      changes.push(change);
      res.setHeader("Content-Type", "application/json");
      if (permissionFail) { res.writeHead(502).end(JSON.stringify({ detail: "Policy refused this mode" })); return; }
      permissionMode = change.mode;
      res.end(JSON.stringify({ session_id: "peer_1", permission_mode: permissionMode, permission_control: true }));
      return;
    }
    if (name === "/api/agent-sessions") {
      res.setHeader("Content-Type", "application/json");
      if (fail) { res.writeHead(502).end("{}"); return; }
      res.end(JSON.stringify({ enabled: true, sessions: [
        { session_id: "peer_1", name: "research", source: "app", state: "working", directory: "/tmp/project", working_with: true, queued_messages: 2, latest_reply: "Found the entry point.", permission_mode: permissionMode, permission_control: true },
        { session_id: "native_2", native_session_id: "native_2", name: "external <script>bad()</script>", source: "external", state: "idle", directory: "/tmp/other", permission_control: false, permission_control_reason: "This app has no control connection to this external session." },
      ] }));
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
    fail = false;
    permissionMode = "default";
    permissionFail = false;
    changes.length = 0;
    const page = await browser.newPage({ viewport: { width, height: 850 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("https://**/*", route => route.abort());
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async () => {
      const { ChatView } = await import("/ui/chat.js");
      const { AgentSessions } = await import("/ui/agent-sessions.js");
      const { S2sRealtimeClient } = await import("/s2s-realtime-client.js");
      document.body.classList.remove("booting");
      window.chat = new ChatView();
      window.sent = [];
      window.shown = [];
      window.client = new S2sRealtimeClient({ transport: "websocket" });
      client._session = { sendMessage: text => sent.push(text) };
      client.addEventListener("transcript", e => shown.push(e.detail.text));
      window.sessions = new AgentSessions({ sendMessage: (text, options) => client.sendUserText(text, options) });
      window.mainClicks = 0;
      document.querySelector("#settings-btn").addEventListener("click", () => mainClicks++);
    });
    await page.locator("#sessions-view").click();
    await page.waitForFunction(() => document.querySelectorAll(".session-card").length === 2);
    assert.match(await page.locator(".session-card").first().textContent(), /working.*2 queued/s);
    assert.match(await page.locator(".session-card").first().textContent(), /Linked to voice agent/);
    assert.equal(await page.locator("#sessions-list script").count(), 0);
    assert.equal(await page.locator("#chat-panel").evaluate(el => el.matches(':modal')), false);
    await page.locator("#settings-btn").click();
    assert.equal(await page.evaluate(() => mainClicks), 1);
    assert.equal(await page.locator("#agent-sessions").isVisible(), true);
    await page.locator(".session-card").first().click();
    assert.equal(await page.locator("#session-permission-mode").inputValue(), "default");
    await page.locator("#session-permission-mode").selectOption("acceptEdits");
    await page.locator("#sessions-refresh").click();
    await page.waitForFunction(() => !sessions.loading);
    assert.equal(await page.locator("#session-permission-mode").inputValue(), "acceptEdits", "Polling overwrote an unapplied choice");
    await page.locator("#session-permission-apply").click();
    await page.waitForFunction(() => document.querySelector("#session-permission-status").textContent.includes("applied"));
    assert.deepEqual(changes, [{ mode: "acceptEdits" }]);
    assert.deepEqual(await page.evaluate(() => sent), [], "Changing permissions invoked the model");
    await page.locator("#sessions-refresh").click();
    await page.waitForFunction(() => !sessions.loading);
    assert.match(await page.locator(".session-card").first().textContent(), /Permissions: Accept edits/);
    permissionFail = true;
    await page.locator("#session-permission-mode").selectOption("bypassPermissions");
    await page.locator("#session-permission-apply").click();
    await page.waitForFunction(() => document.querySelector("#session-permission-status").textContent.includes("Policy refused"));
    assert.equal(await page.evaluate(() => sessions.selected.permission_mode), "acceptEdits");
    assert.equal(await page.locator("#session-permission-apply").isEnabled(), true);
    await page.locator("#session-message").fill("Find the microphone handler");
    await page.locator("#session-message-form button").click();
    assert.match(await page.evaluate(() => sent[0]), /send_agent_message.*"session_id":"peer_1"/);
    assert.equal(await page.evaluate(() => shown[0]), "To research: Find the microphone handler");
    await page.locator(".session-card").nth(1).click();
    assert.equal(await page.locator("#session-permission-mode").isDisabled(), true);
    assert.equal(await page.locator("#session-permission-mode").inputValue(), "");
    assert.equal(await page.locator("#session-permission-apply").isDisabled(), true);
    assert.match(await page.locator("#session-permission-hint").textContent(), /no control connection/);
    await page.locator("#session-message").fill("Please review the update");
    await page.locator("#session-message-form button").click();
    assert.match(await page.evaluate(() => sent[1]), /send_external_agent_message.*"target":"native_2"/);
    fail = true;
    await page.locator("#sessions-refresh").click();
    await page.waitForFunction(() => document.querySelector("#sessions-status").textContent.includes("unavailable"));
    assert.equal(await page.locator(".session-card").count(), 2);
    await page.locator("#conversation-view").click();
    assert.equal(await page.locator("#chat-history").isVisible(), true);
    assert.equal(await page.locator("#agent-sessions").isVisible(), false);
    assert.deepEqual(errors, []);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.close();
  }
});
