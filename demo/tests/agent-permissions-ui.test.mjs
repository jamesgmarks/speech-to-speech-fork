import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");

test("Claude permission cards send decisions and answers through the live transport", async (t) => {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
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
    await page.route("https://**/*", route => route.abort());
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async () => {
      const { AgentPermissions } = await import("/ui/agent-permissions.js");
      const { S2sRealtimeClient } = await import("/s2s-realtime-client.js");
      document.body.classList.remove("booting");
      window.permissions = new AgentPermissions();
      window.client = new S2sRealtimeClient({ transport: "webrtc" });
      window.sent = [];
      client._transport = { sendEvent: event => sent.push(event) };
      client.addEventListener("agent-permission-requested", e => permissions.requested(e.detail, reply => client.replyAgentPermission(reply)));
      client.addEventListener("agent-permission-resolved", e => permissions.resolved(e.detail));
      client.addEventListener("agent-permission-error", e => permissions.error(e.detail));
      client.addEventListener("agent-permission-voice", e => permissions.voice(e.detail));
      window.request = (id, tool, input) => client._onTransportEvent({
        type: "speech_to_speech.agent.permission.requested", request_id: id,
        tool_name: tool, input, response_id: "r1", timeout_s: 300,
      });
      request("write", "Write", { file_path: "/tmp/a.txt", content: "<img src=x onerror=alert(1)>" });
    });
    assert.equal(await page.locator(".agent-permission-card img").count(), 0);
    assert.match(await page.locator(".agent-permission-card pre").textContent(), /onerror/);
    const cardBox = await page.locator(".agent-permission-card").boundingBox();
    assert.ok(cardBox.x >= 0 && cardBox.x + cardBox.width <= width);
    await page.getByRole("button", { name: "Allow once" }).click();
    assert.deepEqual(await page.evaluate(() => sent[0]), {
      type: "speech_to_speech.agent.permission.reply", event_id: "agent_permission_write", request_id: "write", decision: "allow",
    });
    assert.equal(await page.getByRole("button", { name: "Deny", exact: true }).isDisabled(), true);
    await page.evaluate(() => client._onTransportEvent({ type: "error", error: { event_id: "agent_permission_write", message: "Please retry" } }));
    assert.equal(await page.getByRole("button", { name: "Deny", exact: true }).isEnabled(), true);
    await page.getByRole("button", { name: "Deny", exact: true }).click();
    assert.equal(await page.evaluate(() => sent[1].decision), "deny");
    await page.evaluate(() => {
      client._onTransportEvent({ type: "speech_to_speech.agent.permission.resolved", request_id: "write", status: "denied" });
      request("ask", "AskUserQuestion", { questions: [
        { question: "Which style?", options: [{ label: "Brief", description: "Fast answer" }], multiSelect: false },
        { question: "Which colors?", options: [], multiSelect: true },
      ] });
    });
    const inputs = page.locator(".agent-permission-card input");
    await inputs.nth(0).fill("Brief");
    await inputs.nth(1).fill("blue, green");
    await page.getByRole("button", { name: "Send answers" }).click();
    assert.deepEqual(await page.evaluate(() => sent[2].answers), { "Which style?": "Brief", "Which colors?": ["blue", "green"] });
    await page.evaluate(() => permissions.finish("r1"));
    assert.equal(await page.locator(".agent-permission-card").count(), 0);
    await page.close();
  }
});
