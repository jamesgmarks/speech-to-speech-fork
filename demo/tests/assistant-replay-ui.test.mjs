import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
const key = "88edaf87-91f2-4fc3-a153-7b5355fdd7d5";
const audioUrl = id => `/v1/conversation/audio/${key}/${id.repeat(64)}.wav`;
const wav = Buffer.alloc(44 + 16000 * 2 * 5);
wav.write("RIFF", 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write("WAVEfmt ", 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28);
wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write("data", 36); wav.writeUInt32LE(wav.length - 44, 40);

test("assistant buttons replay original speech, mute capture, and survive reloads across origins", async t => {
  const requests = [];
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    if (name.startsWith("/api/conversation/audio/")) {
      requests.push(name);
      if (name.endsWith("e".repeat(64) + ".wav")) { res.writeHead(404).end(); return; }
      res.setHeader("Content-Type", "audio/wav");
      res.end(wav);
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
  const initialize = page => page.evaluate(async () => {
    const { ChatView } = await import("/ui/chat.js");
    window.userMuted = false;
    window.captureMuted = false;
    window.muteEvents = [];
    window.chat = new ChatView({ onAudioPlaybackChange(playing) {
      window.captureMuted = window.userMuted || playing;
      window.muteEvents.push(playing);
    } });
  });
  for (const width of [1280, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 850 } });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(() => localStorage.clear());
    await initialize(page);
    await page.evaluate(({ key, urls }) => {
      chat.adoptSharedConversation({ key, backend: "claude-agent-sdk", reset: false, history: [] });
      for (const [index, url] of urls.entries()) {
        const responseId = `resp_${index}`;
        chat.onTranscript({ role: "assistant", text: `Response ${index}.`, responseId, partial: false });
        chat.onResponseFinished({ responseId, status: "completed", audioUrl: url });
      }
      chat.onTranscript({ role: "assistant", text: "An older response with no recording.", responseId: "old", partial: false });
      chat.onResponseFinished({ responseId: "old", status: "completed" });
    }, { key, urls: [audioUrl("a"), audioUrl("b")] });
    const buttons = page.getByRole("button", { name: "Replay assistant audio", exact: true });
    assert.equal(await buttons.count(), 2);
    const texts = await page.locator(".hist-body").allTextContents();
    await buttons.first().click();
    await page.waitForFunction(() => !document.querySelector(".hist-msg.assistant audio").paused);
    assert.equal(await page.evaluate(() => captureMuted), true);
    await page.getByRole("button", { name: "Replay assistant audio", exact: true }).click();
    await page.waitForFunction(() => {
      const [a, b] = document.querySelectorAll(".hist-msg.assistant audio");
      return a.paused && !b.paused;
    });
    assert.equal(await page.evaluate(() => captureMuted), true);
    // A normal shared-history poll must not recreate the rows or stop replay.
    await page.evaluate(() => chat.adoptSharedConversation({
      key: chat.conversationKey, backend: chat.conversationBackend, reset: false,
      history: chat._historyMessages(),
    }));
    assert.equal(await page.getByRole("button", { name: "Stop replaying assistant audio" }).count(), 1);
    await page.getByRole("button", { name: "Stop replaying assistant audio" }).click();
    await page.waitForFunction(() => !captureMuted);
    assert.deepEqual(await page.locator(".hist-body").allTextContents(), texts);
    // Honor the user's existing mute setting after replay ends.
    await page.evaluate(() => { userMuted = true; });
    await buttons.first().click();
    await page.getByRole("button", { name: "Stop replaying assistant audio" }).click();
    assert.equal(await page.evaluate(() => captureMuted), true);
    // User and assistant recordings share one exclusive player.
    await page.evaluate(async () => {
      userMuted = false;
      const { pcm16ToWavBlob } = await import("/ws/user-audio-recorder.js");
      chat.onUserAudio({ itemId: "u", audio: pcm16ToWavBlob(new Uint8Array(24000 * 2 * 5)) });
      await document.querySelector(".hist-msg.user audio").play();
    });
    await buttons.first().click();
    assert.equal(await page.locator(".hist-msg.user audio").evaluate(el => el.paused), true);
    assert.equal(await page.evaluate(() => captureMuted), true);
    await page.evaluate(() => { chat.reset({ dismiss: true }); chat._persistConversation(); });
    await page.reload();
    await initialize(page);
    assert.equal(await buttons.count(), 2);
    await buttons.first().click();
    await page.waitForFunction(() => !document.querySelector(".hist-msg.assistant audio").paused);
    await page.evaluate(() => { const audio = document.querySelector(".hist-msg.assistant audio"); audio.currentTime = audio.duration - 0.05; });
    await page.waitForFunction(() => !captureMuted);
    // A missing recording reports failure on the control and releases mute.
    await page.evaluate(url => {
      chat.onResponseFinished({ responseId: "missing", status: "completed", transcript: "Missing audio.", audioUrl: url });
    }, audioUrl("e"));
    await buttons.last().click();
    await page.waitForFunction(() => !captureMuted);
    assert.match(await buttons.last().getAttribute("title"), /could not be played/);
    await page.evaluate(() => chat.startNewConversation());
    assert.equal(await buttons.count(), 0);
    assert.equal(await page.evaluate(() => captureMuted), false);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.close();
  }
  // Server metadata is portable; another browser/origin needs no local audio.
  const page = await browser.newPage();
  await page.goto(`http://localhost:${server.address().port}/`);
  await initialize(page);
  await page.evaluate(({ key, url }) => chat.adoptSharedConversation({ key, backend: "claude-agent-sdk", reset: false,
    history: [{ role: "assistant", text: "A restored reply.", audio_url: url }] }), { key, url: audioUrl("a") });
  await page.getByRole("button", { name: "Replay assistant audio", exact: true }).click();
  await page.waitForFunction(() => !document.querySelector(".hist-msg.assistant audio").paused);
  assert.equal(await page.locator(".hist-msg.assistant audio").evaluate(el => new URL(el.src).hostname), "localhost");
  assert.ok(requests.includes(audioUrl("a").replace("/v1/", "/api/")));
});
