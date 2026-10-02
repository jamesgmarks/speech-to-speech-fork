/** Launch the personal voice app, using its existing model/voice configuration. */
import { spawn, execFileSync } from "node:child_process";
import { existsSync, mkdirSync, openSync, closeSync, chmodSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "node:net";
import { setTimeout as delay } from "node:timers/promises";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python = process.env.S2S_PYTHON || path.join(root, ".venv/bin/python");
const launcher = path.join(root, "voice-references/james/start-with-my-voice-claude.py");
const tunnelConfig = process.env.S2S_TUNNEL_CONFIG || path.join(homedir(), ".cloudflared/speech-to-speech.yml");
const argv = process.argv.slice(2);
const localOnly = argv.includes("--local");
const checkOnly = argv.includes("--check");
const backendArgs = argv.filter(arg => !["--local", "--check", "--"].includes(arg));
const logs = path.join(root, ".cache/app-launcher");
const children = [];
let stopping = false;
let shutdown;

function start(name, command, args, env = {}) {
  if (stopping) throw new Error("Startup cancelled.");
  const log = path.join(logs, `${name}.log`);
  const fd = openSync(log, "w", 0o600);
  chmodSync(log, 0o600);
  const child = spawn(command, args, {
    cwd: root, env: { ...process.env, ...env },
    stdio: ["ignore", fd, fd], detached: process.platform !== "win32",
  });
  closeSync(fd);
  children.push(child);
  child.on("error", error => {
    if (!stopping) { console.error(`${name}: ${error.message}`); void stop(1); }
  });
  child.on("exit", (code, signal) => {
    if (!stopping) {
      console.error(`${name} exited (${signal || code}). See ${log}`);
      void stop(code || 1);
    }
  });
  console.log(`Starting ${name}; log: ${log}`);
  return child;
}

function stop(code = 0) {
  if (shutdown) return shutdown;
  stopping = true;
  shutdown = (async () => {
    console.log("Stopping the app…");
    const exits = children.map(child => new Promise(resolve => {
      if (child.exitCode !== null || child.signalCode !== null || !child.pid) { resolve(); return; }
      child.once("exit", resolve);
      child.kill("SIGTERM"); // Give the backend time to save context and close SDK sessions.
    }));
    const timer = setTimeout(() => {
      for (const child of children) {
        if (!child.pid || child.exitCode !== null || child.signalCode !== null) continue;
        try { process.kill(process.platform === "win32" ? child.pid : -child.pid, "SIGKILL"); } catch {}
      }
    }, 20000);
    await Promise.all(exits);
    clearTimeout(timer);
    process.exitCode = code;
  })();
  return shutdown;
}

async function waitFor(label, ready, timeout = 120000) {
  const deadline = Date.now() + timeout;
  while (!stopping && Date.now() < deadline) {
    try { if (await ready()) return; } catch {}
    await delay(250);
  }
  throw new Error(stopping ? "Startup cancelled." : `${label} did not become ready. Check ${logs}`);
}

async function httpReady(url) {
  return (await fetch(url, { signal: AbortSignal.timeout(1500) })).ok;
}

async function checkPort(port) {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", () => reject(new Error(`Port ${port} is already in use. Stop the existing app before running npm start.`)));
    server.listen(port, "127.0.0.1", resolve);
  });
  await new Promise(resolve => server.close(resolve));
}

async function main() {
  for (const file of [python, launcher, path.join(root, "voice-references/james/reference.wav"), path.join(root, "voice-references/james/reference.txt")]) {
    if (!existsSync(file)) throw new Error(`Required local file is missing: ${file}`);
  }
  execFileSync(python, ["-c", "import sys; sys.path.insert(0, 'demo'); import server"], { cwd: root, stdio: "pipe" });
  if (!["demo/vendor/openai-realtime-agents.umd.js", "demo/node_modules/@openai/agents-realtime/dist/bundle/openai-realtime-agents.umd.js"].some(file => existsSync(path.join(root, file)))) {
    throw new Error("Browser SDK is missing. Run npm ci --prefix demo first.");
  }
  let tunnel = null;
  const cloudflared = process.env.S2S_CLOUDFLARED || "cloudflared";
  if (!localOnly && existsSync(tunnelConfig)) {
    const config = JSON.parse(execFileSync(python, ["-c", "import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))", tunnelConfig], { encoding: "utf8" }));
    const routes = config.ingress?.filter(route => route.hostname) || [];
    const hostname = routes[0]?.hostname;
    if (!hostname || !routes.every(route => {
      const access = route.originRequest?.access || config.originRequest?.access;
      return route.hostname === hostname && access?.required === true && access.teamName && access.audTag?.length;
    })) throw new Error("The tunnel configuration must protect the entire app hostname with Cloudflare Access origin validation.");
    execFileSync(cloudflared, ["tunnel", "--config", tunnelConfig, "ingress", "validate"], { stdio: "pipe" });
    tunnel = { hostname, id: config.tunnel };
  }
  console.log(`Model settings: ${path.relative(root, launcher)}`);
  console.log("Saved custom voices and the shared ongoing conversation are retained.");
  if (checkOnly) { console.log(`Configuration ready; tunnel ${tunnel ? `https://${tunnel.hostname}` : "disabled"}.`); return; }
  for (const port of [8765, 7860, ...(tunnel ? [7861] : [])]) await checkPort(port);
  mkdirSync(logs, { recursive: true, mode: 0o700 });
  chmodSync(logs, 0o700);
  start("backend", python, [launcher, ...backendArgs], { HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE || "1" });
  await waitFor("Speech backend", () => httpReady("http://127.0.0.1:8765/v1/pool"));
  const frontendEnv = {
    SPEECH_TO_SPEECH_URL: "ws://127.0.0.1:8765/v1/realtime",
    SPEECH_TO_SPEECH_CLIENT_TOOLS: "false", SPEECH_TO_SPEECH_SHARED_CONVERSATION: "true",
  };
  const frontendArgs = port => ["-m", "uvicorn", "--app-dir", path.join(root, "demo"), "server:app", "--host", "127.0.0.1", "--port", String(port)];
  start("localhost", python, frontendArgs(7860), { ...frontendEnv, SPEECH_TO_SPEECH_PUBLIC_URL: "", SPEECH_TO_SPEECH_RTC: "true" });
  await waitFor("Local frontend", () => httpReady("http://127.0.0.1:7860/api/conversation"));
  console.log("Local app: http://localhost:7860");
  if (tunnel) {
    start("public-frontend", python, frontendArgs(7861), {
      ...frontendEnv, SPEECH_TO_SPEECH_PUBLIC_URL: `wss://${tunnel.hostname}/v1/realtime`, SPEECH_TO_SPEECH_RTC: "false",
    });
    await waitFor("Public frontend", () => httpReady("http://127.0.0.1:7861/api/conversation"));
    start("cloudflare", cloudflared, ["tunnel", "--config", tunnelConfig, "--grace-period", "5s", "run", tunnel.id]);
    await waitFor("Cloudflare tunnel", () => readFileSync(path.join(logs, "cloudflare.log"), "utf8").includes("Registered tunnel connection"), 45000);
    console.log(`Protected app: https://${tunnel.hostname}`);
  }
  console.log("App is ready. Press Ctrl+C to stop it.");
}

process.on("SIGINT", () => void stop());
process.on("SIGTERM", () => void stop());
main().catch(async error => { console.error(error.message); await stop(1); });
