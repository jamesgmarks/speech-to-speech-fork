import { $, escHtml } from "./dom.js";

/** Nonmodal sidebar view; polling discovers sessions without sending agent prompts. */
export class AgentSessions {
  constructor({ sendMessage }) {
    this.sendMessage = sendMessage;
    this.selected = null;
    this.visible = false;
    this.loading = false;
    this.timer = 0;
    this.panel = $("#agent-sessions");
    this.list = $("#sessions-list");
    this.status = $("#sessions-status");
    $("#sessions-view").addEventListener("click", () => this.show(true));
    $("#conversation-view").addEventListener("click", () => this.show(false));
    $("#sessions-refresh").addEventListener("click", () => this.refresh());
    new MutationObserver(() => {
      if (!$("#chat-panel").hidden && this.visible) void this.refresh();
    }).observe($("#chat-panel"), { attributes: true, attributeFilter: ["hidden"] });
    $("#session-message-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const message = $("#session-message").value.trim();
      if (!this.selected || !message) return;
      const tool = this.selected.source === "app" ? "send_agent_message" : "send_external_agent_message";
      const args = this.selected.source === "app"
        ? { session_id: this.selected.session_id, message }
        : { target: this.selected.native_session_id, message };
      try {
        this.sendMessage(`Please use mcp__speech_to_speech__${tool} with these parameters: ${JSON.stringify(args)}`, { displayText: `To ${this.selected.name}: ${message}` });
        $("#session-message").value = "";
        $("#session-message-status").textContent = "Sent to the voice agent. Approvals appear in the usual permission prompt.";
      } catch (error) {
        $("#session-message-status").textContent = error.message;
      }
    });
  }

  show(visible) {
    this.visible = visible;
    this.panel.hidden = !visible;
    $("#chat-history").hidden = visible;
    $("#chat-panel-title").textContent = visible ? "Sessions" : "Conversation";
    $("#conversation-view").setAttribute("aria-pressed", String(!visible));
    $("#sessions-view").setAttribute("aria-pressed", String(visible));
    clearTimeout(this.timer);
    if (visible) void this.refresh();
  }

  async refresh() {
    if (this.loading) return;
    this.loading = true;
    clearTimeout(this.timer);
    try {
      const response = await fetch("/api/agent-sessions", { cache: "no-store" });
      if (!response.ok) throw new Error("Sessions unavailable. Check the speech service, then refresh.");
      const data = await response.json();
      if (!Array.isArray(data.sessions)) throw new Error("The speech service returned an invalid session inventory.");
      this.status.textContent = data.discovery_error || (!data.enabled ? "Session tools are unavailable for this backend." : `${data.sessions.length} sessions · updates every 2 seconds`);
      const selectedId = this.selected?.session_id;
      const focusedId = document.activeElement?.closest(".session-card")?.dataset.sessionId;
      this.selected = data.sessions.find(s => s.session_id === selectedId) || null;
      this.list.replaceChildren();
      for (const session of data.sessions) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "session-card";
        row.dataset.sessionId = session.session_id;
        row.setAttribute("aria-pressed", String(session.session_id === selectedId));
        const state = String(session.state || "unknown").replaceAll("_", " ");
        row.innerHTML = `<strong>${escHtml(session.name)}</strong><span class="session-state">${escHtml(state)}${session.queued_messages ? ` · ${Number(session.queued_messages)} queued` : ""}</span><span>${session.source === "app" ? "App session" : "External session"}${session.working_with ? " · Linked to voice agent" : ""}</span><span>${escHtml(session.directory)}</span>${session.latest_reply ? `<pre>${escHtml(session.latest_reply)}</pre>` : ""}${session.error ? `<span>${escHtml(session.error)}</span>` : ""}`;
        row.addEventListener("click", () => {
          this.selected = session;
          for (const card of this.list.children) card.setAttribute("aria-pressed", String(card === row));
          this.updateComposer();
        });
        this.list.append(row);
        if (session.session_id === focusedId) row.focus({ preventScroll: true });
      }
      if (!data.sessions.length) this.list.textContent = "No sessions found. Ask the voice agent to start an independent session.";
      this.updateComposer();
    } catch (error) {
      this.status.textContent = error.message;
    } finally {
      this.loading = false;
      if (this.visible && !$("#chat-panel").hidden) this.timer = setTimeout(() => this.refresh(), 2000);
    }
  }

  updateComposer() {
    const stopped = ["stopped", "failed"].includes(this.selected?.state);
    $("#session-message-form").hidden = !this.selected || stopped;
    $("#session-message-target").textContent = this.selected?.name || "";
  }
}
