/** Interactive Claude permissions and questions, scoped to the live call. */
export class AgentPermissions {
  constructor() {
    this.cards = new Map();
    this.panel = document.createElement("section");
    this.panel.className = "agent-permissions";
    this.panel.setAttribute("aria-label", "Claude requests");
    this.panel.setAttribute("aria-live", "polite");
    this.panel.hidden = true;
    document.body.append(this.panel);
  }

  requested(event, reply) {
    if (this.cards.has(event.request_id)) return;
    const card = document.createElement("form");
    card.className = "agent-permission-card";
    card.dataset.requestId = event.request_id;
    const title = document.createElement("h2");
    title.textContent = event.title || `Claude requests ${event.tool_name}`;
    card.append(title);
    if (event.description) {
      const description = document.createElement("p");
      description.textContent = event.description;
      card.append(description);
    }
    const questions = event.tool_name === "AskUserQuestion" ? event.input?.questions || [] : [];
    const fields = [];
    for (const question of questions) {
      const label = document.createElement("label");
      label.textContent = question.question;
      const options = document.createElement("p");
      options.textContent = (question.options || []).map(o => `${o.label}: ${o.description || ""}`).join(" · ");
      const input = document.createElement("input");
      input.type = "text";
      input.required = true;
      input.placeholder = question.multiSelect ? "Choices separated by commas, or your own answer" : "Choice or your own answer";
      label.append(options, input);
      card.append(label);
      fields.push({ question, input });
    }
    if (!questions.length) {
      const details = document.createElement("pre");
      // Tool inputs are untrusted content. Display them as plain text only.
      details.textContent = JSON.stringify(event.input, null, 2);
      card.append(details);
    }
    const hint = document.createElement("p");
    hint.className = "agent-permission-hint";
    hint.textContent = questions.length === 1
      ? "Speak your answer when one request is pending, or use the form. Say ‘deny request’ to decline."
      : questions.length > 1
        ? "Answer each question below."
        : "Say ‘approve request’ or ‘deny request’, or choose below. Voice replies apply when one request is pending.";
    card.append(hint);
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    card.append(status);
    const actions = document.createElement("div");
    actions.className = "agent-permission-actions";
    const allow = document.createElement("button");
    allow.type = "submit";
    allow.textContent = questions.length ? "Send answers" : "Allow once";
    const deny = document.createElement("button");
    deny.type = "button";
    deny.textContent = "Deny";
    actions.append(allow, deny);
    card.append(actions);
    const send = decision => {
      const answers = fields.length && decision === "allow"
        ? Object.fromEntries(fields.map(({ question, input }) => [question.question,
          question.multiSelect ? input.value.split(",").map(v => v.trim()).filter(Boolean) : input.value.trim()]))
        : undefined;
      reply({ request_id: event.request_id, decision, ...(answers ? { answers } : {}) });
      allow.disabled = true;
      deny.disabled = true;
      status.textContent = "Sending reply…";
    };
    card.addEventListener("submit", e => { e.preventDefault(); send("allow"); });
    deny.addEventListener("click", () => send("deny"));
    this.cards.set(event.request_id, { card, status, allow, deny, responseId: event.response_id });
    this.panel.append(card);
    this.panel.hidden = false;
    if (this.cards.size === 1) (fields[0]?.input || allow).focus();
  }

  voice(event) {
    const entry = this.cards.get(event.request_id);
    if (!entry) return;
    entry.status.textContent = event.error || `Heard: ${event.transcript}`;
  }

  error(event) {
    const entry = this.cards.get(event.request_id);
    if (!entry) return;
    entry.status.textContent = event.error;
    entry.allow.disabled = false;
    entry.deny.disabled = false;
  }

  resolved(event) {
    const entry = this.cards.get(event.request_id);
    if (!entry) return;
    entry.card.remove();
    this.cards.delete(event.request_id);
    this.panel.hidden = this.cards.size === 0;
  }

  finish(responseId) {
    for (const [requestId, entry] of this.cards) {
      if (entry.responseId === responseId) this.resolved({ request_id: requestId });
    }
  }

  clear() {
    for (const requestId of this.cards.keys()) this.resolved({ request_id: requestId });
  }
}
