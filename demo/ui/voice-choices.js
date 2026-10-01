// @ts-check

/** Populate the existing selector with voices supported by the loaded backend.
 *  Return a valid saved choice, or the backend's default for an older setting.
 *  @param {HTMLSelectElement} select
 *  @param {{ voices?: Array<{id: string, name: string, kind: string}>, default?: string }} catalog
 *  @param {string} saved
 *  @returns {string}
 */
export function populateVoiceChoices(select, catalog, saved) {
  const voices = catalog.voices;
  if (!Array.isArray(voices) || voices.length === 0) return saved;
  if (voices.some(v => !v || typeof v.id !== "string" || !v.id || typeof v.name !== "string" || !["builtin", "custom"].includes(v.kind))) return saved;
  select.replaceChildren();
  for (const kind of ["builtin", "custom"]) {
    const choices = voices.filter(v => v.kind === kind);
    if (!choices.length) continue;
    const group = document.createElement("optgroup");
    group.label = kind === "custom" ? "Custom voices" : "Built-in voices";
    for (const voice of choices) {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = voice.name;
      group.append(option);
    }
    select.append(group);
  }
  const ids = Array.from(select.options, option => option.value);
  if (!ids.length) return saved;
  const selected = ids.includes(saved) ? saved : ids.includes(catalog.default || "") ? catalog.default : ids[0];
  select.value = selected || ids[0];
  return select.value;
}
