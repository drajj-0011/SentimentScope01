/* SentimentScope front-end — no frameworks, no build step.
   Author: Debraj */

const $ = (id) => document.getElementById(id);
const input = $("input");
const runBtn = $("run");
const errorBox = $("error");
const result = $("result");

const LABEL_ORDER = ["positive", "neutral", "negative"];

/* ---------------- boot ---------------- */

async function boot() {
  $("char-count").textContent = input.value.length;

  try {
    const info = await fetch("/api/model").then((r) => r.json());
    $("badge-health").textContent = "model loaded";
    $("badge-health").classList.add("ok");
    $("badge-vocab").textContent = `${info.vocabulary_size.toLocaleString()} features`;
    const acc = info.metrics && info.metrics.accuracy;
    $("badge-acc").textContent = acc != null ? `${(acc * 100).toFixed(1)}% test accuracy` : "accuracy n/a";
  } catch {
    $("badge-health").textContent = "model unavailable";
  }

  try {
    const { examples } = await fetch("/api/examples").then((r) => r.json());
    const box = $("examples");
    examples.forEach((ex) => {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.type = "button";
      chip.textContent = ex.hint;
      chip.title = ex.text;
      chip.addEventListener("click", () => {
        input.value = ex.text;
        $("char-count").textContent = ex.text.length;
        analyse();
      });
      box.appendChild(chip);
    });
  } catch {
    /* examples are optional */
  }
}

/* ---------------- predict ---------------- */

async function analyse() {
  const text = input.value.trim();
  hideError();

  if (!text) {
    showError("Type a sentence first.");
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = "Analysing…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, explain: true }),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (HTTP ${res.status})`);
    }

    render(await res.json());
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Analyse";
  }
}

function render(data) {
  result.hidden = false;

  $("verdict-label").textContent = data.label;
  const dot = $("verdict-dot");
  dot.className = `dot ${data.label}`;
  $("verdict-label").style.color = `var(--${data.label})`;

  $("result-meta").textContent =
    `confidence ${(data.confidence * 100).toFixed(1)}% · ${data.latency_ms.toFixed(1)} ms · model v${data.model_version}`;

  const bars = $("bars");
  bars.innerHTML = "";
  LABEL_ORDER.forEach((label) => {
    const value = data.scores[label] ?? 0;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-name">${label}</span>
      <span class="bar-track"><span class="bar-fill ${label}" data-w="${value}"></span></span>
      <span class="bar-value">${(value * 100).toFixed(1)}%</span>`;
    bars.appendChild(row);
  });

  requestAnimationFrame(() => {
    bars.querySelectorAll(".bar-fill").forEach((el) => {
      el.style.width = `${Math.max(1.5, Number(el.dataset.w) * 100)}%`;
    });
  });

  $("explain-label").textContent = data.label;
  const tokens = $("tokens");
  tokens.innerHTML = "";

  if (!data.explanation || data.explanation.length === 0) {
    tokens.innerHTML = '<span class="hint">No known tokens — the model has not seen these words during training.</span>';
  } else {
    data.explanation.forEach((t) => {
      const span = document.createElement("span");
      const positive = t.weight >= 0;
      span.className = `token ${positive ? "pos" : "neg"}`;
      span.innerHTML = `${escapeHtml(t.token)}<small>${t.weight > 0 ? "+" : ""}${t.weight.toFixed(3)}</small>`;
      tokens.appendChild(span);
    });
  }

  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------------- helpers ---------------- */

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

input.addEventListener("input", () => {
  $("char-count").textContent = input.value.length;
  hideError();
});

runBtn.addEventListener("click", analyse);

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    analyse();
  }
});

boot();
