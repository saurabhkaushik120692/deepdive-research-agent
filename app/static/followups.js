// Follow-up questions under a report: ask, show the stage while it runs, then re-render.
(() => {
  const LABELS = {
    waiting: "Waiting for a free slot…",
    scope: "Checking it's on topic…",
    drafting: "Checking the report's sources…",
    searching: "Searching the web for more…",
    writing: "Writing the answer…",
  };
  const DONE = ["answered", "declined", "failed"];
  const POLL_MS = 2000;

  const section = () => document.getElementById("followups");

  async function refresh(runId) {
    const r = await fetch(`/runs/${runId}/followups`);
    if (!r.ok) return;
    section().outerHTML = await r.text();
    resume();
  }

  function poll(runId, el) {
    const tick = async () => {
      let f;
      try {
        const r = await fetch(`/api/runs/${runId}/followups/${el.dataset.followup}`);
        if (!r.ok) return;
        f = await r.json();
      } catch {
        setTimeout(tick, POLL_MS * 2); // network blip: try again a bit later
        return;
      }
      if (DONE.includes(f.status)) return refresh(runId);
      const stage = f.status === "queued" ? "waiting" : f.stage;
      el.querySelector(".fu-stage").textContent = LABELS[stage] || LABELS.waiting;
      setTimeout(tick, POLL_MS);
    };
    setTimeout(tick, POLL_MS);
  }

  function resume() {
    const s = section();
    if (!s) return;
    s.querySelectorAll("[data-followup]").forEach((el) => poll(s.dataset.run, el));
  }

  function showError(form, text) {
    const p = form.querySelector(".fu-error");
    p.textContent = text;
    p.hidden = false;
  }

  // Delegated: on the live run page the report (and this form) arrives after the page loads.
  document.addEventListener("submit", async (e) => {
    const form = e.target;
    if (form.id !== "followup-form") return;
    e.preventDefault();
    const runId = section().dataset.run;
    const button = form.querySelector("button");
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    try {
      const r = await fetch(`/api/runs/${runId}/followups`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: form.question.value.trim() }),
      });
      if (r.ok) return refresh(runId);
      const body = await r.json().catch(() => ({}));
      const detail =
        typeof body.detail === "string"
          ? body.detail
          : "Please write a question of 3 to 500 characters.";
      showError(form, detail.charAt(0).toUpperCase() + detail.slice(1));
    } catch {
      showError(form, "Couldn't reach the server. Check your connection and try again.");
    }
    button.disabled = false;
    button.removeAttribute("aria-busy");
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", resume);
  } else {
    resume();
  }
})();
