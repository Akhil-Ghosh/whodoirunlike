const state = {
  runs: [],
  current: null,
  samJob: null,
  selection: { type: "unset", positive_points: [], negative_points: [], box: null, mask_path: null },
  promptMode: "positive",
  boxStart: null,
  draftBox: null,
  saveTimer: 0,
  samPollTimer: 0,
};

const svgNS = "http://www.w3.org/2000/svg";

const els = {
  saveState: document.querySelector("#saveState"),
  runCount: document.querySelector("#runCount"),
  runList: document.querySelector("#runList"),
  runBucket: document.querySelector("#runBucket"),
  runTitle: document.querySelector("#runTitle"),
  runMeta: document.querySelector("#runMeta"),
  manifestLink: document.querySelector("#manifestLink"),
  promptModes: document.querySelectorAll("[data-prompt-mode]"),
  undoButton: document.querySelector("#undoButton"),
  clearButton: document.querySelector("#clearButton"),
  savePromptButton: document.querySelector("#savePromptButton"),
  promptImage: document.querySelector("#promptImage"),
  promptOverlay: document.querySelector("#promptOverlay"),
  promptEmpty: document.querySelector("#promptEmpty"),
  selectionSummary: document.querySelector("#selectionSummary"),
  frameSummary: document.querySelector("#frameSummary"),
  angleSummary: document.querySelector("#angleSummary"),
  durationPill: document.querySelector("#durationPill"),
  segmentVideo: document.querySelector("#segmentVideo"),
  segmentEmpty: document.querySelector("#segmentEmpty"),
  promptState: document.querySelector("#promptState"),
  samJobState: document.querySelector("#samJobState"),
  runSamButton: document.querySelector("#runSamButton"),
  stageList: document.querySelector("#stageList"),
  artifactGrid: document.querySelector("#artifactGrid"),
  promptJson: document.querySelector("#promptJson"),
  copyPromptButton: document.querySelector("#copyPromptButton"),
};

function setSaveState(label, className = "") {
  window.clearTimeout(state.saveTimer);
  els.saveState.textContent = label;
  els.saveState.className = `save-state ${className}`.trim();
  if (label === "Saved") {
    state.saveTimer = window.setTimeout(() => setSaveState("Ready"), 1400);
  }
}

function formatTime(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) return "--";
  const minutes = Math.floor(numeric / 60);
  const seconds = Math.floor(numeric % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function clampUnit(value) {
  return Math.max(0, Math.min(1, value));
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function normalizedSelection() {
  const positive = state.selection.positive_points || [];
  const negative = state.selection.negative_points || [];
  const box = state.selection.box || null;
  return {
    type: box ? "box" : positive.length || negative.length ? "point" : "unset",
    positive_points: positive,
    negative_points: negative,
    box,
    mask_path: state.selection.mask_path || null,
  };
}

function setPromptMode(mode) {
  state.promptMode = mode;
  els.promptModes.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.promptMode === mode);
  });
}

function pointerPosition(event) {
  const rect = els.promptOverlay.getBoundingClientRect();
  return {
    x: Number(clampUnit((event.clientX - rect.left) / rect.width).toFixed(6)),
    y: Number(clampUnit((event.clientY - rect.top) / rect.height).toFixed(6)),
  };
}

function makeSvg(name, attributes = {}) {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function renderOverlay() {
  els.promptOverlay.innerHTML = "";
  const selection = normalizedSelection();
  if (selection.box) {
    els.promptOverlay.appendChild(
      makeSvg("rect", {
        class: "prompt-box",
        x: selection.box.x,
        y: selection.box.y,
        width: selection.box.width,
        height: selection.box.height,
      })
    );
  }
  if (state.draftBox) {
    els.promptOverlay.appendChild(
      makeSvg("rect", {
        class: "prompt-box is-draft",
        x: state.draftBox.x,
        y: state.draftBox.y,
        width: state.draftBox.width,
        height: state.draftBox.height,
      })
    );
  }
  selection.positive_points.forEach((point) => {
    els.promptOverlay.appendChild(
      makeSvg("circle", { class: "prompt-point is-positive", cx: point.x, cy: point.y, r: 0.012 })
    );
  });
  selection.negative_points.forEach((point) => {
    els.promptOverlay.appendChild(
      makeSvg("circle", { class: "prompt-point is-negative", cx: point.x, cy: point.y, r: 0.012 })
    );
  });
  renderSelectionSummary();
}

function renderSelectionSummary() {
  const selection = normalizedSelection();
  const positiveCount = selection.positive_points.length;
  const negativeCount = selection.negative_points.length;
  const boxLabel = selection.box ? "box" : "no box";
  const summary =
    selection.type === "unset" ? "Unset" : `${positiveCount}+ / ${negativeCount}- / ${boxLabel}`;
  els.selectionSummary.textContent = summary;
  els.promptState.textContent = selection.type === "unset" ? "Unset" : "Ready";
  els.promptJson.textContent = JSON.stringify(selection, null, 2);
  renderSamJobStatus();
}

function clearSamPoll() {
  window.clearTimeout(state.samPollTimer);
  state.samPollTimer = 0;
}

function scheduleSamPoll() {
  clearSamPoll();
  state.samPollTimer = window.setTimeout(() => {
    refreshSamJobStatus({ reloadOnComplete: true });
  }, 2000);
}

function samStatusLabel(status) {
  if (status === "completed") return "Complete";
  if (status === "failed") return "Failed";
  if (status === "running") return "Running";
  return "Idle";
}

function renderSamJobStatus(job = state.samJob) {
  state.samJob = job || { status: "idle" };
  const status = state.samJob.status || "idle";
  const selection = normalizedSelection();
  const hasPrompt = selection.type !== "unset";
  const artifactsReady = Boolean(
    state.current?.artifacts?.runner_mask?.exists || state.current?.artifacts?.qa_overlay?.exists
  );

  let statusLabel = samStatusLabel(status);
  if (status === "idle" && artifactsReady) statusLabel = "Artifacts ready";

  els.samJobState.textContent = statusLabel;
  els.samJobState.className = `rank-pill sam-job-state ${
    status === "running" ? "is-running" : status === "completed" || artifactsReady ? "is-complete" : ""
  } ${status === "failed" ? "is-error" : ""}`.trim();

  const isRunning = status === "running";
  els.runSamButton.disabled = !state.current || !hasPrompt || isRunning;
  els.runSamButton.textContent = isRunning
    ? "Running..."
    : artifactsReady || status === "completed"
      ? "Run SAM 2 Again"
      : "Run SAM 2";
  els.runSamButton.title = hasPrompt ? "Run SAM 2 on the saved subject prompt" : "Select the runner first";
}

function renderRunList() {
  els.runList.innerHTML = "";
  els.runCount.textContent = `${state.runs.length} run${state.runs.length === 1 ? "" : "s"}`;
  state.runs.forEach((run) => {
    const button = document.createElement("button");
    button.className = `queue-item ${run.candidate_id === state.current?.candidate_id ? "is-active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <div class="queue-kicker">
        <span>${run.runner_name}</span>
        <span class="quality-dot ${run.prompt_ready ? "good" : "mid"}">
          ${run.prompt_ready ? "ready" : "open"}
        </span>
      </div>
      <div class="queue-title">${run.title}</div>
      <div class="queue-meta">
        <span>${run.camera_angle || "unknown"}</span>
        <span>${formatTime(run.duration_seconds)}</span>
      </div>
    `;
    button.addEventListener("click", () => loadRun(run.candidate_id));
    els.runList.appendChild(button);
  });
}

function stageLabel(key) {
  return key.replaceAll("_", " ");
}

function renderStages(stages) {
  els.stageList.innerHTML = "";
  Object.entries(stages || {}).forEach(([key, stage]) => {
    const row = document.createElement("div");
    const status = stage?.status || "pending";
    row.className = "stage-row";
    row.innerHTML = `
      <span>${stageLabel(key)}</span>
      <strong class="${status.includes("complete") || status === "ready" ? "is-ready" : ""}">
        ${status.replaceAll("_", " ")}
      </strong>
    `;
    els.stageList.appendChild(row);
  });
}

function renderArtifactPreview(artifact) {
  if (!artifact.exists || !artifact.url) {
    return `<div class="artifact-empty">Pending</div>`;
  }
  if (artifact.kind === "image") {
    return `<img src="${artifact.url}" alt="${artifact.label}" loading="lazy" />`;
  }
  if (artifact.kind === "video") {
    return `<video src="${artifact.url}" muted controls playsinline preload="metadata"></video>`;
  }
  return `<a class="ghost-button" href="${artifact.url}" target="_blank" rel="noreferrer">Open</a>`;
}

function renderArtifacts(artifacts) {
  const preferredOrder = [
    "source_segment",
    "prompt_frame",
    "person_prompt",
    "runner_mask",
    "pose_landmarks",
    "densepose",
    "skeleton_render",
    "masked_runner",
    "qa_overlay",
    "features",
  ];
  els.artifactGrid.innerHTML = preferredOrder
    .filter((key) => artifacts?.[key])
    .map((key) => {
      const artifact = artifacts[key];
      return `
        <article class="artifact-tile ${artifact.exists ? "is-ready" : ""}">
          <div class="artifact-preview">${renderArtifactPreview(artifact)}</div>
          <div class="artifact-caption">
            <span>${artifact.label}</span>
            <strong>${artifact.exists ? "ready" : "pending"}</strong>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderRun() {
  const run = state.current;
  if (!run) return;

  const manifest = run.manifest || {};
  const review = manifest.review || {};
  const source = manifest.source || {};
  const promptArtifact = run.artifacts.prompt_frame;
  const segmentArtifact = run.artifacts.source_segment;

  state.selection = {
    type: "unset",
    positive_points: [],
    negative_points: [],
    box: null,
    mask_path: null,
    ...(run.prompt?.selection || {}),
  };

  els.runBucket.textContent = review.primary_bucket || "CV run";
  els.runTitle.textContent = source.title || "Untitled run";
  els.runMeta.textContent = `${manifest.runner_name || "Unknown runner"} · ${run.candidate_id}`;
  els.angleSummary.textContent = review.camera_angle || "unknown";
  els.durationPill.textContent = formatTime(review.duration_seconds);
  els.manifestLink.href = run.artifacts.person_prompt?.url || "#";

  if (promptArtifact?.exists && promptArtifact.url) {
    els.promptImage.src = `${promptArtifact.url}?t=${Date.now()}`;
    els.promptEmpty.classList.add("is-hidden");
  } else {
    els.promptImage.removeAttribute("src");
    els.promptEmpty.classList.remove("is-hidden");
  }

  const frame = run.prompt?.frame || {};
  els.frameSummary.textContent = frame.width && frame.height ? `${frame.width} x ${frame.height}` : "--";

  if (segmentArtifact?.exists && segmentArtifact.url) {
    els.segmentVideo.src = segmentArtifact.url;
    els.segmentEmpty.classList.add("is-hidden");
  } else {
    els.segmentVideo.removeAttribute("src");
    els.segmentEmpty.classList.remove("is-hidden");
  }

  renderOverlay();
  renderStages(run.stages);
  renderArtifacts(run.artifacts);
  renderRunList();
  renderSamJobStatus();
  setSaveState("Ready");
}

async function loadRun(candidateId, options = {}) {
  const { refreshJob = true } = options;
  try {
    clearSamPoll();
    setSaveState("Loading", "is-saving");
    state.samJob = null;
    state.current = await fetchJson(`/api/cv-runs/${candidateId}`);
    renderRun();
    if (refreshJob) {
      await refreshSamJobStatus();
    }
  } catch (error) {
    setSaveState("Load failed", "is-error");
    els.runTitle.textContent = "Could not load CV run";
    els.runMeta.textContent = String(error);
  }
}

function addPromptPoint(event) {
  if (!state.current || state.promptMode === "box") return;
  const point = pointerPosition(event);
  if (state.promptMode === "negative") {
    state.selection.negative_points = [...(state.selection.negative_points || []), point];
  } else {
    state.selection.positive_points = [...(state.selection.positive_points || []), point];
  }
  renderOverlay();
}

function boxFromPoints(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);
  if (width < 0.005 || height < 0.005) return null;
  return {
    x: Number(x.toFixed(6)),
    y: Number(y.toFixed(6)),
    width: Number(width.toFixed(6)),
    height: Number(height.toFixed(6)),
  };
}

function undoSelection() {
  if (state.selection.box) {
    state.selection.box = null;
  } else if (state.promptMode === "negative" && state.selection.negative_points?.length) {
    state.selection.negative_points = state.selection.negative_points.slice(0, -1);
  } else if (state.selection.positive_points?.length) {
    state.selection.positive_points = state.selection.positive_points.slice(0, -1);
  } else if (state.selection.negative_points?.length) {
    state.selection.negative_points = state.selection.negative_points.slice(0, -1);
  }
  renderOverlay();
}

function clearSelection() {
  state.selection = { type: "unset", positive_points: [], negative_points: [], box: null, mask_path: null };
  state.draftBox = null;
  renderOverlay();
}

async function savePrompt(options = {}) {
  const { rerender = true, flash = true } = options;
  if (!state.current) return null;
  try {
    setSaveState("Saving", "is-saving");
    state.current = await fetchJson(`/api/cv-runs/${state.current.candidate_id}/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selection: normalizedSelection() }),
    });
    const run = state.runs.find((item) => item.candidate_id === state.current.candidate_id);
    if (run) run.prompt_ready = true;
    if (rerender) {
      renderRun();
    } else {
      renderRunList();
      renderStages(state.current.stages);
      renderArtifacts(state.current.artifacts);
      renderSelectionSummary();
    }
    if (flash) setSaveState("Saved");
    return state.current;
  } catch (error) {
    setSaveState("Save failed", "is-error");
    els.runMeta.textContent = String(error);
    return null;
  }
}

async function refreshSamJobStatus(options = {}) {
  const { reloadOnComplete = false } = options;
  if (!state.current) return null;
  const candidateId = state.current.candidate_id;
  try {
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/sam2`);
    const job = payload.job || { status: "idle" };
    renderSamJobStatus(job);
    if (job.status === "running") {
      scheduleSamPoll();
    } else {
      clearSamPoll();
      if (reloadOnComplete && job.status === "completed" && state.current?.candidate_id === candidateId) {
        await loadRun(candidateId, { refreshJob: false });
        renderSamJobStatus(job);
        setSaveState("SAM 2 complete");
      }
      if (job.status === "failed") {
        setSaveState("SAM 2 failed", "is-error");
      }
    }
    return job;
  } catch (error) {
    clearSamPoll();
    els.samJobState.textContent = "Unavailable";
    els.samJobState.className = "rank-pill sam-job-state is-error";
    els.runMeta.textContent = String(error);
    return null;
  }
}

async function startSamRun() {
  if (!state.current) return;
  if (normalizedSelection().type === "unset") {
    setSaveState("Select target first", "is-error");
    renderSamJobStatus();
    return;
  }

  const candidateId = state.current.candidate_id;
  const savedRun = await savePrompt({ rerender: false, flash: false });
  if (!savedRun || state.current?.candidate_id !== candidateId) return;

  try {
    setSaveState("Starting SAM 2", "is-saving");
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/sam2`, { method: "POST" });
    renderSamJobStatus(payload.job);
    setSaveState("SAM 2 running", "is-saving");
    scheduleSamPoll();
  } catch (error) {
    setSaveState("Start failed", "is-error");
    els.runMeta.textContent = String(error);
    await refreshSamJobStatus();
  }
}

async function init() {
  try {
    const payload = await fetchJson("/api/cv-runs");
    state.runs = payload.runs || [];
    renderRunList();
    if (state.runs.length) {
      await loadRun(state.runs[0].candidate_id);
    } else {
      setSaveState("No runs", "is-error");
      els.runTitle.textContent = "No prepared CV runs";
      els.runMeta.textContent = "Run scripts/prepare_single_clip_cv_run.py first.";
    }
  } catch (error) {
    setSaveState("Load failed", "is-error");
    els.runTitle.textContent = "Could not load CV runs";
    els.runMeta.textContent = String(error);
  }
}

els.promptModes.forEach((button) => {
  button.addEventListener("click", () => setPromptMode(button.dataset.promptMode));
});

els.promptOverlay.addEventListener("click", addPromptPoint);

els.promptOverlay.addEventListener("pointerdown", (event) => {
  if (state.promptMode !== "box") return;
  state.boxStart = pointerPosition(event);
  state.draftBox = null;
  els.promptOverlay.setPointerCapture(event.pointerId);
});

els.promptOverlay.addEventListener("pointermove", (event) => {
  if (state.promptMode !== "box" || !state.boxStart) return;
  state.draftBox = boxFromPoints(state.boxStart, pointerPosition(event));
  renderOverlay();
});

els.promptOverlay.addEventListener("pointerup", (event) => {
  if (state.promptMode !== "box" || !state.boxStart) return;
  const box = boxFromPoints(state.boxStart, pointerPosition(event));
  state.boxStart = null;
  state.draftBox = null;
  if (box) {
    state.selection.box = box;
  }
  renderOverlay();
});

els.undoButton.addEventListener("click", undoSelection);
els.clearButton.addEventListener("click", clearSelection);
els.savePromptButton.addEventListener("click", savePrompt);
els.runSamButton.addEventListener("click", startSamRun);
els.copyPromptButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(els.promptJson.textContent);
  setSaveState("Copied");
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.key === "1") setPromptMode("positive");
  if (event.key === "2") setPromptMode("negative");
  if (event.key === "3") setPromptMode("box");
  if (event.key === "z") undoSelection();
  if (event.key === "s") savePrompt();
});

init();
