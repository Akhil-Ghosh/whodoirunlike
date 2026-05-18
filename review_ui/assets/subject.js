const state = {
  runs: [],
  current: null,
  samJob: null,
  maskBackend: "sam2",
  maskQualityMode: "native",
  subjectCandidates: [],
  subjectCandidatesStatus: "idle",
  selection: {
    type: "unset",
    positive_points: [],
    negative_points: [],
    box: null,
    mask_path: null,
    subject_candidate: null,
  },
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
  detectCandidatesButton: document.querySelector("#detectCandidatesButton"),
  undoButton: document.querySelector("#undoButton"),
  clearButton: document.querySelector("#clearButton"),
  savePromptButton: document.querySelector("#savePromptButton"),
  promptImage: document.querySelector("#promptImage"),
  promptOverlay: document.querySelector("#promptOverlay"),
  promptEmpty: document.querySelector("#promptEmpty"),
  candidateSummary: document.querySelector("#candidateSummary"),
  selectionSummary: document.querySelector("#selectionSummary"),
  frameSummary: document.querySelector("#frameSummary"),
  angleSummary: document.querySelector("#angleSummary"),
  durationPill: document.querySelector("#durationPill"),
  segmentVideo: document.querySelector("#segmentVideo"),
  segmentEmpty: document.querySelector("#segmentEmpty"),
  promptState: document.querySelector("#promptState"),
  samJobState: document.querySelector("#samJobState"),
  maskBackendSelect: document.querySelector("#maskBackendSelect"),
  maskQualityLabel: document.querySelector("#maskQualityLabel"),
  maskQualitySelect: document.querySelector("#maskQualitySelect"),
  runSamButton: document.querySelector("#runSamButton"),
  maskProgress: document.querySelector("#maskProgress"),
  maskProgressLabel: document.querySelector("#maskProgressLabel"),
  maskProgressEta: document.querySelector("#maskProgressEta"),
  maskProgressBar: document.querySelector("#maskProgressBar"),
  maskProgressMeta: document.querySelector("#maskProgressMeta"),
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
    subject_candidate: state.selection.subject_candidate || null,
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
  state.subjectCandidates.forEach((candidate, index) => {
    const candidateBox = candidate.box;
    if (!candidateBox) return;
    const isSelected = selection.subject_candidate?.id === candidate.id;
    const node = makeSvg("rect", {
      class: `subject-candidate-box ${isSelected ? "is-selected" : ""}`.trim(),
      x: candidateBox.x,
      y: candidateBox.y,
      width: candidateBox.width,
      height: candidateBox.height,
      "data-candidate-index": index,
    });
    const title = makeSvg("title");
    title.textContent = `Candidate ${index + 1} · score ${candidate.score}`;
    node.appendChild(title);
    node.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
    });
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      selectSubjectCandidate(candidate);
    });
    els.promptOverlay.appendChild(node);
  });
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
  const boxLabel = selection.subject_candidate ? "candidate" : selection.box ? "box" : "no box";
  const summary =
    selection.type === "unset" ? "Unset" : `${positiveCount}+ / ${negativeCount}- / ${boxLabel}`;
  els.selectionSummary.textContent = summary;
  els.promptState.textContent = selection.type === "unset" ? "Unset" : "Ready";
  els.promptJson.textContent = JSON.stringify(selection, null, 2);
  renderSamJobStatus();
}

function renderCandidateSummary() {
  const count = state.subjectCandidates.length;
  els.detectCandidatesButton.disabled = !state.current || state.subjectCandidatesStatus === "loading";
  if (state.subjectCandidatesStatus === "loading") {
    els.detectCandidatesButton.textContent = "Finding...";
    els.candidateSummary.textContent = "Scanning prompt frame with SAM 3.1";
    return;
  }
  els.detectCandidatesButton.textContent = count ? "Refresh Runners" : "Find Runners";
  if (state.subjectCandidatesStatus === "error") return;
  if (!count) {
    els.candidateSummary.textContent = "No candidates loaded";
    return;
  }
  const selected = normalizedSelection().subject_candidate;
  const selectedLabel = selected ? ` · selected ${selected.index + 1}` : "";
  els.candidateSummary.textContent = `${count} candidate${count === 1 ? "" : "s"} loaded${selectedLabel}`;
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

function backendLabel(backend = state.maskBackend) {
  if (backend === "sam31_mlx") return "SAM 3.1 MLX";
  return "SAM 2.1";
}

function qualityLabel(mode = state.maskQualityMode) {
  if (mode === "max") return "max";
  if (mode === "fast") return "224";
  return "1008";
}

function phaseLabel(phase = "") {
  if (phase === "loading_model") return "Loading model";
  if (phase === "detecting") return "Detecting runner masks";
  if (phase === "writing_outputs") return "Writing outputs";
  if (phase === "completed") return "Complete";
  if (phase === "failed") return "Failed";
  if (phase === "queued") return "Queued";
  return "Preparing mask run";
}

function progressPercent(progress = {}) {
  const percent = Number(progress.percent);
  if (Number.isFinite(percent)) return Math.max(0, Math.min(1, percent));
  const processed = Number(progress.processed_frames);
  const total = Number(progress.total_frames);
  if (Number.isFinite(processed) && Number.isFinite(total) && total > 0) {
    return Math.max(0, Math.min(1, processed / total));
  }
  return 0;
}

function renderMaskProgress(job = state.samJob) {
  const progress = job?.progress;
  const isRunning = job?.status === "running";
  if (!isRunning || !progress) {
    els.maskProgress.classList.add("is-hidden");
    els.maskProgressBar.style.width = "0%";
    return;
  }

  const processed = Number(progress.processed_frames) || 0;
  const total = Number(progress.total_frames) || 0;
  const percent = progressPercent(progress);
  const percentLabel = `${Math.round(percent * 100)}%`;
  const eta = progress.eta_seconds == null ? "--" : formatTime(progress.eta_seconds);
  const elapsed = progress.elapsed_seconds == null ? "--" : formatTime(progress.elapsed_seconds);
  const frame =
    progress.frame_index == null ? "frame --" : `frame ${Number(progress.frame_index) + 1}`;
  const detection =
    progress.detection_count == null ? "" : ` · ${progress.detection_count} detections`;
  const resolution = progress.resolution ? ` · ${progress.resolution}px` : "";

  els.maskProgress.classList.remove("is-hidden");
  els.maskProgressLabel.textContent = `${phaseLabel(progress.phase)} · ${percentLabel}`;
  els.maskProgressEta.textContent = `ETA ${eta}`;
  els.maskProgressBar.style.width = `${Math.round(percent * 100)}%`;
  els.maskProgressMeta.textContent =
    total > 0
      ? `${processed}/${total} frames · elapsed ${elapsed} · ${frame}${detection}${resolution}`
      : `elapsed ${elapsed}${resolution}`;
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
  if (status === "running" && state.samJob.backend) {
    const progress = state.samJob.progress;
    statusLabel = progress
      ? `${backendLabel(state.samJob.backend)} ${Math.round(progressPercent(progress) * 100)}%`
      : `${backendLabel(state.samJob.backend)} running`;
  }
  els.maskQualityLabel.classList.toggle("is-hidden", state.maskBackend !== "sam31_mlx");
  renderMaskProgress(state.samJob);

  els.samJobState.textContent = statusLabel;
  els.samJobState.className = `rank-pill sam-job-state ${
    status === "running" ? "is-running" : status === "completed" || artifactsReady ? "is-complete" : ""
  } ${status === "failed" ? "is-error" : ""}`.trim();

  const isRunning = status === "running";
  els.runSamButton.disabled = !state.current || !hasPrompt || isRunning;
  els.runSamButton.textContent = isRunning
    ? "Running..."
    : artifactsReady || status === "completed"
      ? `Run ${backendLabel()}${state.maskBackend === "sam31_mlx" ? ` ${qualityLabel()}` : ""} Again`
      : `Run ${backendLabel()}${state.maskBackend === "sam31_mlx" ? ` ${qualityLabel()}` : ""}`;
  els.runSamButton.title = hasPrompt
    ? `Run ${backendLabel()}${state.maskBackend === "sam31_mlx" ? ` ${qualityLabel()} mode` : ""} on the saved subject prompt`
    : "Select the runner first";
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
    subject_candidate: null,
    ...(run.prompt?.selection || {}),
  };
  state.subjectCandidates = [];
  state.subjectCandidatesStatus = "idle";

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
  renderCandidateSummary();
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
    await loadSubjectCandidateCache(candidateId);
  } catch (error) {
    setSaveState("Load failed", "is-error");
    els.runTitle.textContent = "Could not load CV run";
    els.runMeta.textContent = String(error);
  }
}

function addPromptPoint(event) {
  if (!state.current || state.promptMode === "box") return;
  if (event.target?.dataset?.candidateIndex != null) return;
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
    state.selection.subject_candidate = null;
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
  state.selection = {
    type: "unset",
    positive_points: [],
    negative_points: [],
    box: null,
    mask_path: null,
    subject_candidate: null,
  };
  state.draftBox = null;
  renderOverlay();
}

async function detectSubjectCandidates(options = {}) {
  const { force = true } = options;
  if (!state.current) return;
  const candidateId = state.current.candidate_id;
  try {
    state.subjectCandidatesStatus = "loading";
    renderCandidateSummary();
    setSaveState("Finding runners", "is-saving");
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/subject-candidates`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quality_mode: "native", force }),
    });
    if (state.current?.candidate_id !== candidateId) return;
    state.subjectCandidates = payload.subject_candidates?.candidates || [];
    state.subjectCandidatesStatus = "ready";
    renderOverlay();
    renderCandidateSummary();
    setSaveState(state.subjectCandidates.length ? "Candidates ready" : "No runners found");
  } catch (error) {
    state.subjectCandidatesStatus = "error";
    els.candidateSummary.textContent = String(error);
    renderCandidateSummary();
    setSaveState("Candidate scan failed", "is-error");
  }
}

async function loadSubjectCandidateCache(candidateId) {
  try {
    const payload = await fetchJson(
      `/api/cv-runs/${candidateId}/subject-candidates?quality_mode=native`
    );
    if (state.current?.candidate_id !== candidateId) return;
    const subjectCandidates = payload.subject_candidates;
    if (!subjectCandidates?.cached || !subjectCandidates.candidates?.length) return;
    state.subjectCandidates = subjectCandidates.candidates;
    state.subjectCandidatesStatus = "ready";
    renderOverlay();
    renderCandidateSummary();
  } catch {
    // Candidate cache is optional; keep the manual selector available if it is absent.
  }
}

async function selectSubjectCandidate(candidate) {
  if (!candidate?.box) return;
  state.selection = {
    type: "box",
    positive_points: [],
    negative_points: [],
    box: candidate.box,
    mask_path: null,
    subject_candidate: candidate,
  };
  renderOverlay();
  renderCandidateSummary();
  await savePrompt({ rerender: false, flash: true });
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
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/mask`);
    const job = payload.job || { status: "idle" };
    renderSamJobStatus(job);
    if (job.status === "running") {
      scheduleSamPoll();
    } else {
      clearSamPoll();
      if (reloadOnComplete && job.status === "completed" && state.current?.candidate_id === candidateId) {
        await loadRun(candidateId, { refreshJob: false });
        renderSamJobStatus(job);
        setSaveState(`${backendLabel(job.backend)} complete`);
      }
      if (job.status === "failed") {
        setSaveState(`${backendLabel(job.backend)} failed`, "is-error");
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
    setSaveState(`Starting ${backendLabel()}`, "is-saving");
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/mask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend: state.maskBackend, quality_mode: state.maskQualityMode }),
    });
    renderSamJobStatus(payload.job);
    setSaveState(`${backendLabel()} running`, "is-saving");
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
els.detectCandidatesButton.addEventListener("click", () => detectSubjectCandidates({ force: true }));

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
els.maskBackendSelect.addEventListener("change", () => {
  state.maskBackend = els.maskBackendSelect.value;
  renderSamJobStatus();
});
els.maskQualitySelect.addEventListener("change", () => {
  state.maskQualityMode = els.maskQualitySelect.value;
  renderSamJobStatus();
});
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
