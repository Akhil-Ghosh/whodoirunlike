const state = {
  runs: [],
  current: null,
  samJob: null,
  poseJob: null,
  denseposeJob: null,
  fusionJob: null,
  featuresJob: null,
  openposeJob: null,
  pipelineJob: null,
  denseposeSetup: null,
  openposeSetup: null,
  mmposeSetup: null,
  prepCandidates: [],
  prepStatus: "idle",
  maskBackend: "sam31_mlx",
  maskQualityMode: "native",
  poseBackend: "openpose",
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
  posePollTimer: 0,
  denseposePollTimer: 0,
  fusionPollTimer: 0,
  featuresPollTimer: 0,
  openposePollTimer: 0,
  pipelinePollTimer: 0,
};

const svgNS = "http://www.w3.org/2000/svg";

const els = {
  saveState: document.querySelector("#saveState"),
  runCount: document.querySelector("#runCount"),
  runList: document.querySelector("#runList"),
  prepCount: document.querySelector("#prepCount"),
  prepList: document.querySelector("#prepList"),
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
  poseJobState: document.querySelector("#poseJobState"),
  denseposeJobState: document.querySelector("#denseposeJobState"),
  fusionJobState: document.querySelector("#fusionJobState"),
  featuresJobState: document.querySelector("#featuresJobState"),
  openposeJobState: document.querySelector("#openposeJobState"),
  pipelineJobState: document.querySelector("#pipelineJobState"),
  poseBackendSelect: document.querySelector("#poseBackendSelect"),
  maskQualityLabel: document.querySelector("#maskQualityLabel"),
  maskQualitySelect: document.querySelector("#maskQualitySelect"),
  runPipelineButton: document.querySelector("#runPipelineButton"),
  runSamButton: document.querySelector("#runSamButton"),
  runPoseButton: document.querySelector("#runPoseButton"),
  runDenseposeButton: document.querySelector("#runDenseposeButton"),
  runFusionButton: document.querySelector("#runFusionButton"),
  runFeaturesButton: document.querySelector("#runFeaturesButton"),
  runOpenposeButton: document.querySelector("#runOpenposeButton"),
  maskProgress: document.querySelector("#maskProgress"),
  maskProgressLabel: document.querySelector("#maskProgressLabel"),
  maskProgressEta: document.querySelector("#maskProgressEta"),
  maskProgressBar: document.querySelector("#maskProgressBar"),
  maskProgressMeta: document.querySelector("#maskProgressMeta"),
  poseProgress: document.querySelector("#poseProgress"),
  poseProgressLabel: document.querySelector("#poseProgressLabel"),
  poseProgressEta: document.querySelector("#poseProgressEta"),
  poseProgressBar: document.querySelector("#poseProgressBar"),
  poseProgressMeta: document.querySelector("#poseProgressMeta"),
  denseposeProgress: document.querySelector("#denseposeProgress"),
  denseposeProgressLabel: document.querySelector("#denseposeProgressLabel"),
  denseposeProgressEta: document.querySelector("#denseposeProgressEta"),
  denseposeProgressBar: document.querySelector("#denseposeProgressBar"),
  denseposeProgressMeta: document.querySelector("#denseposeProgressMeta"),
  fusionProgress: document.querySelector("#fusionProgress"),
  fusionProgressLabel: document.querySelector("#fusionProgressLabel"),
  fusionProgressEta: document.querySelector("#fusionProgressEta"),
  fusionProgressBar: document.querySelector("#fusionProgressBar"),
  fusionProgressMeta: document.querySelector("#fusionProgressMeta"),
  featuresProgress: document.querySelector("#featuresProgress"),
  featuresProgressLabel: document.querySelector("#featuresProgressLabel"),
  featuresProgressEta: document.querySelector("#featuresProgressEta"),
  featuresProgressBar: document.querySelector("#featuresProgressBar"),
  featuresProgressMeta: document.querySelector("#featuresProgressMeta"),
  openposeProgress: document.querySelector("#openposeProgress"),
  openposeProgressLabel: document.querySelector("#openposeProgressLabel"),
  openposeProgressEta: document.querySelector("#openposeProgressEta"),
  openposeProgressBar: document.querySelector("#openposeProgressBar"),
  openposeProgressMeta: document.querySelector("#openposeProgressMeta"),
  pipelineProgress: document.querySelector("#pipelineProgress"),
  pipelineProgressLabel: document.querySelector("#pipelineProgressLabel"),
  pipelineProgressEta: document.querySelector("#pipelineProgressEta"),
  pipelineProgressBar: document.querySelector("#pipelineProgressBar"),
  pipelineProgressMeta: document.querySelector("#pipelineProgressMeta"),
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

function clearCvPoll(kind) {
  const timerKey = `${kind}PollTimer`;
  window.clearTimeout(state[timerKey]);
  state[timerKey] = 0;
}

function clearPipelinePoll() {
  window.clearTimeout(state.pipelinePollTimer);
  state.pipelinePollTimer = 0;
}

function clearAllCvPolls() {
  clearSamPoll();
  clearCvPoll("pose");
  clearCvPoll("densepose");
  clearCvPoll("fusion");
  clearCvPoll("features");
  clearCvPoll("openpose");
  clearPipelinePoll();
}

function scheduleSamPoll() {
  clearSamPoll();
  state.samPollTimer = window.setTimeout(() => {
    refreshSamJobStatus({ reloadOnComplete: true });
  }, 2000);
}

function scheduleCvPoll(kind) {
  clearCvPoll(kind);
  state[`${kind}PollTimer`] = window.setTimeout(() => {
    refreshCvJobStatus(kind, { reloadOnComplete: true });
  }, 2000);
}

function schedulePipelinePoll() {
  clearPipelinePoll();
  state.pipelinePollTimer = window.setTimeout(() => {
    refreshPipelineJobStatus({ reloadOnComplete: true });
  }, 2000);
}

function samStatusLabel(status) {
  if (status === "completed") return "Complete";
  if (status === "failed") return "Failed";
  if (status === "running") return "Running";
  if (status === "unavailable") return "Unavailable";
  return "Idle";
}

function backendLabel(backend = state.maskBackend) {
  if (backend === "sam31_mlx") return "SAM 3.1 MLX";
  return "SAM 2.1";
}

function poseBackendLabel(backend = state.poseBackend) {
  if (backend === "mediapipe") return "MediaPipe";
  if (backend === "mmpose_rtmw_l_384") return "RTMW-L 384";
  if (backend === "mmpose_rtmw_l_256") return "RTMW-L 256";
  if (backend === "mmpose_rtmw_m_256") return "RTMW-M 256";
  if (backend === "mmpose_rtmpose_l_384") return "RTMPose-L";
  return "OpenPose";
}

function qualityLabel(mode = state.maskQualityMode) {
  if (mode === "max") return "max";
  if (mode === "fast") return "224";
  return "1008";
}

function phaseLabel(phase = "") {
  if (phase === "loading_model") return "Loading model";
  if (phase === "detecting") return "Detecting runner masks";
  if (phase === "detecting_pose") return "Estimating pose";
  if (phase === "estimating_pose") return "Estimating pose";
  if (phase === "running_pose") return "Estimating pose";
  if (phase === "loading_mmpose_model") return "Loading RTMW";
  if (phase === "loading_rtmw_model") return "Loading RTMW";
  if (phase === "running_mmpose") return "Running RTMW";
  if (phase === "running_rtmw") return "Running RTMW";
  if (phase === "running_densepose") return "Running DensePose";
  if (phase === "fusing_form") return "Fusing form";
  if (phase === "reading_inputs") return "Reading inputs";
  if (phase === "compiling_features") return "Compiling features";
  if (phase === "summarizing_features") return "Summarizing features";
  if (phase === "preparing_openpose_frames") return "Preparing OpenPose frames";
  if (phase === "running_openpose") return "Running OpenPose";
  if (phase === "reading_openpose_results") return "Reading OpenPose";
  if (phase === "starting") return "Starting pipeline";
  if (phase === "running") return "Running pipeline";
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

function artifactExists(key) {
  return Boolean(state.current?.artifacts?.[key]?.exists);
}

function hasDenseposeInputs() {
  return artifactExists("runner_mask") || artifactExists("qa_overlay");
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

function renderCvProgress(kind, job) {
  const progress = job?.progress;
  const isRunning = job?.status === "running";
  const progressEl = els[`${kind}Progress`];
  const barEl = els[`${kind}ProgressBar`];
  if (!isRunning || !progress) {
    progressEl.classList.add("is-hidden");
    barEl.style.width = "0%";
    return;
  }

  const processed = Number(progress.processed_frames) || 0;
  const total = Number(progress.total_frames) || 0;
  const percent = progressPercent(progress);
  const percentLabel = `${Math.round(percent * 100)}%`;
  const eta = progress.eta_seconds == null ? "--" : formatTime(progress.eta_seconds);
  const elapsed = progress.elapsed_seconds == null ? "--" : formatTime(progress.elapsed_seconds);
  const phase = phaseLabel(progress.phase);

  progressEl.classList.remove("is-hidden");
  els[`${kind}ProgressLabel`].textContent = `${phase} · ${percentLabel}`;
  els[`${kind}ProgressEta`].textContent = `ETA ${eta}`;
  barEl.style.width = `${Math.round(percent * 100)}%`;
  els[`${kind}ProgressMeta`].textContent =
    total > 0 ? `${processed}/${total} frames · elapsed ${elapsed}` : `elapsed ${elapsed}`;
}

function renderPipelineProgress(job = state.pipelineJob) {
  const progress = job?.progress;
  const isRunning = job?.status === "running";
  if (!isRunning || !progress) {
    els.pipelineProgress.classList.add("is-hidden");
    els.pipelineProgressBar.style.width = "0%";
    return;
  }
  const percent = progressPercent(progress);
  const stage = progress.stage || "mask";
  const stageProgress = progress.stage_progress || {};
  const stagePercent =
    stageProgress.percent == null ? "" : ` · ${Math.round(progressPercent(stageProgress) * 100)}% stage`;
  const stepIndex = Number(progress.step_index || 0) + 1;
  const stepCount = Number(progress.step_count || 5);
  els.pipelineProgress.classList.remove("is-hidden");
  els.pipelineProgressLabel.textContent = `${phaseLabel(progress.phase)} · ${Math.round(percent * 100)}%`;
  els.pipelineProgressEta.textContent = `${stepIndex}/${stepCount}`;
  els.pipelineProgressBar.style.width = `${Math.round(percent * 100)}%`;
  els.pipelineProgressMeta.textContent = `${stage.replaceAll("_", " ")}${stagePercent}`;
}

function renderPipelineJobStatus(job = state.pipelineJob) {
  state.pipelineJob = job || { status: "idle" };
  const status = state.pipelineJob.status || "idle";
  const selection = normalizedSelection();
  const hasPrompt = selection.type !== "unset";
  const isRunning = status === "running";
  const isOpenPose = state.poseBackend === "openpose";
  const isMMPose = state.poseBackend.startsWith("mmpose_");
  const selectedMmposeSetup = isMMPose ? state.mmposeSetup?.[state.poseBackend] : null;
  const setupWarning =
    isOpenPose && state.openposeSetup && !state.openposeSetup.ready
      ? state.openposeSetup.reasons?.[0] || "OpenPose is not configured"
      : isMMPose && selectedMmposeSetup && !selectedMmposeSetup.ready
        ? selectedMmposeSetup.reasons?.[0] || "RTMW is not configured"
        : "";
  const label =
    isRunning && state.pipelineJob.progress
      ? `Pipeline ${Math.round(progressPercent(state.pipelineJob.progress) * 100)}%`
      : `Pipeline ${samStatusLabel(status).toLowerCase()}`;

  renderPipelineProgress(state.pipelineJob);
  els.pipelineJobState.textContent = label;
  els.pipelineJobState.className = `rank-pill cv-job-state ${
    isRunning ? "is-running" : status === "completed" ? "is-complete" : ""
  } ${status === "failed" ? "is-error" : ""}`.trim();
  els.runPipelineButton.disabled = !state.current || !hasPrompt || isRunning || Boolean(setupWarning);
  els.runPipelineButton.textContent = isRunning
    ? "Running Pipeline..."
    : `Run Full Pipeline`;
  els.runPipelineButton.title = setupWarning
    ? setupWarning
    : hasPrompt
      ? `Run SAM 3.1, ${poseBackendLabel()}, DensePose, Fusion, and Features`
    : "Select and save the runner first";
}

function renderSamJobStatus(job = state.samJob) {
  state.samJob = job || { status: "idle" };
  const status = state.samJob.status || "idle";
  const selection = normalizedSelection();
  const hasPrompt = selection.type !== "unset";
  const artifactsReady = hasDenseposeInputs();

  let statusLabel = samStatusLabel(status);
  if (status === "idle" && artifactsReady) statusLabel = "Artifacts ready";
  if (status === "running" && state.samJob.backend) {
    const progress = state.samJob.progress;
    statusLabel = progress
      ? `${backendLabel(state.samJob.backend)} ${Math.round(progressPercent(progress) * 100)}%`
      : `${backendLabel(state.samJob.backend)} running`;
  }
  els.maskQualityLabel.classList.remove("is-hidden");
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
      ? `Run ${backendLabel()} ${qualityLabel()} Again`
      : `Run ${backendLabel()} ${qualityLabel()}`;
  els.runSamButton.title = hasPrompt
    ? `Run ${backendLabel()} ${qualityLabel()} mode on the saved subject prompt`
    : "Select the runner first";
}

function cvJobConfig(kind) {
  if (kind === "densepose") {
    const setupReady = state.denseposeSetup ? Boolean(state.denseposeSetup.ready) : true;
    const setupReason = state.denseposeSetup?.reasons?.[0] || "DensePose is not configured";
    return {
      label: "DensePose",
      stateKey: "denseposeJob",
      stateEl: els.denseposeJobState,
      buttonEl: els.runDenseposeButton,
      ready: artifactExists("densepose"),
      canRun: Boolean(state.current) && hasDenseposeInputs() && setupReady,
      blockedTitle: !hasDenseposeInputs() ? "Run mask first" : setupReason,
    };
  }
  if (kind === "fusion") {
    const hasPose = artifactExists("pose_landmarks");
    const hasDensepose = artifactExists("densepose");
    return {
      label: "Fusion",
      stateKey: "fusionJob",
      stateEl: els.fusionJobState,
      buttonEl: els.runFusionButton,
      ready: artifactExists("fused_form") || artifactExists("fused_overlay"),
      canRun: Boolean(state.current) && hasPose && hasDensepose,
      blockedTitle: !hasPose ? "Run Pose first" : !hasDensepose ? "Run DensePose first" : "Load a CV run first",
    };
  }
  if (kind === "features") {
    const hasPose = artifactExists("pose_landmarks");
    const hasFused = artifactExists("fused_form");
    return {
      label: "Features",
      stateKey: "featuresJob",
      stateEl: els.featuresJobState,
      buttonEl: els.runFeaturesButton,
      ready: artifactExists("form_features") || artifactExists("form_feature_arrays"),
      canRun: Boolean(state.current) && hasPose && hasFused,
      blockedTitle: !hasPose ? "Run Pose first" : !hasFused ? "Run Fusion first" : "Load a CV run first",
    };
  }
  if (kind === "openpose") {
    const hasPose = artifactExists("pose_landmarks");
    const hasMask = artifactExists("runner_mask");
    const setupReady = state.openposeSetup ? Boolean(state.openposeSetup.ready) : true;
    return {
      label: "OpenPose",
      stateKey: "openposeJob",
      stateEl: els.openposeJobState,
      buttonEl: els.runOpenposeButton,
      ready: artifactExists("openpose_landmarks") || artifactExists("pose_comparison"),
      canRun: Boolean(state.current) && hasPose && hasMask,
      blockedTitle: !hasPose ? "Run Pose first" : !hasMask ? "Run mask first" : "Load a CV run first",
      setupWarning: setupReady
        ? ""
        : state.openposeSetup?.reasons?.[0] || "OpenPose is not configured",
    };
  }
  const isOpenPose = state.poseBackend === "openpose";
  const isMMPose = state.poseBackend.startsWith("mmpose_");
  const hasMask = artifactExists("runner_mask");
  const openposeReady = state.openposeSetup ? Boolean(state.openposeSetup.ready) : true;
  const selectedMmposeSetup = isMMPose ? state.mmposeSetup?.[state.poseBackend] : null;
  const mmposeReady = selectedMmposeSetup ? Boolean(selectedMmposeSetup.ready) : true;
  const setupReady = isOpenPose ? openposeReady : isMMPose ? mmposeReady : true;
  const setupReason = isOpenPose
    ? state.openposeSetup?.reasons?.[0] || "OpenPose is not configured"
    : selectedMmposeSetup?.reasons?.[0] || "RTMW is not configured";
  const needsMask = isOpenPose || isMMPose;
  return {
    label: poseBackendLabel(),
    stateKey: "poseJob",
    stateEl: els.poseJobState,
    buttonEl: els.runPoseButton,
    ready: artifactExists("pose_landmarks") || artifactExists("skeleton_render"),
    canRun: Boolean(state.current) && (!needsMask || hasMask) && setupReady,
    blockedTitle:
      needsMask && !hasMask
        ? "Run SAM 3.1 first"
        : !setupReady
          ? setupReason
          : "Load a CV run first",
    setupWarning: !setupReady ? setupReason : "",
  };
}

function renderCvJobStatus(kind, job = state[cvJobConfig(kind).stateKey]) {
  const config = cvJobConfig(kind);
  state[config.stateKey] = job || { status: "idle" };
  const currentJob = state[config.stateKey];
  const status = currentJob.status || "idle";
  const isRunning = status === "running";
  const setupBlocked = kind === "densepose" && state.denseposeSetup && !state.denseposeSetup.ready;
  const setupWarning = config.setupWarning || "";
  const percent = currentJob.progress ? Math.round(progressPercent(currentJob.progress) * 100) : null;
  let label = `${config.label} ${samStatusLabel(status).toLowerCase()}`;

  if (setupBlocked) label = `${config.label} setup`;
  if (!setupBlocked && setupWarning && status === "idle" && !config.ready) label = `${config.label} setup`;
  if (status === "idle" && config.ready) label = `${config.label} ready`;
  if (isRunning) label = percent == null ? `${config.label} running` : `${config.label} ${percent}%`;

  renderCvProgress(kind, currentJob);
  config.stateEl.textContent = label;
  config.stateEl.className = `rank-pill cv-job-state ${
    isRunning ? "is-running" : status === "completed" || config.ready ? "is-complete" : ""
  } ${status === "failed" || status === "unavailable" || setupBlocked || setupWarning ? "is-error" : ""}`.trim();

  config.buttonEl.disabled = !config.canRun || isRunning;
  config.buttonEl.textContent = isRunning
    ? "Running..."
    : config.ready || status === "completed"
      ? `Run ${config.label} Again`
      : `Run ${config.label}`;
  config.buttonEl.title = config.canRun
    ? setupWarning || `Run ${config.label} on the current CV run`
    : config.blockedTitle;
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

function renderPrepList() {
  els.prepList.innerHTML = "";
  if (state.prepStatus === "loading") {
    els.prepCount.textContent = "Loading";
    els.prepList.innerHTML = `<div class="artifact-empty">Loading reviewed clips</div>`;
    return;
  }
  els.prepCount.textContent = `${state.prepCandidates.length} clip${
    state.prepCandidates.length === 1 ? "" : "s"
  }`;
  state.prepCandidates.forEach((clip) => {
    const prepared = Boolean(clip.cv_run_prepared);
    const canPrepare = Boolean(clip.can_prepare);
    const button = document.createElement("button");
    button.className = `prep-item ${prepared ? "is-prepared" : ""}`.trim();
    button.type = "button";
    button.disabled = !prepared && !canPrepare;
    const quality = clip.review_quality || "unreviewed";
    button.innerHTML = `
      <div class="prep-kicker">
        <span>${clip.runner_name || "Unknown runner"}</span>
        <span class="quality-dot ${quality}">${prepared ? "ready" : quality}</span>
      </div>
      <div class="prep-title">${clip.title || "Untitled clip"}</div>
      <div class="prep-meta">
        <span>${clip.camera_angle || "unknown"}</span>
        <span>${prepared ? "Open" : canPrepare ? "Prepare" : "Needs review"}</span>
      </div>
    `;
    button.addEventListener("click", () => {
      if (prepared) {
        loadRun(clip.candidate_id);
      } else if (canPrepare) {
        prepareCvRun(clip.candidate_id);
      }
    });
    els.prepList.appendChild(button);
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
    "fused_form",
    "skeleton_render",
    "masked_runner",
    "qa_overlay",
    "fused_overlay",
    "features",
    "form_features",
    "form_feature_arrays",
    "mmpose_landmarks",
    "openpose_landmarks",
    "openpose_skeleton_render",
    "openpose_qa_overlay",
    "pose_comparison",
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
  els.poseBackendSelect.value = state.poseBackend;
  els.maskQualitySelect.value = state.maskQualityMode;

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
  renderCvJobStatus("pose");
  renderCvJobStatus("densepose");
  renderCvJobStatus("fusion");
  renderCvJobStatus("features");
  renderCvJobStatus("openpose");
  renderPipelineJobStatus();
  setSaveState("Ready");
}

async function loadRun(candidateId, options = {}) {
  const { refreshJob = true, preservePolls = false } = options;
  try {
    if (!preservePolls) clearAllCvPolls();
    setSaveState("Loading", "is-saving");
    if (!preservePolls) {
      state.samJob = null;
      state.poseJob = null;
      state.denseposeJob = null;
      state.fusionJob = null;
      state.featuresJob = null;
      state.openposeJob = null;
      state.pipelineJob = null;
    }
    state.current = await fetchJson(`/api/cv-runs/${candidateId}`);
    renderRun();
    if (refreshJob) {
      await refreshSamJobStatus();
      await refreshCvJobStatus("pose");
      await refreshCvJobStatus("densepose");
      await refreshCvJobStatus("fusion");
      await refreshCvJobStatus("features");
      await refreshCvJobStatus("openpose");
      await refreshPipelineJobStatus();
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
        await loadRun(candidateId, { refreshJob: false, preservePolls: true });
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

async function refreshCvJobStatus(kind, options = {}) {
  const { reloadOnComplete = false } = options;
  if (!state.current) return null;
  const candidateId = state.current.candidate_id;
  const config = cvJobConfig(kind);
  try {
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/${kind}`);
    const job = payload.job || { status: "idle" };
    if (kind === "densepose" && payload.setup) {
      state.denseposeSetup = payload.setup;
    }
    if (kind === "pose" && payload.openpose_setup) {
      state.openposeSetup = payload.openpose_setup;
    }
    if (kind === "pose" && payload.mmpose_setup) {
      state.mmposeSetup = payload.mmpose_setup;
    }
    if (kind === "openpose" && payload.setup) {
      state.openposeSetup = payload.setup;
    }
    renderCvJobStatus(kind, job);
    if (job.status === "running") {
      scheduleCvPoll(kind);
    } else {
      clearCvPoll(kind);
      if (
        reloadOnComplete &&
        (job.status === "completed" || job.status === "unavailable") &&
        state.current?.candidate_id === candidateId
      ) {
        await loadRun(candidateId, { refreshJob: false, preservePolls: true });
        renderCvJobStatus(kind, job);
        setSaveState(job.status === "unavailable" ? `${config.label} unavailable` : `${config.label} complete`);
      }
      if (job.status === "failed") {
        setSaveState(`${config.label} failed`, "is-error");
      }
    }
    return job;
  } catch (error) {
    clearCvPoll(kind);
    config.stateEl.textContent = `${config.label} unavailable`;
    config.stateEl.className = "rank-pill cv-job-state is-error";
    els.runMeta.textContent = String(error);
    return null;
  }
}

async function refreshPipelineJobStatus(options = {}) {
  const { reloadOnComplete = false } = options;
  if (!state.current) return null;
  const candidateId = state.current.candidate_id;
  try {
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/pipeline`);
    const job = payload.job || { status: "idle" };
    if (payload.openpose_setup) state.openposeSetup = payload.openpose_setup;
    if (payload.mmpose_setup) state.mmposeSetup = payload.mmpose_setup;
    renderPipelineJobStatus(job);
    if (payload.jobs?.mask) renderSamJobStatus(payload.jobs.mask);
    if (payload.jobs?.pose) renderCvJobStatus("pose", payload.jobs.pose);
    if (payload.jobs?.densepose) renderCvJobStatus("densepose", payload.jobs.densepose);
    if (payload.jobs?.fusion) renderCvJobStatus("fusion", payload.jobs.fusion);
    if (payload.jobs?.features) renderCvJobStatus("features", payload.jobs.features);
    if (job.status === "running") {
      schedulePipelinePoll();
    } else {
      clearPipelinePoll();
      if (reloadOnComplete && job.status === "completed" && state.current?.candidate_id === candidateId) {
        await loadRun(candidateId, { refreshJob: false, preservePolls: true });
        renderPipelineJobStatus(job);
        setSaveState("Pipeline complete");
      }
      if (job.status === "failed") {
        setSaveState("Pipeline failed", "is-error");
      }
    }
    return job;
  } catch (error) {
    clearPipelinePoll();
    els.pipelineJobState.textContent = "Pipeline unavailable";
    els.pipelineJobState.className = "rank-pill cv-job-state is-error";
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

async function startPipelineRun() {
  if (!state.current) return;
  if (normalizedSelection().type === "unset") {
    setSaveState("Select target first", "is-error");
    renderPipelineJobStatus();
    return;
  }
  const candidateId = state.current.candidate_id;
  const savedRun = await savePrompt({ rerender: false, flash: false });
  if (!savedRun || state.current?.candidate_id !== candidateId) return;

  try {
    setSaveState("Starting pipeline", "is-saving");
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/pipeline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pose_backend: state.poseBackend,
        mask_quality_mode: state.maskQualityMode,
      }),
    });
    if (payload.openpose_setup) state.openposeSetup = payload.openpose_setup;
    if (payload.mmpose_setup) state.mmposeSetup = payload.mmpose_setup;
    renderPipelineJobStatus(payload.job);
    setSaveState("Pipeline running", "is-saving");
    schedulePipelinePoll();
  } catch (error) {
    setSaveState("Start failed", "is-error");
    els.runMeta.textContent = String(error);
    await refreshPipelineJobStatus();
  }
}

async function startCvRun(kind) {
  if (!state.current) return;
  const candidateId = state.current.candidate_id;
  const config = cvJobConfig(kind);
  if (!config.canRun) {
    setSaveState(config.blockedTitle, "is-error");
    renderCvJobStatus(kind);
    return;
  }

  try {
    setSaveState(`Starting ${config.label}`, "is-saving");
    const payload = await fetchJson(`/api/cv-runs/${candidateId}/${kind}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(kind === "pose" ? { backend: state.poseBackend } : {}),
    });
    if (kind === "densepose" && payload.setup) {
      state.denseposeSetup = payload.setup;
    }
    if (kind === "pose" && payload.openpose_setup) {
      state.openposeSetup = payload.openpose_setup;
    }
    if (kind === "pose" && payload.mmpose_setup) {
      state.mmposeSetup = payload.mmpose_setup;
    }
    if (kind === "openpose" && payload.setup) {
      state.openposeSetup = payload.setup;
    }
    renderCvJobStatus(kind, payload.job);
    setSaveState(`${config.label} running`, "is-saving");
    scheduleCvPoll(kind);
  } catch (error) {
    setSaveState("Start failed", "is-error");
    els.runMeta.textContent = String(error);
    await refreshCvJobStatus(kind);
  }
}

async function loadRuns() {
  const payload = await fetchJson("/api/cv-runs");
  state.runs = payload.runs || [];
  renderRunList();
}

async function loadPrepCandidates() {
  try {
    state.prepStatus = "loading";
    renderPrepList();
    const payload = await fetchJson("/api/cv-run-candidates");
    state.prepCandidates = payload.clips || [];
    state.prepStatus = "ready";
    renderPrepList();
  } catch (error) {
    state.prepStatus = "error";
    els.prepCount.textContent = "Unavailable";
    els.prepList.innerHTML = `<div class="artifact-empty">${String(error)}</div>`;
  }
}

async function prepareCvRun(candidateId) {
  try {
    setSaveState("Preparing clip", "is-saving");
    await fetchJson(`/api/cv-runs/${candidateId}/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: false }),
    });
    await loadRuns();
    await loadPrepCandidates();
    await loadRun(candidateId);
    setSaveState("CV run ready");
  } catch (error) {
    setSaveState("Prepare failed", "is-error");
    els.runMeta.textContent = String(error);
  }
}

async function init() {
  try {
    await Promise.all([loadRuns(), loadPrepCandidates()]);
    if (state.runs.length) {
      await loadRun(state.runs[0].candidate_id);
    } else {
      setSaveState("No prepared runs");
      els.runTitle.textContent = "No prepared CV runs";
      els.runMeta.textContent = "Pick a reviewed clip from the left rail to prepare one.";
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
els.poseBackendSelect.addEventListener("change", () => {
  state.poseBackend = els.poseBackendSelect.value;
  renderCvJobStatus("pose");
  renderPipelineJobStatus();
});
els.maskQualitySelect.addEventListener("change", () => {
  state.maskQualityMode = els.maskQualitySelect.value;
  renderSamJobStatus();
  renderPipelineJobStatus();
});
els.runPipelineButton.addEventListener("click", startPipelineRun);
els.runSamButton.addEventListener("click", startSamRun);
els.runPoseButton.addEventListener("click", () => startCvRun("pose"));
els.runDenseposeButton.addEventListener("click", () => startCvRun("densepose"));
els.runFusionButton.addEventListener("click", () => startCvRun("fusion"));
els.runFeaturesButton.addEventListener("click", () => startCvRun("features"));
els.runOpenposeButton.addEventListener("click", () => startCvRun("openpose"));
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
