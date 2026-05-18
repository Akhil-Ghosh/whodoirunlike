const state = {
  clips: [],
  currentIndex: 0,
  quality: "",
  viewMode: "full",
  saveTimer: 0,
};

const els = {
  saveState: document.querySelector("#saveState"),
  limitPill: document.querySelector("#limitPill"),
  reviewedCount: document.querySelector("#reviewedCount"),
  totalCount: document.querySelector("#totalCount"),
  goodCount: document.querySelector("#goodCount"),
  midCount: document.querySelector("#midCount"),
  badCount: document.querySelector("#badCount"),
  queueList: document.querySelector("#queueList"),
  bucketLabel: document.querySelector("#bucketLabel"),
  clipTitle: document.querySelector("#clipTitle"),
  clipMeta: document.querySelector("#clipMeta"),
  sourceLink: document.querySelector("#sourceLink"),
  videoPlayer: document.querySelector("#videoPlayer"),
  videoEmpty: document.querySelector("#videoEmpty"),
  currentTime: document.querySelector("#currentTime"),
  durationTime: document.querySelector("#durationTime"),
  seekRange: document.querySelector("#seekRange"),
  segmentLine: document.querySelector("#segmentLine"),
  segmentStatus: document.querySelector("#segmentStatus"),
  viewModeButtons: document.querySelectorAll("[data-view-mode]"),
  setStartButton: document.querySelector("#setStartButton"),
  setEndButton: document.querySelector("#setEndButton"),
  jumpStartButton: document.querySelector("#jumpStartButton"),
  jumpEndButton: document.querySelector("#jumpEndButton"),
  runnerName: document.querySelector("#runnerName"),
  rankPill: document.querySelector("#rankPill"),
  qualityButtons: document.querySelectorAll("[data-quality]"),
  angleSelect: document.querySelector("#angleSelect"),
  startInput: document.querySelector("#startInput"),
  endInput: document.querySelector("#endInput"),
  notesInput: document.querySelector("#notesInput"),
  metricStack: document.querySelector("#metricStack"),
  saveNextButton: document.querySelector("#saveNextButton"),
  prevButton: document.querySelector("#prevButton"),
  nextButton: document.querySelector("#nextButton"),
};

function formatTime(value) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function formatCount(value) {
  if (!Number.isFinite(value)) return "0";
  return new Intl.NumberFormat("en-US", { notation: "compact" }).format(value);
}

function clipDuration(clip) {
  return Number(clip?.duration_seconds_local || clip?.duration_seconds || 0);
}

function currentClip() {
  return state.clips[state.currentIndex];
}

function setSaveState(label, className = "") {
  window.clearTimeout(state.saveTimer);
  els.saveState.textContent = label;
  els.saveState.className = `save-state ${className}`.trim();
  if (label === "Saved") {
    state.saveTimer = window.setTimeout(() => setSaveState("Ready"), 1400);
  }
}

function qualityLabel(value) {
  return value && value !== "unreviewed" ? value : "open";
}

function renderSummary(payload) {
  const counts = payload.counts || {};
  els.reviewedCount.textContent = payload.reviewed || 0;
  els.totalCount.textContent = `/ ${payload.total || 0}`;
  els.limitPill.textContent = `${payload.total || 0} clips`;
  els.goodCount.textContent = counts.good || 0;
  els.midCount.textContent = counts.mid || 0;
  els.badCount.textContent = counts.bad || 0;
}

function renderQueue() {
  els.queueList.innerHTML = "";
  state.clips.forEach((clip, index) => {
    const quality = qualityLabel(clip.review_quality);
    const button = document.createElement("button");
    button.className = `queue-item ${index === state.currentIndex ? "is-active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <div class="queue-kicker">
        <span>${String(clip.rank).padStart(2, "0")} · ${clip.runner_name}</span>
        <span class="quality-dot ${quality}">${quality}</span>
      </div>
      <div class="queue-title">${clip.title}</div>
      <div class="queue-meta">
        <span>CV ${clip.cv_score}</span>
        <span>${formatTime(clipDuration(clip))}</span>
      </div>
    `;
    button.addEventListener("click", () => selectClip(index));
    els.queueList.appendChild(button);
  });
}

function metricRow(label, value) {
  return `<div class="metric-row"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderMetrics(clip) {
  const rows = [
    metricRow("CV score", clip.cv_score),
    metricRow("Camera", clip.camera_angle_proxy || "unknown"),
    metricRow("Pose hit", `${Math.round(Number(clip.pose_hit_rate || 0) * 100)}%`),
    metricRow("Full body", `${Math.round(Number(clip.full_body_rate || 0) * 100)}%`),
    metricRow("Runner size", `${Math.round(Number(clip.size_ok_rate || 0) * 100)}%`),
    metricRow(
      "Review file",
      clip.review_file_size_mb ? `${Number(clip.review_file_size_mb).toFixed(1)} MB` : "triage"
    ),
    metricRow("Views", formatCount(Number(clip.view_count || 0))),
  ];
  els.metricStack.innerHTML = rows.join("");
}

function annotationFor(clip) {
  return clip?.annotation || {};
}

function readDraft() {
  const clip = currentClip();
  return {
    candidate_id: clip.candidate_id,
    quality: state.quality,
    camera_angle: els.angleSelect.value || "unknown",
    start_seconds: els.startInput.value,
    end_seconds: els.endInput.value,
    notes: els.notesInput.value,
  };
}

function segmentBounds() {
  const clip = currentClip();
  const duration = els.videoPlayer.duration || clipDuration(clip) || 0;
  const start = Number(els.startInput.value);
  const end = Number(els.endInput.value);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return null;
  }
  const safeStart = Math.max(0, Math.min(start, duration || start));
  const safeEnd = duration ? Math.max(safeStart, Math.min(end, duration)) : end;
  if (safeEnd <= safeStart) {
    return null;
  }
  return { start: safeStart, end: safeEnd, duration };
}

function updateSegmentLine() {
  const clip = currentClip();
  const duration = clipDuration(clip) || els.videoPlayer.duration || 0;
  const start = Number(els.startInput.value || 0);
  const end = Number(els.endInput.value || duration);
  const startPct = duration ? Math.max(0, Math.min(100, (start / duration) * 100)) : 0;
  const endPct = duration ? Math.max(0, Math.min(100, (end / duration) * 100)) : 100;
  els.segmentLine.style.setProperty("--start-pct", `${Math.min(startPct, endPct)}%`);
  els.segmentLine.style.setProperty("--end-pct", `${Math.max(startPct, endPct)}%`);
  applyPlaybackWindow({ seek: false });
}

function setQuality(quality) {
  state.quality = quality;
  els.qualityButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.quality === quality);
  });
}

function setViewMode(viewMode, { seek = true } = {}) {
  state.viewMode = viewMode === "segment" ? "segment" : "full";
  applyPlaybackWindow({ seek });
}

function applyPlaybackWindow({ seek = false } = {}) {
  const clip = currentClip();
  if (!clip) return;

  const duration = els.videoPlayer.duration || clipDuration(clip) || 1;
  const bounds = segmentBounds();
  const canShowSegment = Boolean(bounds);
  if (state.viewMode === "segment" && !canShowSegment) {
    state.viewMode = "full";
  }

  els.viewModeButtons.forEach((button) => {
    const isSegmentButton = button.dataset.viewMode === "segment";
    button.classList.toggle("is-active", button.dataset.viewMode === state.viewMode);
    button.disabled = isSegmentButton && !canShowSegment;
  });

  if (state.viewMode === "segment" && bounds) {
    els.seekRange.min = bounds.start;
    els.seekRange.max = bounds.end;
    els.segmentStatus.textContent = `Viewing ${formatTime(bounds.start)}-${formatTime(bounds.end)}`;
    if (
      seek ||
      els.videoPlayer.currentTime < bounds.start ||
      els.videoPlayer.currentTime > bounds.end
    ) {
      els.videoPlayer.currentTime = bounds.start;
    }
  } else {
    els.seekRange.min = 0;
    els.seekRange.max = duration;
    els.segmentStatus.textContent = canShowSegment
      ? `Segment ${formatTime(bounds.start)}-${formatTime(bounds.end)} ready`
      : "Set start and end to preview a segment.";
  }

  els.seekRange.value = els.videoPlayer.currentTime || Number(els.seekRange.min || 0);
  els.durationTime.textContent =
    state.viewMode === "segment" && bounds ? formatTime(bounds.end) : formatTime(duration);
}

function selectClip(index) {
  state.currentIndex = Math.max(0, Math.min(index, state.clips.length - 1));
  const clip = currentClip();
  if (!clip) return;

  const annotation = annotationFor(clip);
  setQuality(annotation.quality || "");
  els.bucketLabel.textContent = clip.primary_bucket.replace("_", " / ");
  els.clipTitle.textContent = clip.title;
  els.clipMeta.textContent = `${clip.runner_name} · ${clip.channel} · ${formatTime(clipDuration(clip))}`;
  els.runnerName.textContent = clip.runner_name;
  els.rankPill.textContent = `#${clip.rank}`;
  els.sourceLink.href = clip.url || "#";
  els.angleSelect.value = annotation.camera_angle || "unknown";
  els.startInput.value = annotation.start_seconds ?? "";
  els.endInput.value = annotation.end_seconds ?? "";
  els.notesInput.value = annotation.notes || "";
  els.videoPlayer.src = clip.video_url;
  els.videoEmpty.classList.add("is-hidden");
  els.seekRange.value = 0;
  els.seekRange.max = clipDuration(clip) || 1;
  els.currentTime.textContent = "0:00";
  els.durationTime.textContent = formatTime(clipDuration(clip));
  renderMetrics(clip);
  renderQueue();
  updateSegmentLine();
  applyPlaybackWindow({ seek: false });
}

async function saveCurrent({ advance = false } = {}) {
  const clip = currentClip();
  if (!clip) return;
  setSaveState("Saving", "is-saving");
  const response = await fetch("/api/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readDraft()),
  });
  const payload = await response.json();
  if (!response.ok) {
    setSaveState(payload.error || "Save failed", "is-error");
    return;
  }

  const currentId = clip.candidate_id;
  state.clips = payload.clips;
  renderSummary(payload);
  const nextIndex = state.clips.findIndex((item) => item.candidate_id === currentId);
  state.currentIndex = nextIndex >= 0 ? nextIndex : state.currentIndex;
  setSaveState("Saved");
  if (advance) {
    selectClip(Math.min(state.currentIndex + 1, state.clips.length - 1));
  } else {
    selectClip(state.currentIndex);
  }
}

function jumpToInput(input) {
  const value = Number(input.value);
  if (Number.isFinite(value)) {
    els.videoPlayer.currentTime = value;
  }
}

async function init() {
  try {
    const response = await fetch("/api/clips");
    const payload = await response.json();
    state.clips = payload.clips || [];
    renderSummary(payload);
    const firstOpen = state.clips.findIndex((clip) => clip.review_quality === "unreviewed");
    selectClip(firstOpen >= 0 ? firstOpen : 0);
  } catch (error) {
    setSaveState("Load failed", "is-error");
    els.clipTitle.textContent = "Could not load review queue";
    els.clipMeta.textContent = String(error);
  }
}

els.videoPlayer.addEventListener("loadedmetadata", () => {
  const duration = els.videoPlayer.duration || clipDuration(currentClip()) || 1;
  els.seekRange.max = duration;
  els.durationTime.textContent = formatTime(duration);
  updateSegmentLine();
  applyPlaybackWindow({ seek: state.viewMode === "segment" });
});

els.videoPlayer.addEventListener("timeupdate", () => {
  const bounds = state.viewMode === "segment" ? segmentBounds() : null;
  if (bounds && els.videoPlayer.currentTime >= bounds.end) {
    if (!els.videoPlayer.paused) {
      els.videoPlayer.currentTime = bounds.start;
      els.videoPlayer.play().catch(() => {});
    }
  }
  els.seekRange.value = els.videoPlayer.currentTime || 0;
  els.currentTime.textContent = formatTime(els.videoPlayer.currentTime || 0);
});

els.seekRange.addEventListener("input", () => {
  els.videoPlayer.currentTime = Number(els.seekRange.value || 0);
});

els.setStartButton.addEventListener("click", () => {
  els.startInput.value = (els.videoPlayer.currentTime || 0).toFixed(2);
  updateSegmentLine();
  applyPlaybackWindow({ seek: false });
});

els.setEndButton.addEventListener("click", () => {
  els.endInput.value = (els.videoPlayer.currentTime || 0).toFixed(2);
  updateSegmentLine();
  applyPlaybackWindow({ seek: false });
});

els.jumpStartButton.addEventListener("click", () => jumpToInput(els.startInput));
els.jumpEndButton.addEventListener("click", () => jumpToInput(els.endInput));

els.viewModeButtons.forEach((button) => {
  button.addEventListener("click", () => setViewMode(button.dataset.viewMode));
});

els.qualityButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setQuality(button.dataset.quality);
    saveCurrent();
  });
});

[els.startInput, els.endInput].forEach((input) => {
  input.addEventListener("input", () => {
    updateSegmentLine();
    applyPlaybackWindow({ seek: false });
  });
});

els.angleSelect.addEventListener("change", () => saveCurrent());

els.saveNextButton.addEventListener("click", () => saveCurrent({ advance: true }));
els.prevButton.addEventListener("click", () => selectClip(state.currentIndex - 1));
els.nextButton.addEventListener("click", () => selectClip(state.currentIndex + 1));

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) return;
  if (event.key === "1") setQuality("good");
  if (event.key === "2") setQuality("mid");
  if (event.key === "3") setQuality("bad");
  if (event.key === "[") els.setStartButton.click();
  if (event.key === "]") els.setEndButton.click();
  if (event.key === "s") saveCurrent();
});

init();
