import {
  ArrowsClockwise,
  CalendarBlank,
  CaretDown,
  CheckCircle,
  ClockCountdown,
  Cpu,
  Gauge,
  ListMagnifyingGlass,
  SquaresFour,
  Stack,
  Warning,
  XCircle,
} from "@phosphor-icons/react";
import { useEffect, useMemo, useState } from "react";
import { loadAttemptDetail, loadDashboardData } from "./api";
import {
  AttemptWaterfall,
  ColdStartCost,
  EmptyChart,
  FailureTable,
  StageLatencyChart,
  WorkloadScatter,
} from "./charts";
import { ageLabel, compactNumber, percent, seconds, shortId, spanLabel, stageLabel, utcTime } from "./format";
import { mockAttemptEvents, mockData } from "./mockData";
import type {
  AttemptEventRow,
  AttemptRow,
  DashboardData,
  DashboardFilters,
  SpanLatencyRow,
  StallRow,
} from "./types";

type View = "overview" | "attempts" | "stages" | "failures";

const INITIAL_FILTERS: DashboardFilters = {
  rangeDays: 30,
  environment: "production",
  durationBucket: "5_10s",
  gpuType: "all",
  backend: "all",
  coldStart: "all",
};

const EMPTY_DATA: DashboardData = {
  overview: null,
  stages: [],
  spans: [],
  attempts: [],
  failures: [],
  stalls: [],
  freshness: null,
  coldStages: [],
  warmStages: [],
};

const CACHE_PREFIX = "wdirl.analytics.v1:";
const CACHE_MAX_AGE_MS = 24 * 60 * 60 * 1_000;

type AppProps = {
  demoMode?: boolean;
  dataLoader?: typeof loadDashboardData;
  attemptLoader?: typeof loadAttemptDetail;
  storage?: Pick<Storage, "getItem" | "setItem">;
};

function cacheKey(filters: DashboardFilters): string {
  return `${CACHE_PREFIX}${JSON.stringify(filters)}`;
}

function cachedData(filters: DashboardFilters, storage?: Pick<Storage, "getItem">): DashboardData | null {
  try {
    const value = storage?.getItem(cacheKey(filters));
    if (!value) return null;
    const parsed = JSON.parse(value) as { savedAt?: number; data?: DashboardData };
    if (!parsed.savedAt || !parsed.data || Date.now() - parsed.savedAt > CACHE_MAX_AGE_MS) return null;
    return parsed.data;
  } catch {
    return null;
  }
}

function storeCachedData(filters: DashboardFilters, data: DashboardData, storage?: Pick<Storage, "setItem">): void {
  try {
    storage?.setItem(cacheKey(filters), JSON.stringify({ savedAt: Date.now(), data }));
  } catch {
    // A private browser or full storage must not break the operator dashboard.
  }
}

const NAVIGATION: Array<{ view: View; label: string; icon: typeof SquaresFour }> = [
  { view: "overview", label: "Overview", icon: SquaresFour },
  { view: "attempts", label: "Attempts", icon: ListMagnifyingGlass },
  { view: "stages", label: "Stages", icon: Stack },
  { view: "failures", label: "Failures", icon: Warning },
];

function KpiCard({ label, value, sample, tone = "default" }: { label: string; value: string; sample: string; tone?: "default" | "healthy" | "danger" }) {
  return (
    <section className={`kpi-card tone-${tone}`}>
      <div><span>{label}</span><Gauge size={15} aria-hidden="true" /></div>
      <strong>{value}</strong>
      <small>{sample}</small>
    </section>
  );
}

function FilterSelect({
  icon: Icon,
  label,
  value,
  onChange,
  children,
}: {
  icon: typeof CalendarBlank;
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
}) {
  return (
    <label className="filter-select">
      <Icon size={18} aria-hidden="true" />
      <span className="sr-only">{label}</span>
      <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>{children}</select>
      <CaretDown size={13} aria-hidden="true" />
    </label>
  );
}

function Panel({ title, aside, children, className = "" }: { title: string; aside?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`.trim()}>
      <header className="panel-header"><h2>{title}</h2>{aside}</header>
      {children}
    </section>
  );
}

function AttemptTable({ attempts, selectedId, onSelect }: { attempts: AttemptRow[]; selectedId?: string; onSelect: (attempt: AttemptRow) => void }) {
  if (!attempts.length) return <EmptyChart message="No processing attempts match these filters." />;
  return (
    <div className="attempt-table-wrap">
      <table className="attempt-table">
        <thead><tr><th>Run / attempt</th><th>Status</th><th>Started (UTC)</th><th>Clip</th><th>GPU</th><th>Result Ready</th><th>Bottleneck</th></tr></thead>
        <tbody>
          {attempts.map((attempt) => (
            <tr key={attempt.attempt_id} className={attempt.attempt_id === selectedId ? "is-selected" : ""}>
              <td><button type="button" aria-label={`Run ${attempt.run_id}, attempt ${attempt.attempt_id}`} onClick={() => onSelect(attempt)}><span>{shortId(attempt.run_id)}</span><small>{shortId(attempt.attempt_id)}</small></button></td>
              <td><span className={`status status-${attempt.status}`}>{attempt.status === "complete" ? <CheckCircle size={14} /> : attempt.status === "failed" ? <XCircle size={14} /> : <ClockCountdown size={14} />}{attempt.status}</span></td>
              <td>{utcTime(attempt.first_event_at)}</td>
              <td>{attempt.clip_frame_count ? `${attempt.clip_frame_count}f` : "—"} · {attempt.resolution_bucket ?? "unknown"}</td>
              <td>{attempt.gpu_type ?? "—"}{attempt.cold_start ? <em>cold</em> : null}</td>
              <td>{seconds(attempt.result_ready_seconds)}</td>
              <td>{stageLabel(attempt.bottleneck_stage)} <small>{seconds(attempt.bottleneck_seconds)}</small></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StallTable({ stalls }: { stalls: StallRow[] }) {
  return (
    <div className="attempt-table-wrap">
      <table className="attempt-table stall-table">
        <thead><tr><th>Run</th><th>Attempt</th><th>Last event (UTC)</th><th>Stale for</th><th>Last boundary</th></tr></thead>
        <tbody>
          {stalls.map((stall) => (
            <tr key={stall.attempt_id}>
              <td><code>{shortId(stall.run_id)}</code></td>
              <td><code>{shortId(stall.attempt_id)}</code></td>
              <td>{utcTime(stall.last_event_at)}</td>
              <td>{seconds(stall.stale_seconds)}</td>
              <td>{stageLabel(stall.last_stage)}{stall.last_span ? ` · ${spanLabel(stall.last_span)}` : ` · ${stall.last_event_type}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SpanRows({ rows, stage }: { rows: SpanLatencyRow[]; stage: string | null }) {
  const filtered = rows.filter((row) => !stage || row.stage === stage).slice(0, 9);
  if (!filtered.length) return <EmptyChart message="No nested span timings for this stage." />;
  const max = Math.max(1, ...filtered.map((row) => row.p95_seconds ?? 0));
  return (
    <div className="span-list">
      {filtered.map((row) => (
        <div className="span-row" key={`${row.stage}-${row.span}`}>
          <div><strong>{spanLabel(row.span)}</strong><span>{stageLabel(row.stage)} · n={row.samples}</span></div>
          <div className="span-track"><i style={{ width: `${((row.p95_seconds ?? 0) / max) * 100}%` }} /></div>
          <span><small>p50 {seconds(row.p50_seconds)}</small><strong>p95 {seconds(row.p95_seconds)}</strong></span>
        </div>
      ))}
    </div>
  );
}

function Overview({
  data,
  selectedAttempt,
  selectedStage,
  attemptEvents,
  onSelectStage,
}: {
  data: DashboardData;
  selectedAttempt: AttemptRow | null;
  selectedStage: string | null;
  attemptEvents: AttemptEventRow[];
  onSelectStage: (stage: string) => void;
}) {
  const overview = data.overview;
  const successRate = overview?.success_rate ?? null;
  const selectedStageRow = data.stages.find((row) => row.stage === selectedStage) ?? data.stages[0];
  const selectedIsBottleneck = selectedStageRow?.stage === overview?.bottleneck_stage;
  const dominantSpan = data.spans
    .filter((row) => row.stage === selectedStageRow?.stage)
    .sort((a, b) => (b.p95_seconds ?? 0) - (a.p95_seconds ?? 0))[0];

  return (
    <>
      <div className="kpi-grid">
        <KpiCard label="Result Ready p50" value={seconds(overview?.p50_result_ready_seconds)} sample={`n=${overview?.result_ready_attempts ?? 0}`} />
        <KpiCard label="Result Ready p95" value={seconds(overview?.p95_result_ready_seconds)} sample={`n=${overview?.result_ready_attempts ?? 0}`} tone={(overview?.p95_result_ready_seconds ?? 0) > 180 ? "danger" : "default"} />
        <KpiCard label="Success rate" value={percent(successRate)} sample={`n=${overview?.terminal_attempts ?? 0} terminal`} tone={successRate != null && successRate >= 0.95 ? "healthy" : "default"} />
        <KpiCard label="Attempts" value={String(overview?.attempts ?? 0)} sample={`${data.stalls.length} stalled`} />
      </div>

      <div className="primary-grid">
        <Panel title="Which stage is slow?" aside={<div className="legend"><span className="legend-p50">p50</span><span className="legend-p95">p95</span><span className="legend-alert">bottleneck</span></div>}>
          <StageLatencyChart rows={data.stages} selectedStage={selectedStageRow?.stage} onSelect={onSelectStage} limit={8} />
        </Panel>
        <Panel title={selectedIsBottleneck ? "Current bottleneck" : "Selected stage"} className="bottleneck-panel">
          {selectedStageRow ? (
            <div className="bottleneck-content">
              <h3>{stageLabel(selectedStageRow.stage)}</h3>
              <span>p95</span>
              <strong>{seconds(selectedStageRow.p95_seconds)}</strong>
              <p>{selectedStageRow.confidence === "low" ? "Low-confidence tail estimate" : selectedIsBottleneck ? "Largest p95 contributor in this cohort" : "p95 for the selected stage in this cohort"}</p>
              <footer><div><b>n={selectedStageRow.samples}</b><span>sample count</span></div><div><Gauge size={38} /><span>{dominantSpan ? `${spanLabel(dominantSpan.span)} dominates` : `${seconds(selectedStageRow.average_ms_per_frame, 0)} ms/frame`}</span></div></footer>
            </div>
          ) : <EmptyChart message="No bottleneck can be calculated yet." />}
        </Panel>
      </div>

      <Panel
        title="Selected attempt waterfall"
        className="waterfall-panel"
        aside={selectedAttempt ? <div className="waterfall-meta"><span>Run <b>{shortId(selectedAttempt.run_id)}</b> · Attempt <b>{shortId(selectedAttempt.attempt_id)}</b></span><span>Result Ready <strong>{seconds(selectedAttempt.result_ready_seconds)}</strong></span></div> : undefined}
      >
        <AttemptWaterfall attempt={selectedAttempt} events={attemptEvents} />
      </Panel>

      <div className="support-grid">
        <Panel title="Workload vs latency"><WorkloadScatter attempts={data.attempts} /></Panel>
        <Panel title="Cold start cost"><ColdStartCost cold={data.coldStages} warm={data.warmStages} /></Panel>
        <Panel title="Recent failures"><FailureTable rows={data.failures} /></Panel>
      </div>
    </>
  );
}

export function App({
  demoMode = import.meta.env.DEV,
  dataLoader = loadDashboardData,
  attemptLoader = loadAttemptDetail,
  storage = window.localStorage,
}: AppProps = {}) {
  const [view, setView] = useState<View>("overview");
  const [filters, setFilters] = useState<DashboardFilters>(INITIAL_FILTERS);
  const [data, setData] = useState<DashboardData>(() => demoMode ? mockData : cachedData(INITIAL_FILTERS, storage) ?? EMPTY_DATA);
  const [loading, setLoading] = useState(!demoMode);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedAttemptId, setSelectedAttemptId] = useState<string | null>(demoMode ? mockData.attempts[0].attempt_id : null);
  const [attemptEvents, setAttemptEvents] = useState<AttemptEventRow[]>(demoMode ? mockAttemptEvents : []);
  const [selectedStage, setSelectedStage] = useState<string | null>(demoMode ? "runner_mask" : null);

  useEffect(() => {
    if (demoMode) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    dataLoader(filters, controller.signal)
      .then((result) => {
        setData(result);
        storeCachedData(filters, result, storage);
        setSelectedStage((current) => current ?? result.overview?.bottleneck_stage ?? result.stages[0]?.stage ?? null);
        setSelectedAttemptId((current) => current && result.attempts.some((attempt) => attempt.attempt_id === current) ? current : result.attempts[0]?.attempt_id ?? null);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Analytics could not be loaded.");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [dataLoader, demoMode, filters, refreshKey, storage]);

  const selectedAttempt = useMemo(
    () => data.attempts.find((attempt) => attempt.attempt_id === selectedAttemptId) ?? data.attempts[0] ?? null,
    [data.attempts, selectedAttemptId],
  );

  useEffect(() => {
    if (demoMode || !selectedAttempt) return;
    const controller = new AbortController();
    setAttemptEvents([]);
    attemptLoader(selectedAttempt.attempt_id, controller.signal)
      .then(setAttemptEvents)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Attempt detail could not be loaded.");
      });
    return () => controller.abort();
  }, [attemptLoader, demoMode, selectedAttempt?.attempt_id]);

  const gpuOptions = useMemo(() => [...new Set(data.attempts.map((attempt) => attempt.gpu_type).filter(Boolean) as string[])], [data.attempts]);
  const backendOptions = useMemo(() => [...new Set(data.attempts.map((attempt) => attempt.backend).filter(Boolean) as string[])], [data.attempts]);
  const selectedStageRow = data.stages.find((stage) => stage.stage === selectedStage) ?? null;

  const setFilter = <Key extends keyof DashboardFilters>(key: Key, value: DashboardFilters[Key]) => {
    setFilters((current) => ({ ...current, [key]: value }));
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="wordmark">WDIRL</div>
        <nav aria-label="Analytics views">
          {NAVIGATION.map(({ view: itemView, label, icon: Icon }) => (
            <button className={view === itemView ? "is-active" : ""} key={itemView} onClick={() => setView(itemView)} type="button">
              <Icon size={21} weight={view === itemView ? "fill" : "regular"} /><span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer"><span>Private operator surface</span><b>v1.0</b></div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><span className="eyebrow">Hosted pipeline</span><h1>Processing Analytics</h1></div>
          <div className="toolbar">
            <FilterSelect icon={Stack} label="Environment" value={filters.environment} onChange={(value) => setFilter("environment", value)}><option value="production">Production</option></FilterSelect>
            <FilterSelect icon={CalendarBlank} label="Date range" value={String(filters.rangeDays)} onChange={(value) => setFilter("rangeDays", Number(value) as DashboardFilters["rangeDays"])}><option value="1">Last 24 hours</option><option value="7">Last 7 days</option><option value="14">Last 14 days</option><option value="30">Last 30 days</option><option value="90">Last 90 days</option></FilterSelect>
            <FilterSelect icon={SquaresFour} label="Clip duration" value={filters.durationBucket} onChange={(value) => setFilter("durationBucket", value)}><option value="all">All clip lengths</option><option value="0_5s">0–5s clips</option><option value="5_10s">5–10s clips</option><option value="10_20s">10–20s clips</option><option value="over_20s">20s+ clips</option></FilterSelect>
            <FilterSelect icon={Cpu} label="GPU" value={filters.gpuType} onChange={(value) => setFilter("gpuType", value)}><option value="all">All GPUs</option>{gpuOptions.map((gpu) => <option value={gpu} key={gpu}>{gpu}</option>)}</FilterSelect>
            {backendOptions.length ? <FilterSelect icon={Gauge} label="Mask backend" value={filters.backend} onChange={(value) => setFilter("backend", value)}><option value="all">All mask backends</option>{backendOptions.map((backend) => <option value={backend} key={backend}>{backend}</option>)}</FilterSelect> : null}
            <span className="freshness"><i className={data.freshness?.event_age_seconds != null && data.freshness.event_age_seconds < 600 ? "is-live" : ""} />{ageLabel(data.freshness?.event_age_seconds)}</span>
            <button className="refresh" type="button" aria-label="Refresh analytics" onClick={() => setRefreshKey((value) => value + 1)} disabled={loading}><ArrowsClockwise size={20} className={loading ? "spin" : ""} /></button>
          </div>
        </header>

        {error ? <div className="error-banner" role="alert"><Warning size={18} /><span>{error}</span><button type="button" onClick={() => setRefreshKey((value) => value + 1)}>Retry</button></div> : null}
        {loading ? <div className="loading-line" aria-label="Loading analytics" /> : null}

        {view === "overview" ? (
          <Overview data={data} selectedAttempt={selectedAttempt} selectedStage={selectedStage} attemptEvents={attemptEvents} onSelectStage={(stage) => setSelectedStage(stage)} />
        ) : null}

        {view === "attempts" ? (
          <div className="page-stack">
            <Panel title="Processing attempts" aside={<span className="panel-count">{data.attempts.length} shown</span>}><AttemptTable attempts={data.attempts} selectedId={selectedAttempt?.attempt_id} onSelect={(attempt) => setSelectedAttemptId(attempt.attempt_id)} /></Panel>
            <Panel title="Attempt timing" aside={selectedAttempt ? <span className="panel-count">{shortId(selectedAttempt.attempt_id)} · {seconds(selectedAttempt.result_ready_seconds)}</span> : undefined}><AttemptWaterfall attempt={selectedAttempt} events={attemptEvents} /></Panel>
          </div>
        ) : null}

        {view === "stages" ? (
          <div className="stage-page">
            <Panel title="Stage tail latency" aside={<span className="panel-count">p50 → p95</span>}><StageLatencyChart rows={data.stages} selectedStage={selectedStage} onSelect={setSelectedStage} /></Panel>
            <Panel title={selectedStageRow ? `${stageLabel(selectedStageRow.stage)} spans` : "Processing spans"}><SpanRows rows={data.spans} stage={selectedStage} /></Panel>
            <Panel title="Cold start contribution"><ColdStartCost cold={data.coldStages} warm={data.warmStages} /></Panel>
          </div>
        ) : null}

        {view === "failures" ? (
          <div className="failure-page">
            <div className="kpi-grid failure-kpis">
              <KpiCard label="Failed attempts" value={String(data.overview?.failed_attempts ?? 0)} sample={`of ${data.overview?.terminal_attempts ?? 0} terminal`} tone={(data.overview?.failed_attempts ?? 0) > 0 ? "danger" : "healthy"} />
              <KpiCard label="Failure rate" value={percent(data.overview?.failure_rate)} sample="terminal attempts" tone={(data.overview?.failure_rate ?? 0) > 0.05 ? "danger" : "default"} />
              <KpiCard label="Stalled attempts" value={String(data.stalls.length)} sample="no event ≥10m" tone={data.stalls.length ? "danger" : "healthy"} />
              <KpiCard label="Events · 24h" value={compactNumber(data.freshness?.events_last_24_hours)} sample={`${data.freshness?.attempts_last_24_hours ?? 0} attempts`} />
            </div>
            <Panel title="Failure classifications"><FailureTable rows={data.failures} /></Panel>
            <Panel title="Stalled attempts">
              {data.stalls.length ? <StallTable stalls={data.stalls} /> : <EmptyChart message="No stalled attempts in this cohort." />}
            </Panel>
          </div>
        ) : null}
      </main>
    </div>
  );
}
