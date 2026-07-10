import { WarningCircle } from "@phosphor-icons/react";
import { compactNumber, seconds, spanLabel, stageLabel } from "./format";
import type { AttemptEventRow, AttemptRow, FailureRow, StageLatencyRow } from "./types";

export function EmptyChart({ message }: { message: string }) {
  return (
    <div className="empty-chart" role="status">
      <div className="empty-chart-mark" aria-hidden="true" />
      <p>{message}</p>
      <span>Process a clip to populate this view.</span>
    </div>
  );
}

export function StageLatencyChart({
  rows,
  selectedStage,
  onSelect,
  limit,
}: {
  rows: StageLatencyRow[];
  selectedStage?: string | null;
  onSelect?: (stage: string) => void;
  limit?: number;
}) {
  if (!rows.length) return <EmptyChart message="No completed stage timings in this cohort." />;
  const eligible = rows.filter((row) => row.p95_seconds != null);
  const displayed = limit == null ? eligible : eligible.slice(0, limit);
  const maximum = Math.max(1, ...displayed.map((row) => row.p95_seconds ?? 0)) * 1.08;

  return (
    <div className="interval-chart" data-testid="stage-latency-chart">
      <div className="interval-header">
        <span>Stage</span>
        <span>p50 → p95 (seconds)</span>
      </div>
      {displayed.map((row) => {
        const selected = row.stage === selectedStage;
        const p50 = ((row.p50_seconds ?? 0) / maximum) * 100;
        const p95 = ((row.p95_seconds ?? 0) / maximum) * 100;
        return (
          <button
            className={`interval-row${selected ? " is-selected" : ""}`}
            key={row.stage}
            onClick={() => onSelect?.(row.stage)}
            type="button"
          >
            <span className="interval-name">
              <strong>{stageLabel(row.stage)}</strong>
              <span>n={row.samples}</span>
              {row.confidence === "low" ? <em>low confidence</em> : null}
            </span>
            <span className="interval-plot" aria-label={`${stageLabel(row.stage)} p50 ${seconds(row.p50_seconds)}, p95 ${seconds(row.p95_seconds)}`}>
              <span className="interval-line" style={{ left: `${p50}%`, width: `${Math.max(0, p95 - p50)}%` }} />
              <span className="interval-dot p50" style={{ left: `${p50}%` }} />
              <span className="interval-dot p95" style={{ left: `${p95}%` }} />
              <span className="interval-value p50-value" style={{ left: `${p50}%` }}>{seconds(row.p50_seconds)}</span>
              <span className="interval-value p95-value" style={{ left: `${p95}%` }}>{seconds(row.p95_seconds)}</span>
            </span>
          </button>
        );
      })}
      <div className="interval-axis" aria-hidden="true">
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
          <span key={fraction} style={{ left: `${fraction * 100}%` }}>{Math.round(maximum * fraction)}s</span>
        ))}
      </div>
    </div>
  );
}

function waterfallRows(events: AttemptEventRow[]) {
  const stages = events.filter((event) => event.event_type === "stage_completed" || event.event_type === "stage_failed");
  const spansByStage = new Map<string, AttemptEventRow[]>();
  for (const event of events) {
    if (!event.stage || !(event.event_type === "span_completed" || event.event_type === "span_failed")) continue;
    const existing = spansByStage.get(event.stage) ?? [];
    existing.push(event);
    spansByStage.set(event.stage, existing);
  }
  return stages.flatMap((stage) => [
    { ...stage, kind: "stage" as const },
    ...(spansByStage.get(stage.stage ?? "") ?? []).map((span) => ({ ...span, kind: "span" as const })),
  ]);
}

export function AttemptWaterfall({ attempt, events }: { attempt: AttemptRow | null; events: AttemptEventRow[] }) {
  if (!attempt || !events.length) return <EmptyChart message="Select a completed attempt to inspect its timing waterfall." />;
  const rows = waterfallRows(events);
  const maximum = Math.max(attempt.result_ready_seconds ?? 0, ...rows.map((row) => row.end_offset_seconds), 1);
  const bottleneck = attempt.bottleneck_stage;

  return (
    <div className="waterfall" data-testid="attempt-waterfall">
      <div className="waterfall-axis">
        <span />
        <div>
          {[0, 0.2, 0.4, 0.6, 0.8, 1].map((fraction) => (
            <span key={fraction} style={{ left: `${fraction * 100}%` }}>{Math.round(maximum * fraction)}s</span>
          ))}
        </div>
      </div>
      {rows.map((row) => {
        const start = Math.max(0, row.start_offset_seconds);
        const end = Math.max(start, row.end_offset_seconds);
        const left = (start / maximum) * 100;
        const width = Math.max(((end - start) / maximum) * 100, 0.6);
        const active = row.stage === bottleneck;
        return (
          <div className={`waterfall-row ${row.kind}${active ? " is-bottleneck" : ""}`} key={`${row.sequence}-${row.event_type}`}>
            <div className="waterfall-label">
              <span>{row.kind === "span" ? spanLabel(row.span) : stageLabel(row.stage)}</span>
              {row.kind === "stage" ? <small>{seconds(row.elapsed_seconds)}</small> : null}
            </div>
            <div className="waterfall-track">
              <span className="waterfall-grid" aria-hidden="true" />
              <span className="waterfall-bar" style={{ left: `${left}%`, width: `${width}%` }}>
                {width > 10 ? <span>{row.kind === "span" ? spanLabel(row.span) : `${start.toFixed(1)}–${end.toFixed(1)}s`}</span> : null}
              </span>
            </div>
          </div>
        );
      })}
      {attempt.unattributed_seconds > 0.05 ? (
        <div className="waterfall-gap">
          <span>Unattributed gap</span>
          <strong>{seconds(attempt.unattributed_seconds)}</strong>
        </div>
      ) : null}
    </div>
  );
}

export function WorkloadScatter({ attempts }: { attempts: AttemptRow[] }) {
  const points = attempts.filter((attempt) => attempt.clip_frame_count && attempt.result_ready_seconds);
  if (!points.length) return <EmptyChart message="No completed attempts with workload metadata." />;
  const width = 420;
  const height = 210;
  const padding = { left: 46, right: 16, top: 14, bottom: 36 };
  const xMax = Math.max(...points.map((point) => point.clip_frame_count ?? 0), 1) * 1.08;
  const yMax = Math.max(...points.map((point) => point.result_ready_seconds ?? 0), 1) * 1.08;
  const x = (value: number) => padding.left + (value / xMax) * (width - padding.left - padding.right);
  const y = (value: number) => height - padding.bottom - (value / yMax) * (height - padding.top - padding.bottom);

  return (
    <svg className="scatter" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Clip frame count versus Result Ready latency">
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
        <g key={fraction}>
          <line x1={padding.left} x2={width - padding.right} y1={y(yMax * fraction)} y2={y(yMax * fraction)} className="chart-grid" />
          <text x={padding.left - 8} y={y(yMax * fraction) + 4} textAnchor="end">{Math.round(yMax * fraction)}</text>
        </g>
      ))}
      {points.map((point) => (
        <circle key={point.attempt_id} cx={x(point.clip_frame_count ?? 0)} cy={y(point.result_ready_seconds ?? 0)} r="4" className={point.status === "failed" ? "scatter-failed" : "scatter-point"}>
          <title>{`${compactNumber(point.clip_frame_count)} frames · ${seconds(point.result_ready_seconds)} · ${point.gpu_type ?? "GPU unknown"}`}</title>
        </circle>
      ))}
      <text x={width / 2} y={height - 5} textAnchor="middle" className="axis-title">Clip frames</text>
      <text transform={`translate(13 ${height / 2}) rotate(-90)`} textAnchor="middle" className="axis-title">Result Ready (s)</text>
      <text x={width - padding.right} y={padding.top + 4} textAnchor="end" className="sample-label">n={points.length}</text>
    </svg>
  );
}

export function ColdStartCost({ cold, warm }: { cold: StageLatencyRow[]; warm: StageLatencyRow[] }) {
  const warmByStage = new Map(warm.map((row) => [row.stage, row]));
  const rows = cold
    .flatMap((row) => {
      const warmRow = warmByStage.get(row.stage);
      if (row.p50_seconds == null || warmRow?.p50_seconds == null) return [];
      return [{ stage: row.stage, delta: row.p50_seconds - warmRow.p50_seconds }];
    })
    .filter((row) => row.delta > 0.05)
    .sort((a, b) => b.delta - a.delta)
    .slice(0, 5);
  if (!rows.length) return <EmptyChart message="Cold and warm samples are not both available." />;
  const maximum = Math.max(...rows.map((row) => row.delta), 1);
  const coldSamples = Math.max(0, ...cold.map((row) => row.samples));
  const warmSamples = Math.max(0, ...warm.map((row) => row.samples));

  return (
    <div className="cold-chart">
      <div className="cold-legend"><span>Warm n={warmSamples}</span><span>Cold n={coldSamples}</span></div>
      {rows.map((row) => (
        <div className="cold-row" key={row.stage}>
          <span>{stageLabel(row.stage)}</span>
          <div><i style={{ width: `${(row.delta / maximum) * 100}%` }} /><strong>+{seconds(row.delta)}</strong></div>
        </div>
      ))}
      <p>Additional median time versus comparable warm attempts.</p>
    </div>
  );
}

export function FailureTable({ rows }: { rows: FailureRow[] }) {
  if (!rows.length) return <EmptyChart message="No failures in this cohort." />;
  return (
    <div className="failure-table-wrap">
      <table className="failure-table">
        <thead><tr><th>Stage</th><th>Error</th><th>Attempts</th><th>p95 fail time</th></tr></thead>
        <tbody>
          {rows.slice(0, 6).map((row) => (
            <tr key={`${row.stage}-${row.span}-${row.error_code}`}>
              <td>{stageLabel(row.stage)}</td>
              <td><code>{row.error_code ?? row.error_class ?? "UNCLASSIFIED"}</code></td>
              <td>{row.affected_attempts}</td>
              <td>{seconds(row.p95_time_to_failure_seconds)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="privacy-note"><WarningCircle size={16} /> Sanitized classifications only</div>
    </div>
  );
}
