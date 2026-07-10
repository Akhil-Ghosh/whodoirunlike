import type {
  AttemptEventRow,
  DashboardData,
  DashboardFilters,
  OverviewRow,
  QueryName,
  QueryParameters,
  QueryResultResponse,
  QueryStartResponse,
} from "./types";

const MAX_QUERY_WAIT_MS = 90_000;

export class DashboardApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "DashboardApiError";
  }
}

function delay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timeout = window.setTimeout(resolve, milliseconds);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timeout);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}

async function jsonResponse<ResponseBody>(response: Response): Promise<ResponseBody> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new DashboardApiError("The analytics service returned an unreadable response.", response.status);
  }
  if (!response.ok) {
    const message =
      typeof payload === "object" && payload !== null && "error" in payload
        ? String(payload.error)
        : "The analytics service is unavailable.";
    throw new DashboardApiError(message, response.status);
  }
  return payload as ResponseBody;
}

async function fetchWithRateLimitRetry(
  input: RequestInfo | URL,
  init: RequestInit,
  signal?: AbortSignal,
): Promise<Response> {
  let response: Response | null = null;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    response = await fetch(input, { ...init, signal });
    if (response.status !== 429 || attempt === 3) return response;
    await delay(500 * 2 ** attempt, signal);
  }
  return response as Response;
}

export async function executeQuery<Row>(
  query: QueryName,
  parameters: QueryParameters,
  signal?: AbortSignal,
): Promise<Row[]> {
  const startedAt = Date.now();
  const submission = await jsonResponse<QueryStartResponse>(
    await fetchWithRateLimitRetry("/api/queries", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, filters: parameters }),
      cache: "no-store",
    }, signal),
  );

  let wait = Math.max(1_000, Math.min(submission.poll_after_ms ?? 1_500, 5_000));
  while (Date.now() - startedAt < MAX_QUERY_WAIT_MS) {
    await delay(wait, signal);
    const result = await jsonResponse<QueryResultResponse<Row>>(
      await fetchWithRateLimitRetry(`/api/queries/${encodeURIComponent(submission.query_execution_id)}`, {
        cache: "no-store",
      }, signal),
    );
    if (result.state === "SUCCEEDED") return result.rows ?? [];
    if (result.state === "FAILED" || result.state === "CANCELLED") {
      throw new DashboardApiError(result.error?.message ?? "The analytics query did not complete.");
    }
    wait = Math.max(1_000, Math.min(result.poll_after_ms ?? 1_500, 5_000));
  }
  throw new DashboardApiError("The analytics query timed out. Try a shorter date range.");
}

export function filtersToParameters(filters: DashboardFilters): QueryParameters {
  const parameters: QueryParameters = {
    range_days: filters.rangeDays,
    environment: filters.environment,
  };
  if (filters.durationBucket !== "all") parameters.duration_bucket = filters.durationBucket;
  if (filters.gpuType !== "all") parameters.gpu_type = filters.gpuType;
  if (filters.backend !== "all") parameters.backend = filters.backend;
  if (filters.coldStart !== "all") parameters.cold_start = filters.coldStart === "cold";
  return parameters;
}

export async function loadDashboardData(
  filters: DashboardFilters,
  signal?: AbortSignal,
): Promise<DashboardData> {
  const parameters = filtersToParameters(filters);
  const [overview, stages, spans, attempts, failures, stalls, freshness, coldStages, warmStages] =
    await Promise.all([
      executeQuery<OverviewRow>("overview", parameters, signal),
      executeQuery<DashboardData["stages"][number]>("stage_latency", parameters, signal),
      executeQuery<DashboardData["spans"][number]>("span_latency", parameters, signal),
      executeQuery<DashboardData["attempts"][number]>("attempts", { ...parameters, limit: 50 }, signal),
      executeQuery<DashboardData["failures"][number]>("failures", parameters, signal),
      executeQuery<DashboardData["stalls"][number]>("stalls", parameters, signal),
      executeQuery<NonNullable<DashboardData["freshness"]>>(
        "freshness",
        { range_days: Math.min(filters.rangeDays, 7) },
        signal,
      ),
      executeQuery<DashboardData["coldStages"][number]>(
        "stage_latency",
        { ...parameters, cold_start: true },
        signal,
      ),
      executeQuery<DashboardData["warmStages"][number]>(
        "stage_latency",
        { ...parameters, cold_start: false },
        signal,
      ),
    ]);

  return {
    overview: overview[0] ?? null,
    stages,
    spans,
    attempts,
    failures,
    stalls,
    freshness: freshness[0] ?? null,
    coldStages,
    warmStages,
  };
}

export async function loadAttemptDetail(attemptId: string, signal?: AbortSignal): Promise<AttemptEventRow[]> {
  return executeQuery<AttemptEventRow>("attempt_detail", { attempt_id: attemptId, range_days: 90 }, signal);
}
