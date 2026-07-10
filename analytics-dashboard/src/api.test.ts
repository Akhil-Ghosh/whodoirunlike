import { afterEach, describe, expect, it, vi } from "vitest";
import { executeQuery, filtersToParameters } from "./api";

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("dashboard query filters", () => {
  it("omits unselected dimensions and preserves the comparison cohort", () => {
    expect(
      filtersToParameters({
        rangeDays: 30,
        environment: "production",
        durationBucket: "5_10s",
        gpuType: "all",
        backend: "all",
        coldStart: "all",
      }),
    ).toEqual({ range_days: 30, environment: "production", duration_bucket: "5_10s" });
  });

  it("maps cold and warm selection to booleans", () => {
    expect(
      filtersToParameters({
        rangeDays: 7,
        environment: "production",
        durationBucket: "all",
        gpuType: "NVIDIA L4",
        backend: "sam31_gpu",
        coldStart: "warm",
      }),
    ).toEqual({
      range_days: 7,
      environment: "production",
      gpu_type: "NVIDIA L4",
      backend: "sam31_gpu",
      cold_start: false,
    });
  });

  it("backs off and retries a throttled query request", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response('{"error":"rate limited"}', { status: 429 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        query: "overview",
        query_execution_id: "2f47c767-1b08-4a72-95a1-bd5da365fe60",
        state: "QUEUED",
        poll_after_ms: 1_500,
      }), { status: 202, headers: { "content-type": "application/json" } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        query: "overview",
        query_execution_id: "2f47c767-1b08-4a72-95a1-bd5da365fe60",
        state: "SUCCEEDED",
        rows: [{ attempts: 1 }],
      }), { status: 200, headers: { "content-type": "application/json" } }));

    const pending = executeQuery<{ attempts: number }>("overview", { range_days: 30 });
    await vi.advanceTimersByTimeAsync(500);
    await vi.advanceTimersByTimeAsync(1_500);

    await expect(pending).resolves.toEqual([{ attempts: 1 }]);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });
});
