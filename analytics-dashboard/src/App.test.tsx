import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { mockAttemptEvents, mockData } from "./mockData";

describe("processing analytics dashboard", () => {
  it("renders the mockup's diagnostic hierarchy in development", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Processing Analytics" })).toBeInTheDocument();
    expect(screen.getByText("Result Ready p50")).toBeInTheDocument();
    expect(screen.getByText("Which stage is slow?")).toBeInTheDocument();
    expect(screen.getByTestId("stage-latency-chart")).toBeInTheDocument();
    expect(screen.getByTestId("attempt-waterfall")).toBeInTheDocument();
    expect(screen.getByText("Workload vs latency")).toBeInTheDocument();
    expect(screen.getByText("Cold start cost")).toBeInTheDocument();
    expect(screen.getByText("Recent failures")).toBeInTheDocument();
  });

  it("keeps the detailed operator views reachable", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Attempts" }));
    expect(screen.getByText("Processing attempts")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stages" }));
    expect(screen.getByText("Stage tail latency")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Failures" }));
    expect(screen.getByText("Failure classifications")).toBeInTheDocument();
  });

  it("keeps the selected waterfall loaded while refreshing the same attempt", async () => {
    const dataLoader = vi.fn().mockResolvedValue(mockData);
    const attemptLoader = vi.fn().mockResolvedValue(mockAttemptEvents);
    render(<App demoMode={false} dataLoader={dataLoader} attemptLoader={attemptLoader} />);

    await screen.findByText("116.3s");
    expect(attemptLoader).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Refresh analytics" }));
    await waitFor(() => expect(dataLoader).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByText("116.3s")).toBeInTheDocument());
    expect(attemptLoader).toHaveBeenCalledTimes(1);
  });

  it("renders the last successful snapshot immediately on first load", () => {
    const values = new Map<string, string>();
    values.set(
      "wdirl.analytics.v1:{\"rangeDays\":30,\"environment\":\"production\",\"durationBucket\":\"5_10s\",\"gpuType\":\"all\",\"backend\":\"all\",\"coldStart\":\"all\"}",
      JSON.stringify({ savedAt: Date.now(), data: mockData }),
    );
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    };
    const dataLoader = vi.fn(() => new Promise<typeof mockData>(() => undefined));
    render(<App demoMode={false} dataLoader={dataLoader} storage={storage} />);
    expect(screen.getByText("214.7s")).toBeInTheDocument();
  });
});
