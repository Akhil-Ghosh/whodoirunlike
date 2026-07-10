import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

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
});
