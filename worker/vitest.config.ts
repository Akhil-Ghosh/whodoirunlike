import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: {
        bindings: {
          PROCESSOR_SHARED_SECRET: "worker-test-processor-secret",
          RUNPOD_API_KEY: "worker-test-runpod-key",
          AWS_ANALYTICS_INGEST_URL: "",
          AWS_ANALYTICS_SHARED_SECRET: "worker-test-analytics-secret",
        },
      },
    }),
  ],
  test: {
    testTimeout: 15_000,
  },
});
