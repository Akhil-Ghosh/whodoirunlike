import assert from "node:assert/strict";
import test from "node:test";

import { jobResultReady } from "./uploadJobState.ts";

test("explicit result_ready false overrides a fused artifact", () => {
  assert.equal(
    jobResultReady({
      result_ready: false,
      artifacts: { "fused_overlay.mp4": { href: "https://example.test/stale.mp4" } },
    }),
    false,
  );
});

test("legacy jobs fall back to fused artifact presence when result_ready is absent", () => {
  assert.equal(
    jobResultReady({
      artifacts: { "fused_overlay.mp4": { href: "https://example.test/ready.mp4" } },
    }),
    true,
  );
  assert.equal(jobResultReady({ artifacts: {} }), false);
});
