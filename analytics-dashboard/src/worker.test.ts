import { describe, expect, it } from "vitest";
import worker, {
  canonicalDashboardMessage,
  parseBasicAuthorization,
  readLimitedBody,
} from "./worker";

describe("dashboard Worker authentication helpers", () => {
  it("parses a valid basic authorization header", () => {
    expect(parseBasicAuthorization(`Basic ${btoa("akhil:test-password")}`)).toEqual({
      username: "akhil",
      password: "test-password",
    });
  });

  it("rejects malformed authorization headers", () => {
    expect(parseBasicAuthorization(null)).toBeNull();
    expect(parseBasicAuthorization("Bearer token")).toBeNull();
    expect(parseBasicAuthorization("Basic not-base64-%%%" )).toBeNull();
    expect(parseBasicAuthorization(`Basic ${btoa("missing-separator")}`)).toBeNull();
  });

  it("builds the exact AWS canonical signing bytes", () => {
    const body = new TextEncoder().encode('{"query":"overview","filters":{"range_days":30}}');
    const message = canonicalDashboardMessage("1783660000", "POST", "/queries", body);
    expect(new TextDecoder().decode(message)).toBe(
      '1783660000\nPOST\n/queries\n{"query":"overview","filters":{"range_days":30}}',
    );
  });

  it("redirects plaintext HTTP before requesting credentials", async () => {
    const response = await worker.fetch(
      new Request("https://analytics.whodoirunlike.com/healthz", {
        headers: { "x-forwarded-proto": "http" },
      }) as never,
      {} as Env,
    );
    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      "https://analytics.whodoirunlike.com/healthz",
    );
    expect(response.headers.get("www-authenticate")).toBeNull();
  });

  it("stops buffering request bodies at the configured limit", async () => {
    const accepted = await readLimitedBody(
      new Request("https://analytics.whodoirunlike.com/api/queries", {
        method: "POST",
        body: "12345678",
      }) as never,
      8,
    );
    expect(new TextDecoder().decode(accepted ?? undefined)).toBe("12345678");

    const rejected = await readLimitedBody(
      new Request("https://analytics.whodoirunlike.com/api/queries", {
        method: "POST",
        body: "123456789",
      }) as never,
      8,
    );
    expect(rejected).toBeNull();
  });
});
