const encoder = new TextEncoder();
const API_POST_PATH = "/api/queries";
const API_POLL_PATH = /^\/api\/queries\/([0-9a-f-]{20,64})$/i;
const MAX_BODY_BYTES = 8_192;

function securityHeaders(headers = new Headers()): Headers {
  headers.set("x-content-type-options", "nosniff");
  headers.set("x-frame-options", "DENY");
  headers.set("strict-transport-security", "max-age=31536000; includeSubDomains");
  headers.set("referrer-policy", "no-referrer");
  headers.set("permissions-policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()");
  headers.set("x-robots-tag", "noindex, nofollow, noarchive");
  headers.set(
    "content-security-policy",
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
  );
  return headers;
}

function json(status: number, payload: Record<string, unknown>): Response {
  const headers = securityHeaders(new Headers({
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  }));
  return new Response(JSON.stringify(payload), { status, headers });
}

function unauthorized(): Response {
  const headers = securityHeaders(new Headers({
    "content-type": "text/plain; charset=utf-8",
    "cache-control": "no-store",
    "www-authenticate": 'Basic realm="WDIRL Processing Analytics", charset="UTF-8"',
  }));
  return new Response("Authentication required.", { status: 401, headers });
}

function redirectToHttps(url: URL): Response {
  url.protocol = "https:";
  return new Response(null, {
    status: 308,
    headers: securityHeaders(new Headers({ location: url.toString(), "cache-control": "no-store" })),
  });
}

export function parseBasicAuthorization(value: string | null): { username: string; password: string } | null {
  if (!value?.startsWith("Basic ")) return null;
  const encoded = value.slice(6).trim();
  if (!encoded || encoded.length > 4_096) return null;
  try {
    const decoded = atob(encoded);
    const separator = decoded.indexOf(":");
    if (separator < 1) return null;
    return { username: decoded.slice(0, separator), password: decoded.slice(separator + 1) };
  } catch {
    return null;
  }
}

async function secureEqual(left: string, right: string): Promise<boolean> {
  const [leftDigest, rightDigest] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(left)),
    crypto.subtle.digest("SHA-256", encoder.encode(right)),
  ]);
  const subtle = crypto.subtle as SubtleCrypto & {
    timingSafeEqual?: (a: ArrayBuffer | ArrayBufferView, b: ArrayBuffer | ArrayBufferView) => boolean;
  };
  if (subtle.timingSafeEqual) return subtle.timingSafeEqual(leftDigest, rightDigest);
  const leftBytes = new Uint8Array(leftDigest);
  const rightBytes = new Uint8Array(rightDigest);
  let difference = 0;
  for (let index = 0; index < leftBytes.length; index += 1) difference |= leftBytes[index] ^ rightBytes[index];
  return difference === 0;
}

async function authorized(request: Request, env: Env): Promise<boolean> {
  const credentials = parseBasicAuthorization(request.headers.get("authorization"));
  if (!credentials || !env.DASHBOARD_PASSWORD || !env.DASHBOARD_USERNAME) return false;
  const [usernameMatches, passwordMatches] = await Promise.all([
    secureEqual(credentials.username, env.DASHBOARD_USERNAME),
    secureEqual(credentials.password, env.DASHBOARD_PASSWORD),
  ]);
  return usernameMatches && passwordMatches;
}

function bytesToHex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function canonicalDashboardMessage(
  timestamp: string,
  method: string,
  canonicalPath: string,
  body: Uint8Array,
): Uint8Array {
  const prefix = encoder.encode(`${timestamp}\n${method}\n${canonicalPath}\n`);
  const message = new Uint8Array(prefix.length + body.length);
  message.set(prefix);
  message.set(body, prefix.length);
  return message;
}

async function signature(secret: string, message: Uint8Array): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signable = new Uint8Array(message).buffer;
  return bytesToHex(await crypto.subtle.sign("HMAC", key, signable));
}

export async function readLimitedBody(
  request: Request,
  maximumBytes = MAX_BODY_BYTES,
): Promise<Uint8Array | null> {
  const contentLength = request.headers.get("content-length");
  if (contentLength && /^\d+$/.test(contentLength) && Number(contentLength) > maximumBytes) {
    return null;
  }
  if (!request.body) return new Uint8Array();

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximumBytes) {
        await reader.cancel("request body is too large");
        return null;
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

function upstreamTarget(requestUrl: URL, env: Env): { url: URL; canonicalPath: string } | null {
  const base = new URL(env.AWS_DASHBOARD_API_URL);
  if (base.protocol !== "https:" || !base.pathname.endsWith("/queries")) throw new Error("AWS dashboard API URL is invalid");
  if (requestUrl.pathname === API_POST_PATH) return { url: base, canonicalPath: "/queries" };
  const match = API_POLL_PATH.exec(requestUrl.pathname);
  if (!match) return null;
  base.pathname = `${base.pathname}/${match[1]}`;
  return { url: base, canonicalPath: `/queries/${match[1]}` };
}

async function proxyAnalytics(request: Request, env: Env): Promise<Response> {
  const requestUrl = new URL(request.url);
  const target = upstreamTarget(requestUrl, env);
  if (!target) return json(404, { error: "not found" });
  const method = request.method.toUpperCase();
  if ((requestUrl.pathname === API_POST_PATH && method !== "POST") || (requestUrl.pathname !== API_POST_PATH && method !== "GET")) {
    return json(405, { error: "method not allowed" });
  }

  const body = method === "POST" ? await readLimitedBody(request) : new Uint8Array();
  if (body === null) return json(413, { error: "request body is too large" });
  const timestamp = Math.floor(Date.now() / 1_000).toString();
  const digest = await signature(
    env.AWS_DASHBOARD_SHARED_SECRET,
    canonicalDashboardMessage(timestamp, method, target.canonicalPath, body),
  );
  const upstreamHeaders = new Headers({
    accept: "application/json",
    "x-wdirl-dashboard-timestamp": timestamp,
    "x-wdirl-dashboard-signature": digest,
  });
  if (method === "POST") upstreamHeaders.set("content-type", "application/json");

  const upstream = await fetch(target.url, {
    method,
    headers: upstreamHeaders,
    body: method === "POST" ? new Uint8Array(body).buffer : undefined,
    redirect: "manual",
  });
  if (upstream.status >= 300 && upstream.status < 400) {
    return json(502, { error: "analytics service unavailable" });
  }
  const headers = securityHeaders(new Headers(upstream.headers));
  headers.set("cache-control", "no-store");
  headers.delete("set-cookie");
  return new Response(upstream.body, { status: upstream.status, headers });
}

async function serveAsset(request: Request, env: Env): Promise<Response> {
  if (request.method !== "GET" && request.method !== "HEAD") return json(405, { error: "method not allowed" });
  const response = await env.ASSETS.fetch(request);
  const headers = securityHeaders(new Headers(response.headers));
  const contentType = headers.get("content-type") ?? "";
  headers.set("cache-control", contentType.includes("text/html") ? "no-store" : "private, max-age=300");
  return new Response(response.body, { status: response.status, headers });
}

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url);
    if (url.protocol !== "https:" || request.headers.get("x-forwarded-proto") === "http") {
      return redirectToHttps(url);
    }
    if (!(await authorized(request, env))) return unauthorized();
    if (url.pathname === "/healthz") return json(200, { ok: true, service: "whodoirunlike-analytics" });
    try {
      if (url.pathname.startsWith("/api/")) return await proxyAnalytics(request, env);
      return await serveAsset(request, env);
    } catch (error) {
      console.error(JSON.stringify({
        message: "dashboard request failed",
        path: url.pathname.startsWith("/api/") ? "api" : "asset",
        error: error instanceof Error ? error.name : "UnknownError",
      }));
      return json(502, { error: "analytics service unavailable" });
    }
  },
} satisfies ExportedHandler<Env>;
