# Private Processing Analytics Dashboard

Private operator dashboard for the hosted clip-processing pipeline. The browser reaches only this Cloudflare Worker. The Worker enforces HTTP Basic authentication, signs allowlisted AWS query requests with a separate HMAC secret, and proxies results from the cost-bounded Athena dashboard API.

No AWS credential, API URL, HMAC secret, raw SQL, source clip URL, filename, prompt, or exception text is exposed to the browser.

## Local development

```bash
cp .dev.vars.example .dev.vars
npm ci
npm run types
npm run dev
```

Vite development uses local fixture data so the visual surface can be reviewed before production telemetry exists. Production builds always query AWS and show an honest empty state when no matching events exist.

## Deployment inputs

Set `AWS_DASHBOARD_API_URL` in `wrangler.jsonc` to the CloudFormation `DashboardApiUrl` output. Install these secrets without printing their values:

```bash
npx wrangler secret put DASHBOARD_PASSWORD
npx wrangler secret put AWS_DASHBOARD_SHARED_SECRET
```

`AWS_DASHBOARD_SHARED_SECRET` must be the exact plaintext from the Secrets Manager resource identified by the stack's `DashboardSecretArn` output. Do not reuse the telemetry ingestion secret.

Deploy:

```bash
npm run check
npm run deploy
```

The production custom domain is `https://analytics.whodoirunlike.com`. The username is the non-secret `DASHBOARD_USERNAME` Wrangler variable. The generated password is stored locally in macOS Keychain under the service `whodoirunlike-analytics-dashboard`.

Retrieve it on this Mac when you need to sign in:

```bash
security find-generic-password \
  -a akhil \
  -s whodoirunlike-analytics-dashboard \
  -w
```

## Security contract

- Every request, including static assets, requires Basic authentication over HTTPS.
- Browser query bodies are limited to 8 KiB and contain only a fixed query name plus validated filters.
- The Worker signs `<timestamp>\n<METHOD>\n<CANONICAL_PATH>\n<exact body bytes>` with HMAC-SHA256.
- AWS accepts a five-minute timestamp skew and only `/queries` or `/queries/<execution-id>` canonical paths.
- Responses use `no-store`, deny framing, disable indexing, and carry a same-origin content security policy.
