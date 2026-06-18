# Who Do I Run Like Site

Next.js technical-preview UI for the running-form CV pipeline.

The page is intentionally demo-first:

- hero comparison visual
- featured four-stage walkthrough from a processed clip
- upload card that calls the local FastAPI service in dev and the Worker job API in production

## Run Locally

Start the API from the repo root:

```bash
uvicorn whodoirunlike.api:app --host 127.0.0.1 --port 8000
```

Then start the site:

```bash
npm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://127.0.0.1:4173`.

To test the async Worker flow locally, run the Worker on port 8787 and start the site with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8787 NEXT_PUBLIC_UPLOAD_API_MODE=async npm run dev
```

## Build

```bash
npm run typecheck
npm run build
```

Assets under `public/assets/demos/` are web-sized derivatives of local CV artifacts. The larger source artifacts stay outside git under `artifacts/`.

## Cloudflare Pages

Use `whodoirunlike.com` as the production domain.

Dashboard settings:

```text
Root directory: site
Build command: npm run build:pages
Build output directory: out
Production branch: main
Node version: 24
Environment variable:
  NEXT_PUBLIC_API_BASE_URL=https://api.whodoirunlike.com
  NEXT_PUBLIC_UPLOAD_API_MODE=async
```

Direct deploy:

```bash
npm run pages:deploy
```
