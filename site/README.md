# Who Do I Run Like Site

Next.js technical-preview UI for the running-form CV pipeline.

The page is intentionally demo-first:

- hero comparison visual
- featured four-stage walkthrough from a processed clip
- upload card that calls the FastAPI service at `NEXT_PUBLIC_API_BASE_URL`

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

## Build

```bash
npm run typecheck
npm run build
```

Assets under `public/assets/demos/` are web-sized derivatives of local CV artifacts. The larger source artifacts stay outside git under `artifacts/`.
