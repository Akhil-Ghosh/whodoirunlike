# Who Do I Run Like landing app

This folder contains the Next.js landing page prototype plus the image, SVG, and token
assets needed to match the reference mockup.

## Running locally

```bash
npm install
npm run dev
```

Then open `http://127.0.0.1:4173`.

## Hero overlay behavior

The hero comparison uses the gray runner as the base image and clips the color runner on top.
The interactive wipe is isolated in `app/components/CompareRunner.tsx`.

## Assets

Assets are served from `public/assets`, so component paths should use `/assets/...`.

## Notes before production

The athlete images and likenesses are generated assets. Review rights, approvals, copy, and factual labels before public launch.
