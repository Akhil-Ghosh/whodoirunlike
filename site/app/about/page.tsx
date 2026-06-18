import { ArrowRight, PlayCircle } from "@phosphor-icons/react/dist/ssr";
import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { Header } from "../components/Header";
import { analyzedRunners } from "../data/analyzedRunners";

export const metadata: Metadata = {
  title: "About | Who Do I Run Like",
  description: "Why Who Do I Run Like exists and what the technical preview can do today.",
};

const ideaChips = ["video first", "pose over vibes", "volunteer only", "matching later", "reviewable outputs"];

const previewStats = [
  { label: "Analyzed runners", value: `${analyzedRunners.length}` },
  { label: "Processed clips", value: `${analyzedRunners.reduce((total, runner) => total + runner.clipCount, 0)}` },
  { label: "Primary output", value: "Stride metrics" },
];

const aboutBlocks = [
  {
    title: "Why video",
    body: "Pace and splits are useful, but they flatten how someone moves. Video keeps the messy parts: posture, rhythm, arms, knees, camera motion, and people blocking the view.",
  },
  {
    title: "What exists now",
    body: "A clip can move through the backend and come back with runner isolation, a pose skeleton, quality checks, and feature files that describe the stride.",
  },
  {
    title: "What it is not",
    body: "It is not a coach, a diagnosis tool, or a serious matcher yet. The matching layer needs more clean examples before it can make honest comparisons.",
  },
];

export default function AboutPage() {
  return (
    <main className="min-h-[100dvh] overflow-x-hidden bg-[var(--paper)]">
      <Header />

      <section className="hero-field" aria-labelledby="about-title">
        <div className="mx-auto grid max-w-[1416px] gap-8 px-5 pb-10 pt-8 sm:px-8 lg:grid-cols-[minmax(0,0.9fr)_minmax(360px,0.72fr)] lg:items-center lg:px-8 lg:pb-14 lg:pt-10 2xl:px-0">
          <div>
            <h1 id="about-title" className="max-w-[760px] text-[54px] font-medium leading-[0.96] text-[var(--ink)] sm:text-[76px] lg:text-[88px]">
              A running-form demo before a matching product.
            </h1>
            <p className="mt-6 max-w-[620px] text-[18px] leading-[1.55] text-[#222326]">
              I started Who Do I Run Like because running form lives in footage, not in a pace chart. The current version is about making that footage readable.
            </p>

            <div className="mt-7 flex flex-wrap gap-2">
              {ideaChips.map((chip) => (
                <span className="rounded-full border border-[rgba(23,23,25,0.14)] bg-white/62 px-4 py-2 text-[13px] font-medium text-[var(--ink)]" key={chip}>
                  {chip}
                </span>
              ))}
            </div>
          </div>

          <div className="overflow-hidden rounded-lg border border-[rgba(23,23,25,0.1)] bg-white shadow-[0_24px_70px_rgba(28,24,18,0.12)]">
            <div className="relative aspect-[1.12] bg-[#ece7dd]">
              <Image
                alt="Cole Hocker finishing ahead of Josh Kerr and Jakob Ingebrigtsen"
                className="h-full w-full object-cover"
                fill
                priority
                sizes="(min-width: 1024px) 460px, 92vw"
                src="/assets/gallery/runners/cole-hocker-finish.webp"
              />
            </div>
            <div className="grid gap-3 p-5">
              <p className="text-[15px] leading-[1.45] text-[var(--muted)]">
                The question is playful. The work underneath is plain computer vision: pick a runner, read the pose, and inspect the result.
              </p>
              <Link
                className="focus-ring inline-flex h-11 w-fit items-center gap-2 rounded-full bg-[var(--charcoal)] px-4 text-[14px] font-medium text-white transition duration-300 hover:bg-[#202124] active:translate-y-px"
                href="/gallery/cole-hocker"
              >
                See Cole's clips
                <ArrowRight size={17} weight="regular" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-[var(--paper)]">
        <div className="mx-auto grid max-w-[1416px] gap-4 px-5 py-10 sm:px-8 lg:grid-cols-12 lg:px-8 lg:py-14 2xl:px-0">
          <article className="rounded-lg bg-[var(--charcoal)] p-6 text-white lg:col-span-5 lg:p-8">
            <h2 className="max-w-[460px] text-[40px] font-medium leading-[1.02] sm:text-[52px]">
              The first job is to make the clip inspectable.
            </h2>
            <p className="mt-6 text-[17px] leading-[1.6] text-white/72">
              If the pipeline cannot show what it saw, the matcher does not matter. That is why the preview centers on isolation clips, skeleton renders, and stride metrics.
            </p>
          </article>

          <article className="overflow-hidden rounded-lg bg-[#111214] lg:col-span-7">
            <div className="relative aspect-video bg-black">
              <video className="h-full w-full object-cover" src="/assets/demos/cole-fused.mp4" autoPlay muted loop playsInline preload="metadata" />
              <div className="pointer-events-none absolute left-4 top-4 grid h-10 w-10 place-items-center rounded-full bg-black/55 text-white backdrop-blur">
                <PlayCircle size={23} weight="regular" />
              </div>
            </div>
            <div className="grid gap-2 p-5 sm:grid-cols-3">
              {previewStats.map((stat) => (
                <div className="rounded-md bg-white/[0.06] px-4 py-4" key={stat.label}>
                  <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-[#d0ae82]">{stat.label}</p>
                  <p className="mt-2 text-[24px] font-medium leading-none text-white">{stat.value}</p>
                </div>
              ))}
            </div>
          </article>

          <article className="rounded-lg bg-white p-6 shadow-[0_18px_54px_rgba(28,24,18,0.08)] lg:col-span-4 lg:p-7">
            <h2 className="text-[28px] font-medium leading-tight text-[var(--ink)]">Consent stays explicit.</h2>
            <p className="mt-4 text-[16px] leading-[1.55] text-[var(--muted)]">
              If someone shares a clip, it should be because they want to help test the pipeline. The upload flow says that directly.
            </p>
          </article>

          <article className="rounded-lg border border-[rgba(23,23,25,0.1)] bg-[var(--paper-soft)] p-6 lg:col-span-8 lg:p-7">
            <h2 className="max-w-[620px] text-[34px] font-medium leading-[1.08] text-[var(--ink)]">
              The matcher comes after the data earns it.
            </h2>
            <p className="mt-5 max-w-[760px] text-[17px] leading-[1.6] text-[var(--muted)]">
              Similarity search needs enough clean examples to avoid pretending. Right now, the honest product is the technical preview: upload a short clip, process it, and review what the model extracted.
            </p>
          </article>
        </div>
      </section>

      <section className="demo-band text-white">
        <div className="mx-auto max-w-[1416px] px-5 py-10 sm:px-8 lg:px-8 lg:py-12 2xl:px-0">
          <div className="grid gap-4 md:grid-cols-[1.1fr_0.9fr_1fr]">
            {aboutBlocks.map((block) => (
              <article className="rounded-lg border border-white/10 bg-white/[0.035] p-5" key={block.title}>
                <h2 className="text-[24px] font-medium leading-tight">{block.title}</h2>
                <p className="mt-4 text-[15px] leading-[1.55] text-white/72">{block.body}</p>
              </article>
            ))}
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-white/10 pt-6">
            <p className="max-w-[560px] text-[15px] leading-[1.5] text-white/66">
              The gallery shows the clips that have actually gone through the pipeline. It is small on purpose.
            </p>
            <Link
              className="focus-ring inline-flex h-11 items-center gap-2 rounded-full bg-white px-4 text-[14px] font-medium text-[var(--ink)] transition duration-300 hover:bg-[#f2eadf] active:translate-y-px"
              href="/gallery"
            >
              Open gallery
              <ArrowRight size={17} weight="regular" />
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
