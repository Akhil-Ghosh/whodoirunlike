import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { ArrowRight, PlayCircle } from "@phosphor-icons/react/dist/ssr";
import { Header } from "../components/Header";
import { RunnerGallery } from "../components/RunnerGallery";
import { analyzedRunners } from "../data/analyzedRunners";

export const metadata: Metadata = {
  title: "Gallery | Who Do I Run Like",
  description: "Runner stride notes, metrics, and processed clip outputs from the Who Do I Run Like preview.",
};

export default function GalleryPage() {
  const processedClipCount = analyzedRunners.reduce((total, runner) => total + runner.clipCount, 0);
  const featuredRunner = analyzedRunners[0];
  const wallPlacements = [
    "w-[176px] sm:w-[214px] lg:w-[184px] lg:-translate-x-8 lg:translate-y-8",
    "w-[212px] sm:w-[262px] lg:w-[224px] lg:-translate-y-5",
    "w-[246px] sm:w-[310px] lg:w-[274px] lg:translate-y-6",
    "w-[202px] sm:w-[246px] lg:w-[210px] lg:-translate-y-4",
    "w-[184px] sm:w-[226px] lg:w-[196px] lg:translate-x-8 lg:translate-y-9",
  ];
  const wallAspects = ["aspect-[0.72]", "aspect-[0.82]", "aspect-[0.76]", "aspect-[0.9]", "aspect-[0.74]"];

  return (
    <main className="min-h-[100dvh] overflow-x-hidden bg-[var(--paper)]">
      <Header />

      <section className="bg-[#080806] text-white" aria-labelledby="gallery-title">
        <div className="mx-auto max-w-[1416px] px-5 pb-12 pt-8 sm:px-8 lg:px-8 lg:pb-16 lg:pt-10 2xl:px-0">
          <div className="grid gap-5 border-t border-white/12 pt-5 lg:grid-cols-[minmax(0,0.7fr)_minmax(260px,0.3fr)] lg:items-end">
            <h1 id="gallery-title" className="max-w-[760px] text-[58px] font-medium leading-[0.92] tracking-[-0.04em] text-white sm:text-[86px] lg:text-[112px]">
              Runner clip room.
            </h1>
            <p className="max-w-[360px] text-[16px] leading-[1.55] text-white/66 lg:justify-self-end">
              Pick a runner and inspect source clips, isolation masks, skeletons, and stride notes.
            </p>
          </div>

          <div className="mt-8 rounded-lg bg-[#f3f1ec] p-2 text-[var(--ink)] shadow-[0_34px_90px_rgba(0,0,0,0.34)]">
            <div className="relative overflow-hidden rounded-md border border-black/12 bg-[#f7f5f0]">
              <div className="relative flex h-11 items-center gap-4 border-b border-black/10 px-4 text-[10px] font-semibold uppercase tracking-[0.08em] text-black/62">
                <span>index</span>
                <span>runners</span>
                <span>clips</span>
                <span className="absolute left-1/2 top-1/2 hidden -translate-x-1/2 -translate-y-1/2 text-[16px] font-semibold normal-case tracking-[-0.03em] text-black/78 sm:block">
                  Who Do I Run Like
                </span>
                <span className="ml-auto">{processedClipCount} processed clips</span>
              </div>

              <div className="relative min-h-[440px] overflow-hidden sm:min-h-[500px] lg:min-h-[548px]">
                <div className="pointer-events-none absolute inset-x-0 top-1/2 h-px bg-black/8" />
                <div className="pointer-events-none absolute inset-x-0 bottom-24 h-px bg-black/8" />
                <div className="-mx-3 flex min-h-[334px] snap-x items-center gap-5 overflow-x-auto px-5 py-10 sm:-mx-5 sm:min-h-[390px] sm:px-8 lg:mx-0 lg:min-h-[430px] lg:justify-center lg:gap-7 lg:overflow-visible lg:px-4">
                  {analyzedRunners.map((runner, index) => (
                    <Link
                      className={`focus-ring group relative shrink-0 snap-center transition duration-300 hover:-translate-y-2 active:translate-y-px ${wallPlacements[index] ?? wallPlacements[0]}`}
                      href={`/gallery/${runner.slug}`}
                      key={runner.slug}
                    >
                      <span className={`relative block overflow-hidden rounded-md bg-[#e5ded4] shadow-[0_18px_36px_rgba(20,16,12,0.18)] ${wallAspects[index] ?? wallAspects[0]}`}>
                        <Image
                          alt={runner.imageAlt}
                          className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                          fill
                          priority={runner.slug === featuredRunner.slug}
                          sizes="(min-width: 1024px) 260px, (min-width: 640px) 310px, 246px"
                          src={runner.image}
                        />
                        <span className="absolute inset-0 bg-gradient-to-t from-black/72 via-black/5 to-transparent opacity-80 transition duration-300 group-hover:opacity-95" />
                        <span className="absolute bottom-3 left-3 right-3 translate-y-2 opacity-0 transition duration-300 group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100">
                          <span className="block text-[14px] font-semibold leading-none text-white">{runner.name}</span>
                          <span className="mt-2 block text-[11px] leading-[1.35] text-white/78">{runner.cardBlurb}</span>
                        </span>
                      </span>
                    </Link>
                  ))}
                </div>

                <div className="absolute inset-x-0 bottom-0 grid grid-cols-[1fr_auto_1fr] items-end gap-4 px-5 pb-6 sm:px-8">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-black/46">runners</p>
                    <p className="mt-1 text-[58px] font-medium leading-none tracking-[-0.06em] text-black/54 sm:text-[76px]">05</p>
                  </div>
                  <Link
                    className="focus-ring mb-2 hidden items-center gap-2 rounded-full border border-black/14 bg-white/74 px-4 py-2 text-[13px] font-medium text-[var(--ink)] transition duration-300 hover:bg-white active:translate-y-px sm:inline-flex"
                    href={`/gallery/${featuredRunner.slug}`}
                  >
                    Featured walkthrough
                    <PlayCircle size={17} weight="regular" />
                  </Link>
                  <div className="text-right">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-black/46">clips</p>
                    <p className="mt-1 text-[58px] font-medium leading-none tracking-[-0.06em] text-black/54 sm:text-[76px]">
                      {String(processedClipCount).padStart(2, "0")}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 flex flex-wrap items-center justify-between gap-3 text-[13px] text-white/58">
            <span>Current room: five runners, six processed clips.</span>
            <Link className="focus-ring inline-flex items-center gap-2 font-medium text-white transition hover:text-white/78" href={`/gallery/${featuredRunner.slug}`}>
              Open Cole Hocker
              <ArrowRight size={15} weight="regular" />
            </Link>
          </div>
        </div>
      </section>

      <RunnerGallery runners={analyzedRunners} />
    </main>
  );
}
