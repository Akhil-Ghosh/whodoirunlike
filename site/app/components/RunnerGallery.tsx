import { ArrowUpRight } from "@phosphor-icons/react/dist/ssr";
import Image from "next/image";
import Link from "next/link";
import type { AnalyzedRunner } from "../data/analyzedRunners";

type RunnerGalleryProps = {
  runners: AnalyzedRunner[];
};

export function RunnerGallery({ runners }: RunnerGalleryProps) {
  const processedClipCount = runners.reduce((total, runner) => total + runner.clipCount, 0);

  return (
    <section className="bg-[var(--paper)]" aria-label="Analyzed runner clips">
      <div className="mx-auto max-w-[1416px] px-5 py-12 sm:px-8 lg:px-8 lg:py-16 2xl:px-0">
        <div className="grid gap-8 lg:grid-cols-[220px_minmax(0,1fr)] xl:grid-cols-[260px_minmax(0,1fr)]">
          <aside className="border-y border-[var(--line)] py-5 lg:sticky lg:top-24 lg:self-start">
            <p className="text-[32px] font-medium leading-none tracking-[-0.04em] text-[var(--ink)]">Index</p>
            <p className="mt-4 max-w-[36ch] text-[15px] leading-[1.5] text-[var(--muted)]">
              Choose a current study. Each page carries the source clip, processed outputs, and notes from the first pipeline pass.
            </p>
            <dl className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-1">
              <div>
                <dt className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--accent-deep)]">runners</dt>
                <dd className="mt-1 text-[26px] font-medium leading-none text-[var(--ink)]">{runners.length}</dd>
              </div>
              <div>
                <dt className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--accent-deep)]">clips</dt>
                <dd className="mt-1 text-[26px] font-medium leading-none text-[var(--ink)]">{processedClipCount}</dd>
              </div>
            </dl>
          </aside>

          <div className="divide-y divide-[var(--line)] border-y border-[var(--line)]">
          {runners.map((runner, index) => {
            return (
              <Link
                className="focus-ring group grid gap-5 py-6 transition duration-300 hover:bg-white/52 active:translate-y-px sm:px-3 md:grid-cols-[70px_minmax(0,1fr)_190px_40px] md:items-center lg:grid-cols-[86px_minmax(0,1fr)_236px_44px]"
                href={`/gallery/${runner.slug}`}
                key={runner.slug}
              >
                <span className="text-[34px] font-medium leading-none tracking-[-0.06em] text-[var(--accent-deep)] md:text-[44px]">
                  {String(index + 1).padStart(2, "0")}
                </span>

                <span className="min-w-0">
                  <span className="text-[12px] font-semibold uppercase tracking-[0.08em] text-[var(--accent-deep)]">{runner.event}</span>
                  <span className="mt-2 block text-[32px] font-medium leading-[0.98] tracking-[-0.04em] text-[var(--ink)] sm:text-[42px]">
                    {runner.name}
                  </span>
                  <span className="mt-3 block max-w-[620px] text-[15px] leading-[1.45] text-[var(--muted)]">{runner.cardBlurb}</span>

                  <span className="mt-4 flex flex-wrap gap-2">
                    {runner.metricSummary.map((metric) => (
                      <span className="rounded-full bg-white/70 px-3 py-1.5 text-[12px] text-[var(--muted)]" key={metric.label}>
                        <span className="font-semibold text-[var(--ink)]">{metric.label}:</span> {metric.value}
                      </span>
                    ))}
                  </span>
                </span>

                <span className="relative order-first block aspect-[1.45] overflow-hidden rounded-md bg-[#ece7dd] md:order-none md:aspect-[1.18]">
                  <Image
                    alt={runner.imageAlt}
                    className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.035]"
                    fill
                    priority={runner.slug === "cole-hocker"}
                    sizes="(min-width: 1024px) 236px, (min-width: 768px) 190px, 92vw"
                    src={runner.image}
                  />
                </span>

                <span className="inline-grid h-10 w-10 place-items-center rounded-full border border-[rgba(23,23,25,0.13)] text-[var(--ink)] transition duration-300 group-hover:bg-[var(--charcoal)] group-hover:text-white md:justify-self-end">
                  <ArrowUpRight size={17} weight="regular" />
                </span>
              </Link>
            );
          })}
          </div>
        </div>
      </div>
    </section>
  );
}
