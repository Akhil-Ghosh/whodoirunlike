import { ArrowLeft, ArrowSquareOut, PlayCircle } from "@phosphor-icons/react/dist/ssr";
import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Header } from "../../components/Header";
import { analyzedRunners } from "../../data/analyzedRunners";

type RunnerPageProps = {
  params: Promise<{ slug: string }>;
};

export const dynamicParams = false;

function getRunner(slug: string) {
  return analyzedRunners.find((runner) => runner.slug === slug);
}

export function generateStaticParams() {
  return analyzedRunners.map((runner) => ({ slug: runner.slug }));
}

export async function generateMetadata({ params }: RunnerPageProps): Promise<Metadata> {
  const { slug } = await params;
  const runner = getRunner(slug);

  if (!runner) {
    return {
      title: "Runner not found | Who Do I Run Like",
    };
  }

  return {
    title: `${runner.name} | Who Do I Run Like`,
    description: `${runner.name} stride notes, derived metrics, and processed clip outputs.`,
  };
}

export default async function RunnerPage({ params }: RunnerPageProps) {
  const { slug } = await params;
  const runner = getRunner(slug);

  if (!runner) {
    notFound();
  }

  const otherRunners = analyzedRunners.filter((item) => item.slug !== runner.slug);

  return (
    <main className="min-h-[100dvh] overflow-x-hidden bg-[var(--paper)]">
      <Header />

      <section className="demo-band text-white" aria-labelledby="runner-title">
        <div className="mx-auto grid max-w-[1416px] gap-8 px-5 py-8 sm:px-8 lg:grid-cols-[minmax(0,1fr)_440px] lg:items-center lg:px-8 lg:py-12 2xl:px-0">
          <div>
            <Link
              className="focus-ring mb-8 inline-flex h-11 items-center gap-2 rounded-full border border-white/12 bg-white/[0.06] px-4 text-[14px] font-medium text-white/82 transition duration-300 hover:bg-white/[0.1] active:translate-y-px"
              href="/gallery"
            >
              <ArrowLeft size={17} weight="regular" />
              Gallery
            </Link>

            <p className="text-[12px] font-bold uppercase tracking-[0.08em] text-[#d0ae82]">{runner.event}</p>
            <h1 id="runner-title" className="mt-5 max-w-[820px] text-[54px] font-medium leading-[0.96] sm:text-[74px] lg:text-[86px]">
              {runner.name}
            </h1>
            <p className="mt-6 max-w-[650px] text-[18px] leading-[1.55] text-white/76">{runner.detailBlurb}</p>
          </div>

          <div className="relative overflow-hidden rounded-lg border border-white/10 bg-white/[0.035]">
            <div className="relative aspect-[1.15] bg-[#17191d]">
              <Image
                alt={runner.imageAlt}
                className="h-full w-full object-cover"
                fill
                priority
                sizes="(min-width: 1024px) 440px, 92vw"
                src={runner.image}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/64 via-black/8 to-transparent" />
            </div>
            <div className="p-5">
              <p className="text-[18px] font-medium leading-[1.35]">{runner.stridePattern}</p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-[var(--paper)]">
        <div className="mx-auto grid max-w-[1416px] gap-6 px-5 py-10 sm:px-8 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)] lg:px-8 lg:py-14 2xl:px-0">
          <div className="max-w-[560px]">
            <h2 className="text-[40px] font-medium leading-[1.02] text-[var(--ink)] sm:text-[52px]">Stride readout</h2>
            <p className="mt-5 text-[17px] leading-[1.6] text-[var(--muted)]">
              These are clip-derived signals, not a coaching diagnosis. The point is to show what the pipeline can read and where the video still needs review.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            {runner.metricSummary.map((metric) => (
              <article className="rounded-lg bg-white px-5 py-5 shadow-[0_18px_48px_rgba(28,24,18,0.08)]" key={metric.label}>
                <p className="text-[12px] font-bold uppercase tracking-[0.08em] text-[var(--accent-deep)]">{metric.label}</p>
                <p className="mt-3 text-[34px] font-medium leading-none text-[var(--ink)]">{metric.value}</p>
                <p className="mt-4 text-[14px] leading-[1.45] text-[var(--muted)]">{metric.note}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-[var(--paper-soft)]" aria-labelledby="clip-outputs-title">
        <div className="mx-auto max-w-[1416px] px-5 py-10 sm:px-8 lg:px-8 lg:py-14 2xl:px-0">
          <h2 id="clip-outputs-title" className="max-w-[760px] text-[42px] font-medium leading-[1.02] text-[var(--ink)] sm:text-[58px]">
            Processed clips
          </h2>

          <div className="mt-7 grid gap-7">
            {runner.clips.map((clip) => (
              <article className="overflow-hidden rounded-lg bg-white shadow-[0_22px_70px_rgba(28,24,18,0.11)]" key={clip.label}>
                <div className="grid gap-0 lg:grid-cols-[0.85fr_1.15fr]">
                  <div className="p-5 sm:p-7 lg:p-8">
                    <div className="flex flex-wrap gap-2">
                      <span className="rounded-full border border-[rgba(23,23,25,0.12)] px-3 py-1.5 text-[12px] font-medium text-[var(--muted)]">
                        {clip.duration}
                      </span>
                      <span className="rounded-full border border-[rgba(23,23,25,0.12)] px-3 py-1.5 text-[12px] font-medium text-[var(--muted)]">
                        {clip.quality}
                      </span>
                    </div>

                    <h3 className="mt-5 text-[32px] font-medium leading-[1.05] text-[var(--ink)]">{clip.label}</h3>
                    <p className="mt-4 text-[16px] leading-[1.55] text-[var(--muted)]">{clip.context}</p>

                    <div className="mt-6 grid gap-3 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
                      {clip.metrics.map((metric) => (
                        <div className="rounded-md bg-[var(--paper-soft)] px-4 py-4" key={metric.label}>
                          <p className="text-[11px] font-bold uppercase tracking-[0.06em] text-[var(--accent-deep)]">{metric.label}</p>
                          <p className="mt-2 text-[25px] font-medium leading-none text-[var(--ink)]">{metric.value}</p>
                          <p className="mt-3 text-[12px] leading-[1.35] text-[var(--muted)]">{metric.note}</p>
                        </div>
                      ))}
                    </div>

                    <ul className="mt-6 grid gap-3">
                      {clip.strideNotes.map((note) => (
                        <li className="border-t border-[var(--line)] pt-3 text-[15px] leading-[1.5] text-[var(--muted)]" key={note}>
                          {note}
                        </li>
                      ))}
                    </ul>

                    <a
                      className="focus-ring mt-7 inline-flex h-11 items-center gap-2 rounded-full bg-[var(--charcoal)] px-4 text-[14px] font-medium text-white transition duration-300 hover:bg-[#202124] active:translate-y-px"
                      href={clip.sourceUrl}
                      rel="noreferrer"
                      target="_blank"
                    >
                      Source video
                      <ArrowSquareOut size={17} weight="regular" />
                    </a>
                  </div>

                  <div className="grid items-start gap-3 bg-[#111214] p-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                    {clip.outputs.map((output) => (
                      <figure className="self-start overflow-hidden rounded-md border border-white/10 bg-white/[0.035]" key={output.href}>
                        <div className="relative aspect-video bg-black">
                          <video className="h-full w-full object-cover" src={output.href} autoPlay muted loop playsInline preload="metadata" />
                          <div className="pointer-events-none absolute left-3 top-3 grid h-9 w-9 place-items-center rounded-full bg-black/55 text-white backdrop-blur">
                            <PlayCircle size={21} weight="regular" />
                          </div>
                        </div>
                        <figcaption className="p-4">
                          <p className="text-[15px] font-medium text-white">{output.label}</p>
                          <p className="mt-2 text-[13px] leading-[1.4] text-white/62">{output.note}</p>
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-[var(--paper)]">
        <div className="mx-auto max-w-[1416px] px-5 py-10 sm:px-8 lg:px-8 lg:py-12 2xl:px-0">
          <h2 className="text-[32px] font-medium leading-tight text-[var(--ink)]">More analyzed runners</h2>
          <div className="mt-5 flex flex-wrap gap-3">
            {otherRunners.map((item) => (
              <Link
                className="focus-ring inline-flex h-11 items-center rounded-full border border-[rgba(23,23,25,0.14)] bg-white/72 px-4 text-[14px] font-medium text-[var(--ink)] transition duration-300 hover:bg-white active:translate-y-px"
                href={`/gallery/${item.slug}`}
                key={item.slug}
              >
                {item.name}
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
