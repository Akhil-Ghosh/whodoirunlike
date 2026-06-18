import Image from "next/image";
import { CompareRunner } from "./CompareRunner";
import { Reveal } from "./Reveal";
import { UploadCard } from "./UploadCard";

const featureItems = [
  {
    icon: "/assets/icons/video.svg",
    title: "Submit a short running clip",
    description: "MP4, MOV, or WebM",
  },
  {
    icon: "/assets/icons/analytics.svg",
    title: "Extract pose and QA artifacts",
    description: "Landmarks, overlays, and metrics",
  },
  {
    icon: "/assets/icons/user-match.svg",
    title: "Compile form features",
    description: "The matching layer comes next",
  },
];

export function Hero() {
  return (
    <section
      id="studies"
      className="hero-field relative overflow-hidden"
      aria-label="Running form comparison"
    >
      <div className="mx-auto grid max-w-[1416px] grid-cols-1 gap-8 px-5 pb-8 pt-8 sm:px-8 lg:min-h-[684px] lg:grid-cols-[370px_minmax(360px,1fr)_300px] lg:items-center lg:gap-8 lg:px-8 lg:py-6 2xl:grid-cols-[410px_580px_306px] 2xl:gap-[60px] 2xl:px-0">
        <Reveal className="relative z-[1] lg:pt-3" delay={0.02}>
          <h1 className="text-[54px] font-medium leading-[0.96] text-[var(--ink)] sm:text-[68px] lg:text-[64px] 2xl:text-[76px]">
            <span className="block">Who Do You</span>
            <span className="hand-font block text-[var(--accent)]">Run Like?</span>
          </h1>
          <Image
            src="/assets/ui/scribble-underline.svg"
            alt=""
            width={430}
            height={36}
            className="mb-6 mt-1 w-[270px] sm:w-[330px]"
            aria-hidden="true"
          />
          <p className="mb-4 inline-flex rounded-full border border-[var(--line)] bg-white/70 px-4 py-2 text-[12px] font-bold uppercase tracking-[0.08em] text-[var(--accent-deep)]">
            Technical Preview
          </p>
          <p className="max-w-[420px] text-[17px] leading-[1.48] text-[#222326] sm:text-[18px]">
            Who Do I Run Like is a running-form computer vision pipeline: upload a clip,
            extract pose artifacts, and turn motion into searchable features.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <a
              className="focus-ring inline-flex h-12 items-center justify-center rounded-full bg-[var(--charcoal)] px-6 text-[14px] font-medium text-white shadow-[0_12px_35px_rgba(20,20,20,0.10)] transition duration-300 hover:bg-[#202124]"
              href="#demo"
            >
              Watch the demo
            </a>
            <a
              className="focus-ring inline-flex h-12 items-center justify-center rounded-full border border-[rgba(23,23,25,0.16)] bg-white/64 px-6 text-[14px] font-medium text-[var(--ink)] transition duration-300 hover:bg-white"
              href="#upload"
            >
              Volunteer a clip
            </a>
          </div>

          <ul className="mt-7 hidden gap-[18px] lg:grid" aria-label="How it works">
            {featureItems.map((item) => (
              <li className="grid grid-cols-[34px_1fr] items-start gap-4" key={item.title}>
                <Image src={item.icon} alt="" width={28} height={28} className="mt-0.5 h-7 w-7" aria-hidden="true" />
                <span>
                  <strong className="block text-[15px] font-semibold leading-tight text-[var(--ink)]">
                    {item.title}
                  </strong>
                  <small className="mt-1 block text-[14px] leading-tight text-[var(--muted)]">
                    {item.description}
                  </small>
                </span>
              </li>
            ))}
          </ul>
        </Reveal>

        <Reveal className="relative z-[1] min-h-[300px] sm:min-h-[430px] lg:min-h-0" delay={0.12}>
          <CompareRunner />
        </Reveal>

        <Reveal className="relative z-[2] lg:mt-5 2xl:mt-7" delay={0.2}>
          <UploadCard />
        </Reveal>
      </div>
    </section>
  );
}
