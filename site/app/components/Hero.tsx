import Image from "next/image";
import { CompareRunner } from "./CompareRunner";
import { Reveal } from "./Reveal";
import { UploadCard } from "./UploadCard";

const featureItems = [
  {
    icon: "/assets/icons/video.svg",
    title: "Upload a short running video",
    description: "Just 5-10 seconds",
  },
  {
    icon: "/assets/icons/analytics.svg",
    title: "We analyze 50+ movement patterns",
    description: "Stride, posture, arm swing, and more",
  },
  {
    icon: "/assets/icons/user-match.svg",
    title: "See your top matches",
    description: "Discover who you run most like, and why",
  },
];

export function Hero() {
  return (
    <section
      id="studies"
      className="hero-field relative overflow-hidden"
      aria-label="Running form comparison"
    >
      <div className="mx-auto grid max-w-[1416px] grid-cols-1 gap-8 px-5 pb-8 pt-8 sm:px-8 lg:h-[632px] lg:grid-cols-[370px_minmax(360px,1fr)_300px] lg:items-center lg:gap-8 lg:px-8 lg:py-0 2xl:grid-cols-[410px_580px_306px] 2xl:gap-[60px] 2xl:px-0">
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
          <p className="max-w-[420px] text-[17px] leading-[1.48] text-[#222326] sm:text-[18px]">
            Who Do I Run Like uses AI and computer vision to analyze your running form and compare it to the world&apos;s best athletes.
          </p>

          <ul className="mt-8 hidden gap-[22px] lg:grid" aria-label="How it works">
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
