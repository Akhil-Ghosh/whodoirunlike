import { AthleteCard } from "./AthleteCard";
import { Reveal } from "./Reveal";

const athletes = [
  {
    name: "Mo Farah",
    image: "/assets/athletes/mo-farah-card.webp",
    lines: ["Distance Legend", "Olympic Gold Medalist"],
  },
  {
    name: "Jakob\nIngebrigtsen",
    image: "/assets/athletes/jakob-ingebrigtsen-card.webp",
    lines: ["1500m World Record Holder", "Olympic Champion"],
  },
  {
    name: "Sydney\nMcLaughlin",
    image: "/assets/athletes/sydney-mclaughlin-card.webp",
    lines: ["400m Hurdles World Record", "Olympic Champion"],
  },
];

export function AthleteStrip() {
  return (
    <section id="athletes" className="dark-band text-white lg:min-h-[calc(100dvh-728px)]" aria-label="Athlete database">
      <div className="mx-auto grid max-w-[1416px] grid-cols-1 gap-6 px-5 py-9 sm:px-8 lg:grid-cols-[274px_repeat(3,minmax(0,1fr))] lg:gap-7 lg:px-0 lg:py-[34px]">
        <Reveal className="pt-1" delay={0.04}>
          <p className="mb-5 text-[12px] font-bold uppercase text-[#d0ae82]">
            In Our Database
          </p>
          <h2 className="max-w-[270px] text-[29px] font-medium leading-[1.12] sm:text-[36px] lg:text-[32px]">
            Compare against
            <br />
            the world&apos;s best.
          </h2>
          <p className="mt-7 text-[18px] leading-none text-white/78">And thousands more.</p>
        </Reveal>

        {athletes.map((athlete, index) => (
          <AthleteCard athlete={athlete} index={index} key={athlete.name} />
        ))}
      </div>
    </section>
  );
}
