import { ChartLineUp, Crosshair, PersonSimpleRun, Sparkle } from "@phosphor-icons/react/dist/ssr";
import { Reveal } from "./Reveal";

const stages = [
  {
    label: "Source Clip",
    title: "A short race segment enters the pipeline.",
    video: "/assets/demos/cole-source.mp4",
    icon: PersonSimpleRun,
  },
  {
    label: "Target Isolation",
    title: "The target runner is separated from the pack.",
    video: "/assets/demos/cole-isolation.mp4",
    icon: Crosshair,
  },
  {
    label: "Pose Sequence",
    title: "Landmarks are extracted frame by frame.",
    video: "/assets/demos/cole-skeleton.mp4",
    icon: ChartLineUp,
  },
  {
    label: "Form Signal",
    title: "Pose quality and motion features become a reviewable artifact.",
    video: "/assets/demos/cole-fused.mp4",
    icon: Sparkle,
  },
];

export function DemoWalkthrough() {
  return (
    <section id="demo" className="demo-band text-white" aria-label="Running form CV demo">
      <div className="mx-auto grid max-w-[1416px] gap-8 px-5 py-10 sm:px-8 lg:grid-cols-[330px_1fr] lg:px-8 lg:py-12 2xl:px-0">
        <Reveal className="lg:pt-2" delay={0.04}>
          <p className="mb-5 text-[12px] font-bold uppercase text-[#d0ae82]">
            Technical Preview
          </p>
          <h2 className="max-w-[330px] text-[36px] font-medium leading-[1.02] sm:text-[44px] lg:text-[42px]">
            One clip,
            <br />
            four artifacts.
          </h2>
          <p className="mt-6 max-w-[310px] text-[16px] leading-[1.55] text-white/74">
            The FastAPI service accepts a running clip, runs pose inference, and returns
            browser-playable artifacts plus JSON metrics.
          </p>
        </Reveal>

        <div className="grid gap-4 sm:grid-cols-2">
          {stages.map((stage, index) => {
            const Icon = stage.icon;
            return (
              <Reveal delay={0.08 + index * 0.04} key={stage.label}>
                <article className="demo-card overflow-hidden rounded-lg border border-white/10 bg-white/[0.035]">
                  <div className="aspect-video overflow-hidden bg-[#0b0c0e]">
                    <video
                      className="h-full w-full object-cover"
                      src={stage.video}
                      autoPlay
                      muted
                      loop
                      playsInline
                      preload="metadata"
                    />
                  </div>
                  <div className="grid grid-cols-[34px_1fr] gap-3 px-4 py-4">
                    <span className="grid h-8 w-8 place-items-center rounded-full bg-[#d0ae82]/14 text-[#d7b98d]">
                      <Icon size={18} weight="regular" />
                    </span>
                    <span>
                      <span className="block text-[12px] font-bold uppercase tracking-[0.08em] text-[#d0ae82]">
                        {stage.label}
                      </span>
                      <span className="mt-1 block text-[14px] leading-[1.35] text-white/82">
                        {stage.title}
                      </span>
                    </span>
                  </div>
                </article>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
