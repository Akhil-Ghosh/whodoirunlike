"use client";

import Image from "next/image";
import {
  CSSProperties,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  useLayoutEffect,
  useRef,
} from "react";

type RevealStyle = CSSProperties & {
  "--divider-opacity": number;
  "--jakob-note-opacity": number;
  "--reveal-x": string;
  "--you-note-opacity": number;
};

const initialRevealStyle: RevealStyle = {
  "--divider-opacity": 0,
  "--jakob-note-opacity": 0,
  "--reveal-x": "0px",
  "--you-note-opacity": 1,
};

export function CompareRunner() {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const ratioRef = useRef(0);

  function setReveal(clientX: number) {
    const stage = stageRef.current;
    if (!stage) return;

    const rect = stage.getBoundingClientRect();
    const raw = clientX - rect.left;
    const clamped = Math.max(0, Math.min(rect.width, raw));
    setRevealVars(stage, clamped, rect.width);
  }

  function syncRevealWidth() {
    const stage = stageRef.current;
    if (!stage) return;

    const width = stage.getBoundingClientRect().width;
    const x = width * ratioRef.current;
    setRevealVars(stage, x, width);
  }

  function setRevealVars(stage: HTMLDivElement, x: number, width: number) {
    const ratio = width > 0 ? x / width : 0;
    ratioRef.current = ratio;
    stage.style.setProperty("--reveal-x", `${x}px`);
    stage.style.setProperty("--divider-opacity", x > 1 ? "1" : "0");
    stage.style.setProperty("--you-note-opacity", ratio <= 0.5 ? "1" : "0");
    stage.style.setProperty("--jakob-note-opacity", ratio >= 0.5 ? "1" : "0");
  }

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;

    const trackPointer = (event: globalThis.PointerEvent) => {
      setReveal(event.clientX);
    };

    syncRevealWidth();
    window.addEventListener("pointermove", trackPointer, { passive: true });

    const observer = new ResizeObserver(syncRevealWidth);
    observer.observe(stage);

    return () => {
      window.removeEventListener("pointermove", trackPointer);
      observer.disconnect();
    };
  }, []);

  function handlePointer(event: ReactPointerEvent<HTMLDivElement>) {
    setReveal(event.clientX);
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    setReveal(event.clientX);
  }

  function handleMouse(event: ReactMouseEvent<HTMLDivElement>) {
    setReveal(event.clientX);
  }

  return (
    <div
      ref={stageRef}
      data-testid="compare-stage"
      className="focus-ring relative mx-auto h-[300px] w-full max-w-[620px] touch-none select-none rounded-[2px] sm:h-[430px] lg:h-[632px] lg:max-w-none"
      style={initialRevealStyle}
      onPointerMove={handlePointer}
      onPointerDown={handlePointerDown}
      onMouseMove={handleMouse}
      onMouseDown={handleMouse}
      onClick={handleMouse}
      aria-label="Runner comparison reveal"
    >
      <div className="runner-ground absolute bottom-[7%] left-[12%] h-8 w-[76%]" aria-hidden="true" />

      <div
        className="pointer-events-none absolute inset-0 will-change-[clip-path]"
        style={{ clipPath: "inset(0 0 0 var(--reveal-x))" }}
        data-testid="runner-gray-base"
      >
        <Image
          src="/assets/hero/runner-gray.png"
          alt="Gray comparison runner"
          width={1024}
          height={1536}
          priority
          className="absolute left-1/2 top-1/2 h-[105%] w-auto max-w-none -translate-x-1/2 -translate-y-1/2 object-contain lg:h-[104%]"
        />
      </div>

      <div
        className="pointer-events-none absolute inset-0 opacity-[var(--you-note-opacity)] transition-opacity duration-200 ease-out will-change-[opacity]"
        data-testid="runner-gray-note"
      >
        <p className="hand-font absolute left-[12%] top-[9%] text-[22px] leading-none text-[var(--ink)] lg:left-[5%] lg:top-[11%] lg:text-[23px]">
          You
          <Image
            src="/assets/ui/hand-arrow-you.svg"
            alt=""
            width={160}
            height={90}
            className="ml-8 mt-1 w-[108px]"
            aria-hidden="true"
          />
        </p>
      </div>

      <div
        className="pointer-events-none absolute inset-0 will-change-[clip-path]"
        style={{ clipPath: "inset(0 calc(100% - var(--reveal-x)) 0 0)" }}
        data-testid="runner-color-reveal"
      >
        <Image
          src="/assets/hero/runner-color.png"
          alt="Color comparison runner"
          width={1024}
          height={1536}
          priority
          className="absolute left-1/2 top-1/2 h-[105%] w-auto max-w-none -translate-x-1/2 -translate-y-1/2 object-contain lg:h-[104%]"
        />
      </div>

      <div
        className="pointer-events-none absolute inset-0 opacity-[var(--jakob-note-opacity)] transition-opacity duration-200 ease-out will-change-[opacity]"
        data-testid="runner-color-note"
      >
        <p className="hand-font absolute right-[3%] top-[10%] text-center text-[22px] leading-[1.08] text-[var(--ink)] lg:right-[-3%] lg:top-[12%] lg:text-[23px]">
          Jakob
          <br />
          Ingebrigtsen
          <Image
            src="/assets/ui/hand-arrow-athlete.svg"
            alt=""
            width={160}
            height={90}
            className="-ml-8 mt-1 w-[112px]"
            aria-hidden="true"
          />
        </p>
      </div>

      <div
        className="pointer-events-none absolute inset-y-0 left-0 will-change-transform"
        style={{
          opacity: "var(--divider-opacity)",
          transform: "translate3d(var(--reveal-x), 0, 0)",
        }}
        aria-hidden="true"
        data-testid="runner-reveal-divider"
      >
        <div className="h-[96%] w-px -translate-x-px bg-[linear-gradient(to_bottom,rgba(23,23,25,0),rgba(23,23,25,0.13)_14%,rgba(23,23,25,0.13)_86%,rgba(23,23,25,0))]" />
      </div>
    </div>
  );
}
