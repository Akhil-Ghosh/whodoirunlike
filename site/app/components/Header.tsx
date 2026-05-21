import Image from "next/image";
import { List, MagnifyingGlass } from "@phosphor-icons/react/dist/ssr";

const navItems = ["Studies", "Athletes", "About", "Journal"];

export function Header() {
  return (
    <header className="h-20 bg-[rgba(251,250,247,0.88)] lg:h-24">
      <div className="mx-auto grid h-full max-w-[1530px] grid-cols-[minmax(190px,1fr)_auto] items-center gap-4 px-5 sm:px-8 lg:grid-cols-[minmax(300px,1fr)_auto_minmax(260px,1fr)] lg:px-10">
        <a className="focus-ring block w-[224px] sm:w-[270px] lg:w-[300px]" href="#" aria-label="Who Do I Run Like home">
          <Image
            src="/assets/brand/logo-wordmark.svg"
            alt="Who Do I Run Like"
            width={720}
            height={90}
            priority
            className="h-auto w-full"
          />
        </a>

        <nav className="hidden items-center gap-16 text-[15px] font-medium text-[var(--ink)] lg:flex xl:gap-[74px]" aria-label="Primary">
          {navItems.map((item) => (
            <a className="focus-ring rounded-full transition-opacity duration-300 hover:opacity-55" href={`#${item.toLowerCase()}`} key={item}>
              {item}
            </a>
          ))}
        </nav>

        <div className="flex justify-end gap-3">
          <button
            className="focus-ring hidden h-11 w-14 place-items-center rounded-full border border-[rgba(23,23,25,0.14)] bg-white/50 text-[var(--ink)] transition duration-300 hover:bg-white active:translate-y-px sm:grid lg:h-[54px] lg:w-[62px]"
            type="button"
            aria-label="Search"
          >
            <MagnifyingGlass size={23} weight="regular" />
          </button>
          <button
            className="focus-ring inline-grid h-11 grid-flow-col place-items-center gap-2 rounded-full bg-[var(--charcoal)] px-5 text-[14px] font-medium text-white shadow-[0_12px_35px_rgba(20,20,20,0.10)] transition duration-300 hover:bg-[#202124] active:translate-y-px lg:h-[54px] lg:px-6"
            type="button"
          >
            <span>Menu</span>
            <List size={22} weight="regular" />
          </button>
        </div>
      </div>
    </header>
  );
}
