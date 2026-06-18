import Image from "next/image";
import Link from "next/link";

const navItems = [
  { label: "Demo", href: "/#demo" },
  { label: "Gallery", href: "/gallery" },
  { label: "About", href: "/about" },
];

export function Header() {
  return (
    <header className="bg-[rgba(251,250,247,0.9)]">
      <div className="mx-auto grid min-h-20 max-w-[1530px] grid-cols-1 items-center gap-4 px-5 py-4 sm:px-8 lg:min-h-24 lg:grid-cols-[minmax(300px,1fr)_auto_minmax(300px,1fr)] lg:px-10 lg:py-0">
        <Link className="focus-ring block w-[224px] sm:w-[270px] lg:w-[300px]" href="/" aria-label="Who Do I Run Like home">
          <Image
            src="/assets/brand/logo-wordmark.svg"
            alt="Who Do I Run Like"
            width={720}
            height={90}
            priority
            className="h-auto w-full"
          />
        </Link>

        <nav className="flex flex-wrap items-center justify-center gap-3 text-[14px] font-medium text-[var(--ink)] sm:gap-4 lg:gap-16 lg:text-[15px] xl:gap-[74px]" aria-label="Primary">
          {navItems.map((item) => (
            <Link
              className="focus-ring rounded-full border border-[rgba(23,23,25,0.1)] bg-white/48 px-4 py-2 transition duration-300 hover:border-[rgba(23,23,25,0.2)] hover:bg-white lg:border-transparent lg:bg-transparent lg:px-2"
              href={item.href}
              key={item.label}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="hidden lg:block" aria-hidden="true" />
      </div>
    </header>
  );
}
