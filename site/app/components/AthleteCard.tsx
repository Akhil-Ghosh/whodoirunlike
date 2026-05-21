"use client";

import { ArrowUpRight } from "@phosphor-icons/react";
import { motion } from "framer-motion";
import Image from "next/image";

type Athlete = {
  name: string;
  image: string;
  lines: string[];
};

type AthleteCardProps = {
  athlete: Athlete;
  index: number;
};

export function AthleteCard({ athlete, index }: AthleteCardProps) {
  return (
    <motion.article
      className="group relative min-h-[205px] overflow-hidden rounded-lg border border-white/10 bg-[#151618] shadow-[0_22px_60px_rgba(0,0,0,0.28)] sm:min-h-[230px] lg:min-h-[194px]"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 120, damping: 22, delay: 0.16 + index * 0.08 }}
      whileHover={{ y: -4 }}
    >
      <Image
        src={athlete.image}
        alt=""
        fill
        priority={index < 3}
        sizes="(max-width: 1024px) 100vw, 354px"
        className="object-cover transition duration-700 group-hover:scale-[1.035]"
      />
      <div className="athlete-vignette absolute inset-0" />
      <div className="absolute left-[51%] right-[62px] top-8 sm:left-[52%] sm:right-[66px] lg:left-[51%] lg:right-[62px] 2xl:left-[52%] 2xl:right-[68px]">
        <h3 className="flex h-[46px] items-start whitespace-pre-line text-[20px] font-medium leading-[1.05] text-white sm:text-[22px] lg:text-[19px] 2xl:text-[23px]">
          {athlete.name}
        </h3>
        <div className="mt-2 grid gap-1.5">
          {athlete.lines.map((line) => (
            <p className="text-[12px] leading-[1.25] text-white/82 sm:text-[13px] 2xl:text-[14px]" key={line}>
              {line}
            </p>
          ))}
        </div>
      </div>
      <motion.a
        className="focus-ring absolute bottom-1 right-3 grid h-8 w-8 place-items-center rounded-full border border-[#d0ae82]/70 text-[#d7b98d] transition duration-300 group-hover:bg-[#d0ae82]/10 sm:bottom-1 sm:right-3"
        href="#athletes"
        aria-label={`Open ${athlete.name.replace("\n", " ")} profile`}
        whileTap={{ scale: 0.94 }}
      >
        <ArrowUpRight size={17} weight="regular" />
      </motion.a>
    </motion.article>
  );
}
