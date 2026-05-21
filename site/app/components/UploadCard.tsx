"use client";

import { ArrowRight, CheckCircle, LockKey, UploadSimple, WarningCircle } from "@phosphor-icons/react";
import { motion } from "framer-motion";
import Image from "next/image";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

const maxBytes = 20 * 1024 * 1024;

type UploadState = "idle" | "ready" | "error" | "loading" | "complete";

export function UploadCard() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const timerRef = useRef<number | null>(null);
  const [fileName, setFileName] = useState("");
  const [status, setStatus] = useState<UploadState>("idle");
  const [message, setMessage] = useState("MP4, MOV or WebM, max 20MB");
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  function acceptFile(file?: File) {
    if (!file) return;

    if (file.size > maxBytes) {
      setFileName("");
      setStatus("error");
      setMessage("Choose a clip under 20MB.");
      return;
    }

    setFileName(file.name);
    setStatus("ready");
    setMessage("Ready to compare against the athlete database.");
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function handleAnalyze() {
    if (!fileName) {
      setStatus("error");
      setMessage("Upload a short running clip first.");
      return;
    }

    setStatus("loading");
    setMessage("Preparing form comparison preview.");
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      setStatus("complete");
      setMessage("Preview queued. Full analysis connects next.");
    }, 1100);
  }

  const isError = status === "error";
  const isPositive = status === "ready" || status === "complete";

  return (
    <motion.aside
      className="upload-glass relative mx-auto w-full max-w-[360px] rounded-[24px] border border-[rgba(27,27,28,0.07)] bg-white/82 p-6 text-[var(--ink)] backdrop-blur-xl sm:p-8 lg:max-w-[300px] 2xl:max-w-[306px] 2xl:p-[29px]"
      aria-label="Try it yourself"
      whileHover={{ y: -3 }}
      transition={{ type: "spring", stiffness: 140, damping: 20 }}
    >
      <p className="mb-4 text-[11px] font-bold uppercase text-[var(--accent)]">Try It Yourself</p>
      <h2 className="max-w-[230px] text-[31px] font-medium leading-[0.98] sm:text-[33px]">
        See Who You
        <br />
        Run Like
      </h2>
      <Image
        className="absolute right-12 top-14 w-9 sm:right-14 sm:top-14 sm:w-10"
        src="/assets/ui/accent-rays.svg"
        alt=""
        width={52}
        height={44}
        aria-hidden="true"
      />
      <p className="mt-3.5 hidden text-[14px] leading-[1.46] text-[var(--muted)] sm:block">
        Upload a short video of you running and get your top athlete matches.
      </p>

      <label
        className={[
          "focus-ring mt-5 grid min-h-[116px] place-items-center rounded-[18px] border border-dashed p-4 text-center transition duration-300 sm:min-h-[148px] sm:p-5",
          isDragging ? "border-[var(--accent)] bg-[#f8f1e8]" : "border-[#d8cfc2] bg-[#faf7f2]/70",
          isError ? "border-[#b77265] bg-[#fff8f6]" : "",
        ].join(" ")}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className="sr-only"
          type="file"
          accept="video/mp4,video/quicktime,video/webm"
          onChange={handleChange}
        />
        <span className="grid h-11 w-11 place-items-center rounded-full border border-[#d9cec0] bg-white/60 text-[var(--accent)]">
          {isError ? <WarningCircle size={24} weight="regular" /> : isPositive ? <CheckCircle size={24} weight="regular" /> : <UploadSimple size={24} weight="regular" />}
        </span>
        <span className="mt-3 block max-w-full truncate text-[15px] font-medium">
          {fileName || "Upload video"}
        </span>
        <span className={["mt-1 block text-[12px]", isError ? "text-[#935548]" : "text-[var(--muted)]"].join(" ")} aria-live="polite">
          {message}
        </span>
      </label>

      <motion.button
        className="focus-ring mt-5 flex h-[52px] w-full items-center justify-center gap-4 rounded-full bg-[var(--charcoal)] px-5 text-[15px] font-medium text-white shadow-[inset_0_-1px_0_rgba(255,255,255,0.15),0_12px_28px_rgba(0,0,0,0.16)] transition-colors duration-300 hover:bg-[#202124] sm:h-[58px]"
        type="button"
        onClick={handleAnalyze}
        whileTap={{ scale: 0.98, y: 1 }}
        transition={{ type: "spring", stiffness: 220, damping: 24 }}
      >
        <span>{status === "loading" ? "Preparing..." : "Analyze My Run"}</span>
        <ArrowRight size={21} weight="regular" />
      </motion.button>

      <p className="mt-4 flex items-center gap-2 text-[12px] text-[var(--muted)]">
        <LockKey size={15} weight="regular" className="text-[var(--accent)]" />
        Your data is private and secure.
      </p>
    </motion.aside>
  );
}
