import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";

const geist = Geist({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-geist",
});

export const metadata: Metadata = {
  title: "Who Do I Run Like",
  description: "Upload a short running clip and compare your form against elite athletes.",
  icons: {
    icon: "/assets/brand/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geist.variable} min-h-[100dvh] bg-[var(--paper)] font-sans text-[var(--ink)] antialiased`}>
        {children}
      </body>
    </html>
  );
}
