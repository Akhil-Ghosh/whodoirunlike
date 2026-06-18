import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  turbopack: {
    root: process.cwd(),
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
