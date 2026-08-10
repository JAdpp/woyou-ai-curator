import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-hosted deployment: emit a self-contained .next/standalone server so
  // the box runs `node server.js` without an npm install.
  output: "standalone",
};

export default nextConfig;
