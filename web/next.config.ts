import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  httpAgentOptions: {
    keepAlive: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
};

export default nextConfig;
