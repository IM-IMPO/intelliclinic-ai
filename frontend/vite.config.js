import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

// The frontend is a static bundle that talks to the FastAPI backend over HTTP.
// In production both are served from the same origin, so API_BASE falls back to
// `${origin}/api/v1`. For local development against a backend on :8000, either
// set VITE_API_BASE_URL or use the dev proxy below.
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg"],
      manifest: {
        name: "IntelliClinic AI — Intelligent Clinical Assistant",
        short_name: "IntelliClinic",
        description:
          "Clinical documentation, validation and grounded question answering.",
        theme_color: "#0f1517",
        background_color: "#0f1517",
        display: "standalone",
        orientation: "any",
        start_url: ".",
        scope: ".",
        icons: [
          {
            src: "icon-192.png",
            sizes: "192x192",
            type: "image/png",
            purpose: "any maskable",
          },
          {
            src: "icon-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "any maskable",
          },
        ],
      },
      workbox: {
        // Never cache API responses: clinical data must always be fresh and a
        // stale cached answer would be a patient-safety problem.
        navigateFallbackDenylist: [/^\/api/],
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith("/api/"),
            handler: "NetworkOnly",
          },
        ],
      },
      devOptions: { enabled: false },
    }),
  ],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_PROXY_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
});
