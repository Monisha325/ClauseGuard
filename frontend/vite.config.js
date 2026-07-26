import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Port 5173 is held by Docker Desktop/WSL relay processes on this
    // machine, not available for the dev server -- pinned explicitly to
    // 5174 with strictPort so a silent fallback to yet another port
    // never happens (the backend's CORS allow-list must match exactly).
    port: 5174,
    strictPort: true,
  },
});
