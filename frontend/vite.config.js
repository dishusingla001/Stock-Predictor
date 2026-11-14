import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
const target = process.env.STOCKSIGHT_API_URL || 'http://127.0.0.1:8000';
const proxy = { '/api': { target, changeOrigin: true }, '/health': { target, changeOrigin: true } };
export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
});
