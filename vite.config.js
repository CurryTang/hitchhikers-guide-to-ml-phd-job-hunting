import { env } from 'node:process';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const repoName = env.GITHUB_REPOSITORY?.split('/')[1];
const fallbackBase = '/hitchhikers-guide-to-ml-phd-job-hunting/';
const inferredProductionBase = repoName ? `/${repoName}/` : fallbackBase;

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  base: env.VITE_BASE_PATH || (command === 'build' ? inferredProductionBase : '/'),
  plugins: [react()],
}));
