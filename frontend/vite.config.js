import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// No proxy: the app calls the API on its real origin so the demo exercises the same
// CORS path a deployed browser client would. Override with VITE_API_BASE if the API
// is not on :8000.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
