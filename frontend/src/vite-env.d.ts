/// <reference types="vite/client" />

// VITE_API_BASE is read in lib/api.ts. Declaring it explicitly here, rather
// than relying on Vite's generic ImportMetaEnv, is what keeps a typo in the
// variable name (VITE_API_BAS, say) a compile error instead of a silent
// undefined that only shows up as a 404 once deployed.
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
