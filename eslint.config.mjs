import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Browser profiles, generated audio/poster artifacts, and other local QA
    // state live here. They are not application source and may contain large
    // third-party extension bundles that make a repository-wide lint hang.
    "api/runtime/**",
  ]),
]);

export default eslintConfig;
