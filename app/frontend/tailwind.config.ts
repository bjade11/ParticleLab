import type { Config } from "tailwindcss";

// Particle Lab — refined commercial design tokens.
// Paper-white surface · ink blue accent · coral/amber for status.
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg0: "#f6f6f1",
        bg1: "#ffffff",
        bg2: "#fbfbf7",
        bg3: "#ececdf",
        line: "#e6e6df",
        "line-soft": "#eeeee7",
        "line-strong": "#d3d3cb",
        ink: { DEFAULT: "#2057d4", hover: "#1846b5", soft: "rgba(32,87,212,0.10)" },
        coral: { DEFAULT: "#d6336c", soft: "rgba(214,51,108,0.08)" },
        amber: { DEFAULT: "#b8860b", soft: "rgba(184,134,11,0.08)" },
        good: { DEFAULT: "#128a4f", soft: "rgba(18,138,79,0.08)" },
        bad: { DEFAULT: "#c92a2a", soft: "rgba(201,42,42,0.08)" },
        text: "#14161b",
        "text-mute": "#5a606e",
        "text-faint": "#8d92a0",
      },
      fontFamily: {
        sans: [
          "GmarketSans",
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Apple SD Gothic Neo",
          "Pretendard",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SF Mono",
          "Menlo",
          "monospace",
        ],
      },
      letterSpacing: { wide2: "0.08em", wide3: "0.1em" },
      fontSize: {
        "2xs": "10px",
        "3xs": "9px",
      },
      borderRadius: {
        DEFAULT: "8px",
        md: "10px",
        lg: "14px",
        xl: "18px",
      },
      boxShadow: {
        xs: "0 1px 0 rgba(20,22,27,0.04)",
        sm: "0 1px 2px rgba(20,22,27,0.04), 0 1px 0 rgba(20,22,27,0.03)",
        md: "0 1px 0 rgba(20,22,27,0.03), 0 8px 24px -10px rgba(20,22,27,0.10)",
        lg: "0 1px 0 rgba(20,22,27,0.03), 0 24px 40px -16px rgba(20,22,27,0.14)",
        pop: "0 10px 28px -10px rgba(32,87,212,0.40)",
      },
    },
  },
  plugins: [],
};
export default config;
