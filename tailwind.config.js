// tailwind.config.js
import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./docs/**/*.{html,js,md}",
    "./src/**/*.{html,js,ts,vue,jsx,tsx}",
    "./odoo/**/*.{xml,js}",
  ],
  theme: {
    extend: {
      colors: {
        odoo: {
          purple: "#714B67",
          teal: "#008080",
          brass: "#B08D57",
        },
        brassGradient: {
          start: "#B08D57",
          mid: "#FFD700",
          end: "#DAA520",
        },
      },
      fontFamily: {
        sans: ['"Inter var"', "ui-sans-serif", "system-ui", "sans-serif"],
        heading: ['"Poppins"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
      container: {
        center: true,
        padding: "1rem",
        screens: {
          sm: "640px",
          md: "768px",
          lg: "1024px",
          xl: "1280px",
          "2xl": "1536px",
        },
      },
    },
  },
  plugins: [typography],
};
