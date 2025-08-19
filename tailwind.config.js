/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./docs/**/*.{html,js,md}",
    "./odoo_modules/**/*.{xml,js,scss}",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brass: tokens.color.brass.base.value,
        brassDark: tokens.color.brass.dark.value,
        teal: tokens.color.teal.base.value,
        tealLight: tokens.color.teal.light.value,
      },
        gray: {
          light: "#F8F6F4",
          neutral: "#F4F2EE",
          dark: "#333333",
        },
      },
      fontFamily: {
        body: tokens.font.body.value.split(","),
        code: ["Fira Code", "monospace"],
      },
      fontSize: {
        h1: "2.25rem", // ~36px
        h2: "1.75rem", // ~28px
        h3: "1.5rem", // ~24px
        body: "1rem", // ~16px
        small: "0.875rem", // ~14px
      },
      fontWeight: {
        regular: "400",
        medium: "500",
        bold: "700",
      },
      boxShadow: {
        soft: "0 2px 6px rgba(0,0,0,0.1)",
        medium: "0 4px 12px rgba(0,0,0,0.15)",
      },
      borderRadius: {
        sm: "6px",
        md: "12px",
        lg: "20px",
      },
      gradientColorStops: {
        brass: {
          start: "#C5A46D",
          end: "#E6C97B",
        },
      },
    },
  },
  plugins: [],
};
