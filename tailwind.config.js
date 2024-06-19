// tailwind.config.js
module.exports = {
  content: [
    './templates/*.html',  // Adjust this path based on where your templates are located
    './src/**/*.js',          // Include JS files if you use Tailwind in JS
    './src/**/*.css',         // Include CSS files if you use Tailwind in CSS
    './app.py',               // Include Python files if you have inline Tailwind classes (not common)
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
