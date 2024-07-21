// tailwind.config.js
module.exports = {
  content: [
    './templates/*.html',  // Adjust this path based on where your templates are located
    './src/**/*.js',          // Include JS files if you use Tailwind in JS
    './src/**/*.css',         // Include CSS files if you use Tailwind in CSS
    './app.py',               // Include Python files if you have inline Tailwind classes (not common)
  ],
  theme: {
    extend: {
      keyframes: {
        wiggle: {
          '0%, 100%': { transform: 'translateX(0%)' },
          '50%': { transform: 'translateX(calc(-100% - 4px))' },
        },
      },
      animation: {
        wiggle: 'wiggle 1s ease-in-out infinite',
      },
      colors: {
        'customBg': '#A8A29E',
        navbg:'#9F5CB1',
        labelColor:'#474747',
        textcolor:'#522b5b',
        navleft:'#C33BA4',
        card2dark:'#2375E0',
        card2light:'#41B7E5',
        card3dark:'#7B4EE5',
        card3light:'#AE72E4',
      },
      fontFamily: {
        'sans': ['Poppins', 'sans-serif'],
      },
      height:{
        '95vh':'91vh',
        '90vh':'88vh'
      },
      screens:{
        'sm': {'max':'639px'},
        'md': {'max':'1000px'},
      },
    },
  },
  plugins: [],
};
