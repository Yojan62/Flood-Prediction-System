import React from 'react';
import { FaSun, FaMoon } from 'react-icons/fa'; // Sun and Moon icons
import './ThemeToggle.css'; // Import the new CSS

// Receives the current theme ('light' or 'dark') and the toggle function
function ThemeToggle({ theme, toggleTheme }) {
  // Determine if the dark theme is active
  const isDark = theme === 'dark';

  return (
    // The main button element acts as the switch track
    // Add the 'dark' class conditionally
    <button
      onClick={toggleTheme}
      className={`theme-toggle-switch ${isDark ? 'dark' : ''}`} // Add 'dark' class if theme is dark
    >
      {/* The inner span is the sliding circle */}
      <span className="slider-circle">
        {/* Show the correct icon inside the circle based on the theme */}
        {isDark ? (
          <FaSun size={14} className="icon-light" /> // Sun icon for dark theme
        ) : (
          <FaMoon size={14} className="icon-dark" /> // Moon icon for light theme
        )}
      </span>
    </button>
  );
}

export default ThemeToggle;