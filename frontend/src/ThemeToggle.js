import React from 'react';
// Imports the Sun and Moon icons from the 'react-icons/fa' library.
import { FaSun, FaMoon } from 'react-icons/fa';
// Imports the component's specific CSS styles.
import './ThemeToggle.css';

// Defines the ThemeToggle component.
// It receives the current 'theme' ('light' or 'dark') and the 'toggleTheme' function as props.
function ThemeToggle({ theme, toggleTheme }) {
  // Determines if the current theme is 'dark' for conditional rendering and styling.
  const isDark = theme === 'dark';

  // Returns the JSX structure for the theme toggle switch.
  return (
    // The main button element functions as the switch's track.
    // An 'onClick' handler is attached to call the 'toggleTheme' function when clicked.
    // The 'dark' CSS class is conditionally applied based on the 'isDark' state for styling.
    <button
      onClick={toggleTheme}
      className={`theme-toggle-switch ${isDark ? 'dark' : ''}`}
    >
      {/* This inner span represents the sliding circle (or 'thumb') of the switch. */}
      <span className="slider-circle">
        {/* Conditionally renders either the Sun or Moon icon inside the circle. */}
        {/* If 'isDark' is true (dark theme active), the Sun icon is shown. */}
        {isDark ? (
          <FaSun size={14} className="icon-light" /> // Sun icon, typically for a dark background.
        ) : (
          // If 'isDark' is false (light theme active), the Moon icon is shown.
          <FaMoon size={14} className="icon-dark" /> // Moon icon, typically for a light background.
        )}
      </span>
    </button>
  );
}

// Exports the ThemeToggle component so it can be used in other files (like App.js).
export default ThemeToggle;