import React from 'react';
// Imports the ThemeToggle component to be rendered inside the header.
import ThemeToggle from './ThemeToggle';

// Defines the Header component for the dashboard.
// It accepts 'theme' and 'toggleTheme' as props from App.js.
function Header({ theme, toggleTheme }) {
  // Returns the JSX structure for the header.
  return (
    // Uses the standard HTML header element, which is styled with flexbox.
    <header>
      {/* Displays the main title of the dashboard. */}
      <h1>Flood Forecasting Dashboard</h1>

      {/* Renders the ThemeToggle component. */}
      {/* Passes the 'theme' and 'toggleTheme' props down to the toggle switch. */}
      <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
    </header>
  );
}

// Exports the Header component so it can be used in App.js.
export default Header;