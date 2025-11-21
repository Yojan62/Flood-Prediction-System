import React from "react";
import { FaSun, FaMoon } from "react-icons/fa";
import "./ThemeToggle.css";

function ThemeToggle({ theme, toggleTheme }) {
  const isDark = theme === "dark";

  return (
    <button
      className={`theme-toggle-switch ${isDark ? "dark" : ""}`}
      onClick={toggleTheme}
      aria-label="Toggle theme"
      role="switch"
      aria-checked={isDark}
    >
      <span className="slider-circle">
        {isDark ? (
          <FaSun className="icon-light" />
        ) : (
          <FaMoon className="icon-dark" />
        )}
      </span>
    </button>
  );
}

export default ThemeToggle;
