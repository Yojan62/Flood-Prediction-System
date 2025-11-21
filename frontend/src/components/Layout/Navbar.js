import React from "react";
import { Link, useLocation } from "react-router-dom";
import ThemeToggle from "../UI/ThemeToggle"; // Fixed: Changed '...' to '..'
import FlowLogo from "../../assets/Flow.png";
import "./Navbar.css";

function Navbar({ theme, toggleTheme }) {
  const location = useLocation();

  return (
    <nav className="main-navbar">
      <div className="nav-left">
        <Link to="/" className="nav-logo">
          <img src={FlowLogo} alt="Flow logo" />
          <span>FLOW</span>
        </Link>
      </div>

      <div className="nav-links">
        <Link 
          className={location.pathname === "/" ? "active" : ""} 
          to="/"
        >
          Home
        </Link>
        
        <Link 
          className={location.pathname.includes("dashboard") ? "active" : ""} 
          to="/dashboard"
        >
          Dashboard
        </Link>
        
        {/* Fixed: Updated route to match App.js */}
        <Link 
          className={location.pathname.includes("safety-guidance") ? "active" : ""} 
          to="/safety-guidance"
        >
          Safety
        </Link>
      </div>

      <div className="nav-right">
        <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
      </div>
    </nav>
  );
}

export default Navbar;