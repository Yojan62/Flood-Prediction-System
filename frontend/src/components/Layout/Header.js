// src/components/layout/Header.js
import React from 'react';
import { NavLink } from 'react-router-dom';
import ThemeToggle from '../UI/ThemeToggle';
import FlowLogo from '../../assets/Flow.png'; 

function Header({ theme, toggleTheme }) {
  return (
    // Using your class "main-navbar"
    <header className="main-navbar">
      <NavLink className="nav-logo" to="/">
        <img src={FlowLogo} alt="Flow Logo" />
        <span>FLOW</span>
      </NavLink>
      
      <nav className="nav-links">
        <NavLink to="/" end>Home</NavLink>
        <NavLink to="/dashboard">Dashboard</NavLink>
        <NavLink to="/safety-guidance">Safety</NavLink>
        <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
      </nav>
    </header>
  );
}

export default Header;