import React, { useState, useEffect, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import './App.css';

// --- LAZY LOAD COMPONENTS ---
// Ensure these paths match your actual folder structure exactly.
// Based on your uploads, they seem to be nested: src/pages/Name/Name.js
const Layout = React.lazy(() => import("./components/Layout/Layout"));
const LandingPage = React.lazy(() => import("./pages/LandingPage/LandingPage"));
const Dashboard = React.lazy(() => import("./pages/Dashboard/Dashboard"));
const SafetyGuidance = React.lazy(() => import("./pages/SafetyGuidance/SafetyGuidance"));

function App() {
  // Initialize theme state
  const storedTheme = localStorage.getItem("theme") || "light";
  const [theme, setTheme] = useState(storedTheme);

  // Toggle function passed to Navbar/Layout
  const toggleTheme = () => {
    setTheme(prev => (prev === "light" ? "dark" : "light"));
  };

  // Apply theme class to body
  useEffect(() => {
    document.body.className = "";
    if (theme === "dark") {
      document.body.classList.add("dark-theme");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    // Suspense must wrap lazy-loaded components
    <Suspense fallback={<div className="loading-screen">Loading...</div>}>
      <Routes>
        
        {/* 1. Landing Page (Standalone - No Layout wrapper) */}
        {/* We pass theme props so the Navbar inside LandingPage works */}
        <Route 
          path="/" 
          element={<LandingPage theme={theme} toggleTheme={toggleTheme} />} 
        />

        {/* 2. Dashboard & Safety (Wrapped in Layout) */}
        <Route element={<Layout theme={theme} toggleTheme={toggleTheme} />}>
          <Route path="/dashboard" element={<Dashboard theme={theme} />} />
          <Route path="/safety-guidance" element={<SafetyGuidance />} />
        </Route>

      </Routes>
    </Suspense>
  );
}

export default App;