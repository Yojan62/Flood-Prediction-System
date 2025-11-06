// src/App.js
import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';

// --- Import Pages ---
// You'll need to update this path based on your new folder structure
import Dashboard from './pages/Dashboard/Dashboard'; 
// import About from './pages/About/About'; // Example of adding a new page

// --- Import Global Components ---
// You'll need to update this path based on your new folder structure
import Layout from './components/Layout/Layout'; 

function App() {
  // --- Global State ---
  // We "lifted" the theme state up from Dashboard
  // so we can pass it to the global Layout and Header.
  const [theme, setTheme] = useState('light');

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // --- Global Effect ---
  // This applies the theme to the entire page.
  useEffect(() => {
    document.body.className = ''; // Clears any existing theme classes.
    document.body.classList.add(theme + '-theme'); // Adds the current theme class.
  }, [theme]);

  return (
    <Routes>
      {/* This parent Route uses your Layout component. 
        All child routes (like Dashboard) will be rendered 
        inside the <Outlet /> in your Layout.
      */}
      <Route 
        path="/" 
        element={<Layout theme={theme} toggleTheme={toggleTheme} />}
      >
        {/* The 'index' route is the default component for "/" (your homepage) */}
        <Route 
          index 
          element={<Dashboard theme={theme} />} 
        />
        
        {/* --- THIS IS HOW TO ADD A NEW PAGE --- */}
        {/* <Route path="about" element={<About />} /> */}
        
        {/* You could add more pages here, like a detail page */}
        {/* <Route path="location/:id" element={<LocationDetailPage />} /> */}

      </Route>
    </Routes>
  );
}

export default App;