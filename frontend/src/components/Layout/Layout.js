// src/components/layout/Layout.js
import React from 'react';
import { Outlet } from 'react-router-dom';

// Import your Header and Footer
// You'll need to update these paths based on your new folder structure
import Header from './Header'; 
import Footer from './Footer';

// The Layout component receives the global 'theme' and 'toggleTheme' props
// from App.js and passes them down to the Header.
function Layout({ theme, toggleTheme }) {
  return (
    <>
      <Header theme={theme} toggleTheme={toggleTheme} />

      {/* <Outlet /> is the most important part.
        React Router will automatically render your page component 
        (e.g., Dashboard.js or About.js) right here.
      */}
      <Outlet />

      <Footer />
    </>
  );
}

export default Layout;