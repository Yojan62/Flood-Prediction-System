import React from "react";
import { Outlet } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";

import "../../styles/layout.css";

function Layout({ theme, toggleTheme }) {
  return (
    <div className="app-layout">
      <Header theme={theme} toggleTheme={toggleTheme} />
      <main>
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

export default Layout;
