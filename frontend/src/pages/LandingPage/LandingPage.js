import React from "react";
import { Link } from "react-router-dom";
import Navbar from "../../components/Layout/Navbar"; // Import Navbar

// --- ASSETS ---
import HeroImage from "../../assets/Banglades.jpeg";
import FlowLogo from "../../assets/Flow.png";
import FloodVideo from "../../assets/Flood video.mp4";
import Logo2 from "../../assets/LOGO.jpeg";

// --- STYLES ---
import "../../styles/landing.css";

const TEAM_MEMBERS = [
  { name: "Yojan Parajuli", role: "Full Stack" },
  { name: "Muhammad Siddiqui", role: "ML & Backend" },
  { name: "Muhammad Ahmer", role: "Backend Developer" },
  { name: "Sean Caroll", role: "Frontend Developer" },
  { name: "Ismaeel Khan", role: "Frontend Developer" },
  { name: "Marcus Li", role: "Frontend Developer" },
  { name: "Muhammad Afnan", role: "Frontend Developer" },
];

function LandingPage({ theme, toggleTheme }) {
  return (
    <div className="landing-page-wrapper">
      
      {/* 1. Navigation Bar */}
      <Navbar theme={theme} toggleTheme={toggleTheme} />

      {/* 2. Hero Section with Overlay */}
      <header 
        className="hero-section" 
        style={{ backgroundImage: `url(${HeroImage})` }}
      >
        <div className="hero-overlay"></div>
        <div className="hero-content">
          <h1 className="hero-title">FLOW</h1>
          <p className="hero-subtitle">Flood Level Observation & Warning System</p>
          
          <div className="hero-actions">
            <Link to="/dashboard" className="cta-button primary">
              Open Dashboard
            </Link>
            <a href="#about" className="cta-button secondary">
              Learn More
            </a>
          </div>
        </div>
      </header>

      {/* 3. Features Section */}
      <section className="section-spacer" id="about">
        <div className="container">
          <h2 className="section-title">Real-Time Intelligence</h2>
          <p className="section-subtitle">
            Advanced technology designed to protect communities and save lives.
          </p>

          <div className="features-grid">
            <div className="feature-card">
              <span className="feature-icon">📡</span>
              <h3>24/7 Monitoring</h3>
              <p>Continuous tracking of river discharge levels across 145 key stations in Bangladesh.</p>
            </div>

            <div className="feature-card">
              <span className="feature-icon">⚡</span>
              <h3>Instant Alerts</h3>
              <p>Receive immediate email notifications the moment a danger threshold is breached.</p>
            </div>

            <div className="feature-card">
              <span className="feature-icon">🤖</span>
              <h3>AI Predictions</h3>
              <p>Our LightGBM machine learning models forecast water levels up to 3 days in advance.</p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Video Section */}
      <section className="section-spacer video-section">
        <div className="container">
          <div className="video-layout">
            <div className="video-wrapper">
              <video controls playsInline>
                <source src={FloodVideo} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="video-text">
              <h3>Stay Ahead of the Water</h3>
              <p>
                Floods can happen fast. Our system gives you the critical lead time needed to 
                evacuate safely or secure your property. By combining live sensor data with 
                historical patterns, FLOW provides accurate, actionable intelligence.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Team Section */}
      <section className="section-spacer">
        <div className="container">
          <h2 className="section-title">Meet the Team</h2>
          <p className="section-subtitle">The developers and engineers behind FLOW.</p>
          
          <div className="team-grid">
            {TEAM_MEMBERS.map((member, index) => (
              <div className="team-card" key={index}>
                <div className="team-avatar">
                  {member.name.split(" ").map(n => n[0]).join("")}
                </div>
                <h5>{member.name}</h5>
                <span className="team-role">{member.role}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 6. Footer */}
      <footer className="landing-footer">
        <img src={FlowLogo} alt="Flow" style={{ height: '50px', marginBottom: '1rem' }} />
        <p>© 2025 FLOW Project. All rights reserved.</p>
      </footer>

    </div>
  );
}

export default LandingPage;