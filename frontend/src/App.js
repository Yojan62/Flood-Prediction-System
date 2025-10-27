import React, {useState, useEffect} from 'react';
import ThemeToggle from './ThemeToggle';
import './App.css'; // Main styles

// Import all the new components
import Header from './Header';
import MapCard from './MapCard';
import SummaryCard from './SummaryCard';
import InsightsCard from './InsightsCard';
import ForecastTable from './ForecastTable';
import WeatherWidget from './WeatherWidget';
import Footer from './Footer';

function App() {
  // State to manage the current theme ('light' or 'dark')
  const [theme, setTheme] = useState('light'); // Default to light theme

  // Function to toggle the theme
  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // Effect to add/remove the theme class on the body element
  useEffect(() => {
    document.body.className = ''; // Clear existing classes first
    document.body.classList.add(theme + '-theme'); // Add 'light-theme' or 'dark-theme'
  }, [theme]); // Run this effect whenever the theme state changes

  // Sample forecast data (kept in App.js for now)
  const forecastData = [
    { time: "16:00", level: 2.8, risk: "Low" },
    { time: "19:00", level: 3.1, risk: "Medium" },
    { time: "22:00", level: 3.2, risk: "Medium" },
    { time: "01:00", level: 3.4, risk: "High" },
    { time: "04:00", level: 3.3, risk: "High" },
    { time: "07:00", level: 3.0, risk: "Medium" },
    { time: "10:00", level: 2.7, risk: "Low" },
    { time: "13:00", level: 2.5, risk: "Low" }
  ];

  return (
    // Uses a React Fragment (<>) to group the components
    <>
      {/* Renders the Header component */}
      <Header />

      {/* Main container for the dashboard content */}
      <div className="container">
        <main>
          {/* Renders the ThemeToggle component, passing current theme and toggle function as props */}
          <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
          {/* Renders the MapCard component */}
          <MapCard />
          {/* Renders the SummaryCard component */}
          <SummaryCard />
          {/* Renders the InsightsCard component */}
          <InsightsCard />
          {/* Renders the ForecastTable component, passing the data as a prop */}
          <ForecastTable forecastData={forecastData} />
          {/* Renders the WeatherWidget component */}
          <WeatherWidget />
        </main>
      </div>

      {/* Renders the Footer component */}
      <Footer />
    </>
  );
}

// Exports the App component to be used in index.js
export default App;