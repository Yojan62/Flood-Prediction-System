import React from 'react';
import './App.css'; // This imports all the styles from App.css

function App() {
  // 1. Sample data that you would normally get from your backend API
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

  // 2. A helper function to apply the correct color style based on risk
  const getRiskClass = (risk) => {
    if (risk === 'Low') return 'risk-low';
    if (risk === 'Medium') return 'risk-medium';
    if (risk === 'High') return 'risk-high';
    return ''; // Default case
  };

  // The main component structure
  return (
    // Using a React Fragment <> to wrap everything
    <>
      <header>
        <h1>Flood Forecasting Dashboard</h1>
      </header>

      <div className="container">
        <main>
          {/* Map Section */}
          <div className="card" id="map-card">
            <h2>Monitored Location: Dhaka, Bangladesh</h2>
            <div id="map-container">
              <iframe 
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3747020.0084224283!2d87.69690045610567!3d23.489332519505908!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x30adaaed80e18ba7%3A0xf2d28e0c4e1fc6b!2sBangladesh!5e0!3m2!1sen!2suk!4v1760384791432!5m2!1sen!2suk" 
                width="100%" 
                height="400" 
                style={{ border: 0 }} 
                allowFullScreen="" 
                loading="lazy" 
                referrerPolicy="no-referrer-when-downgrade"
                title="Map of Bangladesh"
              ></iframe>
            </div>
          </div>

          {/* Summary Section */}
          <div className="card" id="summary">
            <h2>Forecast Summary</h2>
            <div className="summary-item">
              <span>Current Risk Level:</span>
              <strong className="risk-medium">Medium</strong>
            </div>
            <div className="summary-item">
              <span>Peak Water Level (24h):</span>
              <strong>3.4 meters</strong>
            </div>
            <div className="summary-item">
              <span>Last Updated:</span>
              <strong>13/10/2025, 21:42 BST</strong>
            </div>
          </div>

          {/* AI Insights Section */}
          <div className="card" id="insights">
            <h2>AI Model Insights</h2>
            <p>The model predicts a gradual rise in water levels over the next 12 hours due to sustained rainfall. The risk level is expected to peak at 'High' around 01:00.</p>
          </div>
          
          {/* Forecast Table Section */}
          <div className="card" id="forecast-table-container">
            <h2>Detailed 24-Hour Forecast</h2>
            <table>
              <thead>
                <tr>
                  <th>Time (BST)</th>
                  <th>Predicted Water Level (meters)</th>
                  <th>Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {/* 3. This is the "React way" to loop through data and create table rows */}
                {forecastData.map((dataPoint, index) => (
                  <tr key={index}>
                    <td>{dataPoint.time}</td>
                    <td>{dataPoint.level.toFixed(1)}</td>
                    <td className={getRiskClass(dataPoint.risk)}>{dataPoint.risk}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </main>
      </div>

      <footer>
        <p>© 2025 Flood Prediction Project. All rights reserved.</p>
      </footer>
    </>
  );
}

export default App;