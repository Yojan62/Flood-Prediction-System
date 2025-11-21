import React from 'react';
import "../../styles/safety.css"; // Import the CSS file we just made

function SafetyGuidance() {
  return (
    <div className="safety-page">
      <div className="safety-container">
        
        {/* --- Header --- */}
        <div className="safety-header">
          <h1>Flood Safety Guidance</h1>
          <p>Stay Safe · Stay Prepared · Stay Informed</p>
        </div>

        {/* --- Section 1: Understanding --- */}
        <h2 className="section-title">1. Understanding the Risk</h2>
        <div className="safety-flex">
          <div className="card">
            <h3>🌊 What Causes Flooding?</h3>
            <p>Flooding is the overflow of water onto normally dry land. Common causes include:</p>
            <ul>
              <li><strong>Heavy Rainfall:</strong> Overwhelms drainage systems.</li>
              <li><strong>River Overflow:</strong> Banks burst due to upstream rain.</li>
              <li><strong>Infrastructure Failure:</strong> Dam or levee breaches.</li>
            </ul>
          </div>

          <div className="card">
            <h3>⚠️ The Dangers</h3>
            <p>Risks extend beyond water damage:</p>
            <ul>
              <li><strong>Currents:</strong> Fast-flowing water can knock you over.</li>
              <li><strong>Debris:</strong> Hidden objects can cause injury.</li>
              <li><strong>Disease:</strong> Floodwater is often contaminated.</li>
            </ul>
          </div>
        </div>

        {/* --- Section 2: Preparation --- */}
        <h2 className="section-title">2. Preparation</h2>
        <div className="safety-flex">
          <div className="card">
            <h3>🏠 Fortify Your Home</h3>
            <ul>
              <li>Install flood barriers or sandbags.</li>
              <li>Clear gutters and drains regularly.</li>
              <li>Elevate electrical sockets and appliances.</li>
            </ul>
          </div>

          <div className="card">
            <h3>🎒 The Grab Bag</h3>
            <p>Keep an emergency kit ready with:</p>
            <ul>
              <li>🔦 Torch & Batteries</li>
              <li>📻 Portable Radio</li>
              <li>💊 First Aid Kit & Meds</li>
              <li>📄 Important Documents (in a waterproof bag)</li>
              <li>💧 Bottled Water & Non-perishable food</li>
            </ul>
          </div>

          <div className="card full">
            <h3>🗺️ Make a Plan</h3>
            <p>
              Agree on a family meeting point. Know your evacuation routes. 
              Save emergency numbers (Fire, Police, Ambulance) in your phone and written down.
            </p>
          </div>
        </div>

        {/* --- Section 3: Action --- */}
        <h2 className="section-title">3. Action (When Flooding Starts)</h2>
        <div className="safety-flex">
          <div className="card risk-high">
            <h3>🚨 Imminent Danger</h3>
            <p><strong>ACT IMMEDIATELY:</strong></p>
            <ol style={{ marginLeft: '1.2rem', lineHeight: '1.6' }}>
              <li>Turn off gas, electricity, and water supplies.</li>
              <li>Move family and pets to a high floor.</li>
              <li>Listen to emergency services.</li>
              <li><strong>Evacuate if told to do so.</strong></li>
            </ol>
          </div>

          <div className="card risk-medium">
            <h3>🚗 Travel Safety</h3>
            <p><strong>Turn Around, Don't Drown.</strong></p>
            <p>Never drive through floodwater. Just 30cm of moving water can float a car.</p>
            <p>Avoid walking through water; hidden manholes and debris are dangerous.</p>
          </div>
        </div>

        {/* --- Footer --- */}
        <div className="safety-footer">
          <h3>Respect the Water</h3>
          <p>Floods are life-threatening. Your safety is more important than your property.</p>
        </div>

      </div>
    </div>
  );
}

export default SafetyGuidance;