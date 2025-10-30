import React from 'react';

// Defines the SummaryCard component, which displays key forecast information.
// It expects to receive a 'data' object prop containing summary details.
function SummaryCard({ data }) {

  // Helper function to determine the CSS class based on the risk value for styling.
  const getRiskClass = (risk) => {
    if (risk === 'Low') return 'risk-low';
    if (risk === 'Medium') return 'risk-medium';
    if (risk === 'High') return 'risk-high';
    return ''; // Returns empty string if risk level is unknown or not provided.
  };

  // Destructures the 'data' prop object to extract specific values.
  // Provides default fallback values ('N/A') if 'data' or its properties are missing.
  // Corrected typo: 'currnetRisk' changed to 'currentRisk'.
  const { currentRisk = 'N/A', peakLevel = 'N/A', lastUpdated = 'N/A' } = data || {};

  // Returns the JSX structure for the summary card.
  return (
    // Uses the generic 'card' class and a specific ID for styling.
    <div className="card" id="summary">
      <h2>Forecast Summary</h2>
      {/* Section to display the current risk level. */}
      <div className="summary-item">
        <span>Current Risk Level:</span>
        {/* Displays the currentRisk value from props and applies the corresponding CSS class. */}
        <strong className={getRiskClass(currentRisk)}>{currentRisk}</strong>
      </div>
      {/* Section to display the peak water level. */}
      <div className="summary-item">
        <span>Peak Water Level (24h):</span>
        {/* Displays the peakLevel value from props, adding 'meters' if it's a number. */}
        <strong>{peakLevel} {typeof peakLevel === 'number' ? 'meters' : ''}</strong>
      </div>
      {/* Section to display the last updated timestamp. */}
      <div className="summary-item">
        <span>Last Updated:</span>
        {/* Displays the lastUpdated value from props. */}
        <strong>{lastUpdated}</strong>
      </div>
    </div>
  );
}

// Exports the SummaryCard component for use in other files (like App.js).
export default SummaryCard;