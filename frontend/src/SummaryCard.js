import React from 'react';

function SummaryCard({data}) {
  // Data for the summary card (currently hard-coded)
  //const currentRisk = "Medium";
  //const peakLevel = 3.4;
  //const lastUpdated = "13/10/2025, 21:42 BST";

  // Helper function for risk class
  const getRiskClass = (risk) => {
    if (risk === 'Low') return 'risk-low';
    if (risk === 'Medium') return 'risk-medium';
    if (risk === 'High') return 'risk-high';
    return '';
  };

  const { currnetRisk = "n/a", peakLevel = "n/a", lastUpdated = "n/a" } = data || {};
  return (
    <div className="card" id="summary">
      <h2>Forecast Summary</h2>
      <div className="summary-item">
        <span>Current Risk Level:</span>
        {/* Displays the currentRisk from 'data' prop */}
        <strong>{peakLevel} {typeof peakLevel === 'number' ? 'meters' : ''}</strong>
      </div>
      <div className="summary-item">
        <span>Last Updated:</span>
        {/* Displays the lastUpdated time from the 'data' prop */}
        <strong>{lastUpdated}</strong>
      </div>
    </div>
  );
}

export default SummaryCard;