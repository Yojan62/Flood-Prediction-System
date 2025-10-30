import React from 'react';

// This component shows safety advice based on the risk level.
// It expects to receive the current risk ('Low', 'Medium', 'High') as a prop.
function SafetyRecommendationsCard({ currentRisk = 'Low' }) { // Default to 'Low' if no prop is passed

  // Variables to hold the recommendation text and optional card styling based on risk.
  let recommendations = '';
  let cardStyle = {};

  // Determines which set of recommendations to display based on the 'currentRisk' prop.
  switch (currentRisk) {
    case 'High':
      // Sets the recommendation content for High risk. Uses React Fragments (<>) to group paragraphs.
      recommendations = (
        <>
          <p><strong>Immediate Danger!</strong> Follow evacuation orders immediately.</p>
          <p>Move to higher ground. Avoid walking or driving through floodwaters.</p>
          <p>Listen to local emergency broadcasts.</p>
        </>
      );
      // Applies a specific left border color using a CSS variable for High risk.
      cardStyle = { borderLeft: '5px solid var(--risk-high)', paddingLeft: '20px' };
      break;
    case 'Medium':
      // Sets the recommendation content for Medium risk.
      recommendations = (
        <>
          <p><strong>Be Prepared!</strong> Monitor conditions closely.</p>
          <p>Prepare an emergency kit with essentials (water, food, first aid).</p>
          <p>Secure outdoor items and consider moving valuables to higher floors.</p>
        </>
      );
      // Applies a specific left border color using a CSS variable for Medium risk.
      cardStyle = { borderLeft: '5px solid var(--risk-medium)', paddingLeft: '20px' };
      break;
    case 'Low':
    default: // Catches 'Low' risk and any unexpected values.
      // Sets the recommendation content for Low risk.
      recommendations = (
        <>
          <p><strong>Stay Informed.</strong> No immediate danger expected.</p>
          <p>Review your emergency plan and check supplies periodically.</p>
          <p>Be aware of potential changes in weather forecasts.</p>
        </>
      );
      // Applies a specific left border color using a CSS variable for Low risk.
      cardStyle = { borderLeft: '5px solid var(--risk-low)', paddingLeft: '20px' };
      break;
  }

  // Returns the JSX structure for the card.
  return (
    // Uses the generic 'card' CSS class for base styling (background, shadow).
    // Applies the dynamic 'cardStyle' object for the risk-specific border.
    <div className="card" id="safety-recommendations" style={cardStyle}>
      <h2>Safety Guidance</h2>
      {/* Renders the selected recommendation content. */}
      {recommendations}
    </div>
  );
}

// Exports the component for use in other files (like App.js).
export default SafetyRecommendationsCard;