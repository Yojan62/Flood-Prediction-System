import React from 'react';

// This component displays the forecast table.
// It receives the forecastData array via props from the App component.
function ForecastTable({ forecastData }) {

    // Helper function to determine the CSS class based on the risk value.
    const getRiskClass = (risk) => {
        const riskLower = String(risk).toLowerCase();

        if (riskLower === 'low') return 'risk-low';
        if (riskLower === 'medium') return 'risk-medium';
        if (riskLower === 'high') return 'risk-high';
        return '';
    };

    // Helper function to format the timestamp from the database.
    const formatTimestamp = (timestamp) => {
        const date = new Date(timestamp);
        // Formats to something like "Oct 31, 2:04 PM" (adjust as needed)
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
            hour12: true
        });
    };

    return (
        // The main container card for the forecast table.
        <div className="card" id="forecast-table-container">
            <h2>Detailed Forecast History</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        {/* The 'level' data doesn't exist in the 'predictions' table yet. */}
                        {/* <th>Predicted Water Level (meters)</th> */}
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    {/* Maps over the forecastData array passed in from App.js. */}
                    {forecastData.map((dataPoint) => (
                        // Uses the unique prediction_id from the database as the key.
                        <tr key={dataPoint.prediction_id}>
                            {/* Uses the 'prediction_timestamp' field and formats it. */}
                            <td>{formatTimestamp(dataPoint.prediction_timestamp)}</td>
                            {/* Uses the 'risk_level' field. */}
                            <td className={getRiskClass(dataPoint.risk_level)}>{dataPoint.risk_level}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

// Sets a default value for the prop to prevent errors if it's undefined.
ForecastTable.defaultProps = {
    forecastData: []
};

// Exports the component.
export default ForecastTable;