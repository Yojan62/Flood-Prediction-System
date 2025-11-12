import React from 'react';

// This component displays the forecast table.
// It receives the forecastData array via props from the App component.
function ForecastTable({ forecastData = [] }) { // Modern default props

    // Helper function to determine the CSS class based on the risk value.
    const getRiskClass = (risk) => {
        if (!risk) return ''; // Handle null or undefined risk
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
                        <th>Predicted Discharge</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    {/* Maps over the forecastData array passed in from Dashboard.js. */}
                    {forecastData.map((dataPoint) => (
                        <tr key={dataPoint.prediction_id}>
                            {/* 1. The Timestamp */}
                            <td>{formatTimestamp(dataPoint.prediction_timestamp)}</td>
                            
                            {/* 2. The Predicted Discharge (with bug fix) */}
                            <td style={{ fontWeight: 'bold' }}>
                              {/* Check if it's a number before calling .toFixed() */}
                              {(typeof dataPoint.predicted_discharge === 'number')
                                ? `${dataPoint.predicted_discharge.toFixed(2)} m³/s`
                                : 'N/A' 
                              }
                            </td>
                            
                            {/* 3. The Risk Level (with new badge style) */}
                            <td>
                                <span className={`risk-badge ${getRiskClass(dataPoint.risk_level)}`}>
                                    {dataPoint.risk_level}
                                </span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

// Exports the component.
export default ForecastTable;