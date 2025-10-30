import React from 'react';

// This component displays the forecast table.
// It receives the forecastData array via props from the App component.
function ForecastTable({ forecastData }) {

    // Helper function to determine the CSS class based on the risk value.
    const getRiskClass = (risk) => {
        if (risk === 'Low') return 'risk-low';
        if (risk === 'Medium') return 'risk-medium';
        if (risk === 'High') return 'risk-high';
        return ''; // Return empty string if risk level is unknown
    };

    return (
        // The main container card for the forecast table, styled using CSS classes.
        <div className="card" id="forecast-table-container">
            <h2>Detailed 24-Hour Flood Forecast</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time (BST)</th>
                        <th>Predicted Water Level (meters)</th>
                        <th>Risk Level</th> {/* Added missing Risk Level header */}
                    </tr>
                </thead>
                <tbody>
                    {/* Maps over the forecastData array passed in props. */}
                    {/* For each dataPoint, creates a table row (tr). */}
                    {forecastData.map((dataPoint, index) => (
                        <tr key={index}> {/* Using index as key is okay for this static list. */}
                            <td>{dataPoint.time}</td>
                            <td>{dataPoint.level.toFixed(1)}</td> {/* Formats water level to one decimal place. */}
                            {/* Sets the CSS class dynamically based on risk level for styling. */}
                            <td className={getRiskClass(dataPoint.risk)}>{dataPoint.risk}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

// Exports the component, making it available to import and use elsewhere (e.g., in App.js).
export default ForecastTable;