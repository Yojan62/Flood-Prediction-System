import React from 'react';

// This component displays the forecast table.
// It receives the forecastData array as a 'prop'
function ForecastTable({ forecastData }) {

    // Helper function to get the CSS class for risk level styling.
    const getRiskClass = (risk) => {
        if (risk === 'Low') return 'risk-low';
        if (risk === 'Medium') return 'risk-medium';
        if (risk === 'High') return 'risk-high';
        return '';
    };

    return (
        // This is the main container card for the table.
        <div className="card" id="forecast-table-container">
            <h2>Detailed 24-Hour Flood Forecast</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time (BST)</th>
                        <th>Predicted Water Level (meters)</th>
                    </tr>
                </thead>
                <tbody>
                    {/* Loops through the forecastData prop and creates a row for each item. */}
                    {forecastData.map((dataPoint, index) => (
                        <tr key={index}> {/* Using index as key is okay for static lists */}
                        <td>{dataPoint.time}</td>
                        <td>{dataPoint.level.toFixed(1)}</td> {/* Formats level to one decimal place */}
                        <td className={getRiskClass(dataPoint.risk)}>{dataPoint.risk}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div> 
    );
}

// Makes the ForecastTable component available for import in other files.
export default ForecastTable;