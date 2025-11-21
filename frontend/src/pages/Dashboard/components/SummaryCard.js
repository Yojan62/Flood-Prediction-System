import React from "react";
import "./SummaryCard.css";

function SummaryCard({ data, loading = false, dangerThreshold }) {
  if (!data && !loading) {
    return (
      <div className="card" id="summary">
        <h2>Forecast Summary</h2>
        <p className="summary-empty-state">
          Select a location to view its current risk and peak water level.
        </p>
      </div>
    );
  }

  const { currentRisk, peakLevel, lastUpdated } = data || {};

  const risk = currentRisk ? String(currentRisk).toLowerCase() : null;

  const getRiskClass = (level) => {
    switch (level) {
      case "low":
        return "risk-low";
      case "medium":
        return "risk-medium";
      case "high":
        return "risk-high";
      default:
        return "";
    }
  };

  const peakNumber =
    peakLevel != null && !isNaN(Number(peakLevel))
      ? Number(peakLevel)
      : null;

  const thresholdNumber =
    dangerThreshold != null && !isNaN(Number(dangerThreshold))
      ? Number(dangerThreshold)
      : null;

  return (
    <div className="card" id="summary">
      <h2>Forecast Summary</h2>

      {loading && (
        <p className="summary-empty-state">Loading summary…</p>
      )}

      {!loading && (
        <>
          <div className="summary-item centered">
            <span className={`risk-badge ${getRiskClass(risk)}`}>
              {risk
                ? risk.charAt(0).toUpperCase() + risk.slice(1)
                : "Unknown"}
            </span>
          </div>

          <div className="summary-item">
            <span>Peak Water Level (24h):</span>
            <strong>
              {peakNumber !== null
                ? `${peakNumber.toFixed(2)} m³/s`
                : "No data"}
            </strong>
          </div>

          {thresholdNumber !== null && (
            <div className="summary-item">
              <span>Danger Threshold:</span>
              <strong>
                {thresholdNumber.toFixed(2)} m³/s
              </strong>
            </div>
          )}

          <div className="summary-item">
            <span>Last Updated:</span>
            <strong>{lastUpdated || "No data"}</strong>
          </div>
        </>
      )}
    </div>
  );
}

export default SummaryCard;
