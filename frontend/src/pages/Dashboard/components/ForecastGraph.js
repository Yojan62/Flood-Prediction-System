// src/pages/Dashboard/components/ForecastGraph.js
import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import annotationPlugin from "chartjs-plugin-annotation";

Chart.register(annotationPlugin);

function ForecastGraph({
  forecastData,
  loading = false,
  error = null,
  dangerThreshold,
  theme = "light",
}) {
  const chartRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    // Destroy old chart
    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = null;
    }

    // No data → simple text in the canvas
    if (!forecastData || forecastData.length === 0) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.save();
      ctx.textAlign = "center";
      ctx.fillStyle = "#9ca3af";
      ctx.font = "16px Inter, system-ui, sans-serif";
      ctx.fillText(
        "Select a location on the map to view its forecast.",
        canvasRef.current.width / 2,
        canvasRef.current.height / 2
      );
      ctx.restore();
      return;
    }

    const isDark = theme === "dark";

    const lineColor = isDark ? "#00BFFF" : "#2e7d32"; // electric blue vs forest green
    const fillColor = isDark
      ? "rgba(0, 191, 255, 0.18)"
      : "rgba(46, 125, 50, 0.18)";
    const gridColor = isDark
      ? "rgba(148, 163, 184, 0.35)"
      : "rgba(148, 163, 184, 0.35)";
    const tickColor = isDark ? "#e5e7eb" : "#4b5563";

    const labels = forecastData
      .map((d) => new Date(d.prediction_timestamp).toLocaleString())
      .reverse();

    const values = forecastData
      .map((d) => d.predicted_discharge)
      .reverse();

    const ctx = canvasRef.current.getContext("2d");

    const annotations = {};
    if (
      dangerThreshold != null &&
      !Number.isNaN(dangerThreshold)
    ) {
      annotations.dangerLine = {
        type: "line",
        yMin: dangerThreshold,
        yMax: dangerThreshold,
        borderColor: "#ef4444",
        borderWidth: 2,
        borderDash: [6, 6],
        label: {
          display: true,
          content: `Danger: ${dangerThreshold.toFixed(2)} m³/s`,
          backgroundColor: "rgba(239, 68, 68, 0.9)",
          color: "#fff",
          position: "end",
          yAdjust: -6,
          font: {
            size: 11,
            weight: "bold",
          },
        },
      };
    }

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Predicted Discharge (m³/s)",
            data: values,
            borderColor: lineColor,
            backgroundColor: fillColor,
            fill: true,
            tension: 0.2,
            pointRadius: 3,
            pointHoverRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            labels: {
              color: tickColor,
            },
          },
          tooltip: {
            backgroundColor: isDark
              ? "rgba(15, 23, 42, 0.95)"
              : "rgba(255,255,255,0.95)",
            titleColor: isDark ? "#e5e7eb" : "#111827",
            bodyColor: isDark ? "#e5e7eb" : "#111827",
            borderColor: gridColor,
            borderWidth: 1,
            callbacks: {
              label: (ctx) => `${ctx.parsed.y.toFixed(2)} m³/s`,
            },
          },
          annotation: {
            annotations,
          },
        },
        scales: {
          x: {
            ticks: {
              maxRotation: 0,
              color: tickColor,
            },
            grid: {
              color: gridColor,
            },
          },
          y: {
            beginAtZero: false,
            ticks: {
              color: tickColor,
              callback: (value) =>
                typeof value === "number" ? value.toFixed(2) : value,
            },
            grid: {
              color: gridColor,
            },
          },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [forecastData, dangerThreshold, theme]);

  return (
    <div className="card" id="forecast-graph-container">
      <h2>Forecast Graph</h2>
      {error && <p className="graph-error">{error}</p>}
      <div className="chart-container">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}

export default ForecastGraph;
