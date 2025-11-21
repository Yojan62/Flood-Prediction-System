import React, { useState } from "react";
import axios from "axios";
import "./AlertSubscriptionCard.css";

function AlertSubscriptionCard({ selectedLocationId }) {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL;

  const validateEmail = (value) => {
    const trimmed = value.trim();
    if (!trimmed.includes("@") || trimmed.length < 5) return false;
    return true;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedLocationId) {
      setMessage("Error: Please select a location first.");
      return;
    }

    if (!validateEmail(email)) {
      setMessage("Error: Please enter a valid email.");
      return;
    }

    setLoading(true);
    setMessage("");

    const subscriptionData = {
      email: email.trim(),
      location_id: selectedLocationId,
    };

    try {
      const response = await axios.post(
        `${API_URL}/api/subscribe`,
        subscriptionData
      );

      setMessage(response.data.message || "Subscribed successfully!");
      setEmail("");
    } catch (error) {
      console.error("Subscription failed:", error);

      if (error.response?.data?.detail) {
        setMessage(`Error: ${error.response.data.detail}`);
      } else {
        setMessage("Error: Could not connect to server.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" id="alert-subscription">
      <h2>Get Flood Alerts</h2>

      <p>
        Receive daily flood warnings for the selected river station directly in
        your email inbox.
      </p>

      {!selectedLocationId && (
        <p className="subscription-message info">
          Please select a location on the map before subscribing.
        </p>
      )}

      <form onSubmit={handleSubmit} className="subscription-form">
        <input
          type="email"
          className="subscription-input"
          value={email}
          placeholder="Enter your email address"
          disabled={loading || !selectedLocationId}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <button
          type="submit"
          className="subscription-button"
          disabled={loading || !selectedLocationId}
        >
          {loading ? (
            <span className="spinner"></span>
          ) : (
            "Subscribe"
          )}
        </button>
      </form>

      {message && (
        <p
          className={`subscription-message ${
            message.startsWith("Error") ? "error" : "success"
          }`}
          role={message.startsWith("Error") ? "alert" : "status"}
          aria-live="polite"
        >
          {message}
        </p>
      )}
    </div>
  );
}

export default AlertSubscriptionCard;
