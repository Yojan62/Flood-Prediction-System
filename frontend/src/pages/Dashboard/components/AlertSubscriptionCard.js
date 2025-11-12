// src/pages/Dashboard/components/AlertSubscriptionCard.js
import React, { useState } from 'react';
import axios from 'axios';
import './AlertSubscriptionCard.css';

function AlertSubscriptionCard({ selectedLocationId }) {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setMessage('');

    const subscriptionData = {
      email: email,
      location_id: selectedLocationId 
    };

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/subscribe', subscriptionData);
      setMessage(response.data.message);
      setLoading(false);
      setEmail('');
    } catch (error) {
      console.error("Subscription failed:", error);
      if (error.response) {
        setMessage(`Error: ${error.response.data.detail}`);
      } else {
        setMessage('Error: Could not connect to the server.');
      }
      setLoading(false);
    }
  };

  return (
    <div className="card" id="alert-subscription">
      <h2>Get Flood Alerts</h2>
      <p>Enter your email to receive alerts for the current location.</p>
      
      <form onSubmit={handleSubmit} className="subscription-form">
        <input
          type="email"
          className="subscription-input" 
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter your email address"
          required
          disabled={loading}
        />
        <button
          type="submit"
          className="subscription-button" 
          disabled={loading}
        >
          {loading ? 'Subscribing...' : 'Subscribe'}
        </button>
      </form>

      {/* Use classes for the message for consistent styling */}
      {message && (
        <p className={`subscription-message ${message.startsWith('Error') ? 'error' : 'success'}`}>
          {message}
        </p>
      )}
    </div>
  );
}

export default AlertSubscriptionCard;