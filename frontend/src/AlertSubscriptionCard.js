import React, { useState } from 'react';
import axios from 'axios'; // Used for making the HTTP request

// Defines the AlertSubscriptionCard component.
function AlertSubscriptionCard() {
  // State to hold the email address entered by the user.
  const [email, setEmail] = useState('');
  // State to show a success or error message to the user after submission.
  const [message, setMessage] = useState('');
  // State to disable the button while the request is in progress.
  const [loading, setLoading] = useState(false);

  // Handles the form submission event.
  const handleSubmit = async (event) => {
    // Prevents the default browser behavior of reloading the page.
    event.preventDefault();
    
    // Disables the form button to prevent multiple clicks.
    setLoading(true);
    // Clears any previous messages.
    setMessage('');

    // Defines the data to be sent to the backend.
    const subscriptionData = {
      email: email,
      location_id: 1 // TODO: Hard-coded location 1 (e.g., Dhaka) for now.
                      // Later, this should be passed in as a prop.
    };

    try {
      // Makes the POST request to the backend API endpoint.
      const response = await axios.post('http://127.0.0.1:8000/api/subscribe', subscriptionData);

      // Handles a successful response from the backend.
      setMessage(response.data.message); // Shows success message from the API.
      setLoading(false); // Re-enables the button.
      setEmail(''); // Clears the input field.

    } catch (error) {
      // Handles an error from the backend.
      console.error("Subscription failed:", error);
      if (error.response) {
        // Sets the error message from the API response (e.g., "Location not found").
        setMessage(`Error: ${error.response.data.detail}`);
      } else {
        // Sets a generic error message if the server can't be reached.
        setMessage('Error: Could not connect to the server.');
      }
      setLoading(false); // Re-enables the button.
    }
  };

  // Returns the JSX structure for the subscription card.
  return (
    // Uses the generic 'card' CSS class for base styling and a specific ID.
    <div className="card" id="alert-subscription">
      <h2>Get Flood Alerts</h2>
      <p>Enter your email to receive alerts for the current location.</p>
      
      {/* The form element triggers the handleSubmit function. */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter your email address"
          required // Makes this field mandatory.
          disabled={loading} // Disables input while loading.
          style={{ flexGrow: 1, padding: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        <button
          type="submit"
          disabled={loading} // Disables button while loading.
          style={{
            padding: '10px 15px',
            borderRadius: '4px',
            border: 'none',
            backgroundColor: 'var(--primary-accent)',
            color: 'white',
            cursor: 'pointer',
            opacity: loading ? 0.7 : 1 // Dims the button when disabled.
          }}
        >
          {loading ? 'Subscribing...' : 'Subscribe'}
        </button>
      </form>

      {/* Displays the success or error message to the user. */}
      {message && <p style={{ marginTop: '15px', color: message.startsWith('Error') ? 'var(--risk-high)' : 'var(--risk-low)' }}>{message}</p>}
    </div>
  );
}

// Exports the component for use in other files (like App.js).
export default AlertSubscriptionCard;