import React, { useState } from 'react'; // Imports React and the useState hook for managing component state.

// Defines the AlertSubscriptionCard component.
// This component provides a simple form for users to subscribe to email alerts.
function AlertSubscriptionCard() {
  // State hook to manage the value of the email input field. Initialized to an empty string.
  const [email, setEmail] = useState('');

  // Handles the form submission event.
  const handleSubmit = (event) => {
    // Prevents the default browser behavior of reloading the page on form submission.
    event.preventDefault();
    // Logs the entered email to the console for debugging purposes.
    console.log("Subscribing email:", email);
    // Placeholder for future logic to send the email address to the backend API.
    // TODO: Implement API call to backend subscription endpoint.
    // Provides simple feedback to the user that the request was received (for now).
    alert(`Subscription request for ${email} received! (Backend not connected yet)`);
    // Clears the email input field after submission by resetting the state.
    setEmail('');
  };

  // Returns the JSX structure for the subscription card.
  return (
    // Uses the generic 'card' CSS class for base styling and a specific ID.
    <div className="card" id="alert-subscription">
      <h2>Get Flood Alerts</h2>
      <p>Enter your email to receive alerts for the current location.</p>
      {/* The form element triggers the handleSubmit function upon submission. */}
      {/* Inline styles are used for basic layout (flexbox) and spacing. */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
        {/* Email input field. */}
        <input
          type="email" // Specifies the input type for email validation.
          value={email} // Binds the input's value to the 'email' state variable.
          // Updates the 'email' state whenever the input value changes.
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter your email address"
          required // Makes this field mandatory for form submission.
          // Basic inline styles for the input field.
          style={{ flexGrow: 1, padding: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        {/* Submit button for the form. */}
        <button
          type="submit"
          // Basic inline styles using CSS variables for theme consistency.
          style={{ padding: '10px 15px', borderRadius: '4px', border: 'none', backgroundColor: 'var(--primary-accent)', color: 'white', cursor: 'pointer' }}
        >
          Subscribe
        </button>
      </form>
    </div>
  );
}

// Exports the component for use in other files (like App.js).
export default AlertSubscriptionCard;