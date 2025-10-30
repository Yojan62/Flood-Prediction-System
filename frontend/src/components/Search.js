import React, { useState } from 'react';

// Defines the Search component.
// It receives the 'initialCity' and the 'onCityChange' function as props from App.js.
function Search({ initialCity, onCityChange }) {
  // State hook to manage the value of the search input field.
  const [cityInput, setCityInput] = useState(initialCity);

  // Event handler for the form submission.
  const handleSearch = (event) => {
    event.preventDefault(); // Prevents the page from reloading.
    // Calls the 'onCityChange' function (which is setCity in App.js)
    // to update the global 'city' state.
    onCityChange(cityInput);
  };

  return (
    // Uses the 'card' style for visual consistency.
    <div className="card" id="search-bar" style={{ gridColumn: '1 / -1', marginBottom: '30px' }}>
      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={cityInput}
          onChange={(e) => setCityInput(e.target.value)}
          placeholder="Enter city name to update map and weather"
          style={{ 
            flexGrow: 1, 
            padding: '10px', 
            borderRadius: '20px', 
            border: '1px solid #ccc' }}
        />
        <button
          type="submit"
          style={{ padding: '10px 15px', borderRadius: '4px', border: 'none', backgroundColor: 'var(--primary-accent)', color: 'white', cursor: 'pointer' }}
        >
          Search
        </button>
      </form>
    </div>
  );
}

export default Search;