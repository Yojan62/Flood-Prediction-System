// src/components/ui/Search.js
import React, { useState, useEffect } from 'react';
import './Search.css'; // <-- 1. IMPORT THE NEW CSS FILE

function Search({ initialCity, onCityChange }) {
  const [cityInput, setCityInput] = useState(initialCity);

  // This syncs the input if the prop changes
  useEffect(() => {
    setCityInput(initialCity);
  }, [initialCity]);

  const handleSearch = (event) => {
    event.preventDefault(); 
    onCityChange(cityInput);
  };

  return (
    // 2. USE THE NEW CSS CLASSES
    <div className="card search-card" id="search-bar">
      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          className="search-input" // <-- Use class
          value={cityInput}
          onChange={(e) => setCityInput(e.target.value)}
          placeholder="Enter city name to update map and weather"
        />
        <button type="submit" className="search-button"> {/* <-- Use class */}
          Search
        </button>
      </form>
    </div>
  );
}

export default Search;