// Imports testing utilities from React Testing Library and Jest-DOM.
import React from 'react';
// Imports 'act' to handle asynchronous state updates from API calls.
import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';

// Imports the component to be tested.
import WeatherWidget from './WeatherWidget';
// Imports axios so we can mock it.
import axios from 'axios';

// --- Mocking External Libraries ---

// Mocks the entire axios library.
jest.mock('axios');

// --- Mock Data ---
// Creates mock data to simulate a successful API response for Dhaka.
const mockDhakaData = {
  data: {
    name: 'Dhaka',
    coord: { lat: 23.8103, lon: 90.4125 },
    weather: [{ description: 'haze', icon: '50d' }],
    main: { temp: 28.99 }
  }
};
// Creates mock data to simulate a successful API response for London.
const mockLondonData = {
  data: {
    name: 'London',
    coord: { lat: 51.5074, lon: -0.1278 },
    weather: [{ description: 'overcast clouds', icon: '04d' }],
    main: { temp: 15.0 }
  }
};

// --- Test Suite for WeatherWidget ---
describe('WeatherWidget', () => {

  // Creates mock functions for the props.
  const mockOnCityChange = jest.fn();
  const mockOnCoordsChange = jest.fn();

  // Clears all mock function call history before each test.
  beforeEach(() => {
    mockOnCityChange.mockClear();
    mockOnCoordsChange.mockClear();
    axios.get.mockClear();
    // Silences console.error for expected API errors (like 404).
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  // Restores console.error after each test.
  afterEach(() => {
    console.error.mockRestore();
  });

  // Test case 1: Checks if the component loads and displays the initial city (Dhaka).
  test('renders with initial city (Dhaka) and shows weather', async () => {
    // Configures the mock axios.get to return the Dhaka data.
    axios.get.mockResolvedValue(mockDhakaData);
    
    // Renders the component with its required props.
    await act(async () => {
      render(
        <WeatherWidget 
          initialCity="Dhaka" 
          onCityChange={mockOnCityChange} 
          onCoordsChange={mockOnCoordsChange} 
        />
      );
    });
    
    // Checks that the 'Loading' message is gone.
    expect(screen.queryByText(/Loading weather.../i)).not.toBeInTheDocument();
    
    // Checks that the weather data for 'Dhaka' is displayed correctly.
    expect(screen.getByText(/in Dhaka/i)).toBeInTheDocument();
    expect(screen.getByText(/haze/i)).toBeInTheDocument();
    expect(screen.getByText(/29°C/i)).toBeInTheDocument(); // 28.99 rounded
    
    // Checks that axios was called once with 'Dhaka'.
    expect(axios.get).toHaveBeenCalledTimes(1);
    expect(axios.get).toHaveBeenCalledWith(
      expect.stringContaining('q=Dhaka')
    );
    // Checks that the onCoordsChange prop was called with Dhaka's coordinates.
    expect(mockOnCoordsChange).toHaveBeenCalledWith([23.8103, 90.4125]);
  });

  // Test case 2: Simulates a user searching for a new city.
  test('calls onCityChange prop when user searches for a new city', async () => {
    // Mocks the first call (for Dhaka on load).
    axios.get.mockResolvedValue(mockDhakaData);
    
    await act(async () => {
      render(
        <WeatherWidget 
          initialCity="Dhaka" 
          onCityChange={mockOnCityChange} 
          onCoordsChange={mockOnCoordsChange} 
        />
      );
    });

    // Finds the input field and search button.
    const searchInput = screen.getByPlaceholderText(/enter city name/i);
    const searchButton = screen.getByRole('button', { name: /search/i });

    // Simulates user typing 'London'.
    await act(async () => {
      fireEvent.change(searchInput, { target: { value: 'London' } });
    });
    
    // Simulates user clicking 'Search'.
    await act(async () => {
      fireEvent.click(searchButton);
    });

    // Asserts that the 'onCityChange' prop function was called with 'London'.
    // This confirms the component correctly tells App.js to update the city.
    expect(mockOnCityChange).toHaveBeenCalledWith('London');
  });

  // Test case 3: Checks that an error message is shown if the city is not found.
  test('displays an error message if the city is not found', async () => {
    // Mocks the API call to reject with a 404 error.
    axios.get.mockRejectedValue({
      response: { status: 404 }
    });

    // Renders the component with a city that will trigger the mocked error.
    await act(async () => {
      render(
        <WeatherWidget 
          initialCity="FakeCity" 
          onCityChange={mockOnCityChange} 
          onCoordsChange={mockOnCoordsChange} 
        />
      );
    });

    // Checks that the "City not found" error message is displayed.
    expect(screen.getByText(/City not found: FakeCity/i)).toBeInTheDocument();
  });
});