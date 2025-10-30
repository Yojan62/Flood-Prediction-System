// Imports testing utilities from React Testing Library and Jest-DOM.
import React from 'react';
import { render, screen, fireEvent }from '@testing-library/react';
import '@testing-library/jest-dom';

// Imports the component to be tested.
import Search from './Search';

// Defines a test suite for the Search component.
describe('Search Component', () => {

  // Test case 1: Checks if the component renders with the correct initial value.
  test('renders the search bar with the initial city', () => {
    // Renders the component with a test 'initialCity' prop.
    // Provides an empty mock function for 'onCityChange' as it's a required prop.
    render(<Search initialCity="Dhaka" onCityChange={() => {}} />);

    // Finds the input field by its placeholder text.
    const inputElement = screen.getByPlaceholderText(/enter city name/i);
    // Checks that the input field's value is set to the 'initialCity' prop.
    expect(inputElement.value).toBe('Dhaka');

    // Checks if the "Search" button is present.
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
  });

  // Test case 2: Simulates user typing and submitting the form.
  test('allows user to type and calls onCityChange on submit', () => {
    // Creates a Jest mock function ("spy") to track calls to the 'onCityChange' prop.
    const mockOnCityChange = jest.fn();

    // Renders the component, passing the mock function as the prop.
    render(<Search initialCity="Dhaka" onCityChange={mockOnCityChange} />);

    // Finds the input field and search button.
    const inputElement = screen.getByPlaceholderText(/enter city name/i);
    const searchButton = screen.getByRole('button', { name: /search/i });

    // Simulates a user typing 'London' into the input field.
    fireEvent.change(inputElement, { target: { value: 'London' } });

    // Asserts that the input field's value has updated correctly.
    expect(inputElement.value).toBe('London');

    // Simulates the user clicking the search button.
    fireEvent.click(searchButton);

    // Asserts that the 'onCityChange' prop function was called exactly one time.
    expect(mockOnCityChange).toHaveBeenCalledTimes(1);
    // Asserts that the function was called with the correct argument ('London').
    expect(mockOnCityChange).toHaveBeenCalledWith('London');
  });

});