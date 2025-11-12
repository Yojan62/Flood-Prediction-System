// Imports testing utilities
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios'; // Import axios to mock it

// Imports the component to be tested.
import AlertSubscriptionCard from './AlertSubscriptionCard';

// --- 1. MOCK AXIOS ---
// This tells Jest to "fake" the axios library
jest.mock('axios');

// Defines a test suite for the AlertSubscriptionCard component.
describe('AlertSubscriptionCard', () => {

  // Test case 1: Checks if the component renders the main elements correctly.
  test('renders the subscription form elements', () => {
    // Renders the component with the required prop
    render(<AlertSubscriptionCard selectedLocationId={1} />);

    expect(screen.getByRole('heading', { name: /get flood alerts/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/enter your email address/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /subscribe/i })).toBeInTheDocument();
  });

  // --- 2. THIS IS THE NEW, CORRECT INTERACTION TEST ---
  test('allows user to type, submit, and see a success message', async () => {
    
    // Define what axios.post should "fake"
    const mockSuccessMessage = 'Subscribed successfully!';
    axios.post.mockResolvedValue({
      data: { message: mockSuccessMessage }
    });

    // Renders the component with the required prop
    render(<AlertSubscriptionCard selectedLocationId={1} />);

    const emailInput = screen.getByPlaceholderText(/enter your email address/i);
    const submitButton = screen.getByRole('button', { name: /subscribe/i });

    // Simulate typing
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
    expect(emailInput.value).toBe('test@example.com');

    // Simulate clicking the submit button
    fireEvent.click(submitButton);
    
    // Check that the button is disabled
    expect(submitButton).toBeDisabled();
    expect(screen.getByText(/subscribing.../i)).toBeInTheDocument();

    // --- This is the key ---
    // We "wait" for the success message to appear on the screen
    const successMessage = await screen.findByText(mockSuccessMessage);

    // Assert that the success message appeared
    expect(successMessage).toBeInTheDocument();
    // Assert that the input field was cleared
    expect(emailInput.value).toBe('');
    // Assert that the button is enabled again
    expect(submitButton).not.toBeDisabled();
  });
  
  // Test case 3: Check error handling
  test('shows an error message if the API call fails', async () => {
    
    // Tell axios to "fake" a 404 error
    axios.post.mockRejectedValue({
      response: { data: { detail: 'Location not found.' } }
    });

    render(<AlertSubscriptionCard selectedLocationId={99} />);

    const emailInput = screen.getByPlaceholderText(/enter your email address/i);
    const submitButton = screen.getByRole('button', { name: /subscribe/i });

    // Simulate typing and clicking
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
    fireEvent.click(submitButton);

    // Wait for the error message to appear
    const errorMessage = await screen.findByText(/Error: Location not found./i);
    expect(errorMessage).toBeInTheDocument();

    // The input should NOT be cleared on error
    expect(emailInput.value).toBe('test@example.com');
  });

});