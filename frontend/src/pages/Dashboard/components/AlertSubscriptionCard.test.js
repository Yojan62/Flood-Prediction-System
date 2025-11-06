// Imports testing utilities from React Testing Library and Jest-DOM.
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// Imports the component to be tested.
import AlertSubscriptionCard from './AlertSubscriptionCard';

// Defines a test suite for the AlertSubscriptionCard component.
describe('AlertSubscriptionCard', () => {

  // Test case 1: Checks if the component renders the main elements correctly.
  test('renders the subscription form elements', () => {
    // Renders the component.
    render(<AlertSubscriptionCard />);

    // Checks if the heading "Get Flood Alerts" is present.
    expect(screen.getByRole('heading', { name: /get flood alerts/i })).toBeInTheDocument();

    // Checks if the instructional paragraph is present.
    expect(screen.getByText(/enter your email to receive alerts/i)).toBeInTheDocument();

    // Checks if the email input field (identified by its placeholder) is present.
    expect(screen.getByPlaceholderText(/enter your email address/i)).toBeInTheDocument();

    // Checks if the "Subscribe" button is present.
    expect(screen.getByRole('button', { name: /subscribe/i })).toBeInTheDocument();
  });

  // Test case 2: Simulates user typing into the input and submitting the form.
  test('allows user to type email and clears input on submit', () => {
    // Mocks the window.alert function to prevent it from popping up during the test.
    jest.spyOn(window, 'alert').mockImplementation(() => {});

    // Renders the component.
    render(<AlertSubscriptionCard />);

    // Finds the email input field by its placeholder text.
    const emailInput = screen.getByPlaceholderText(/enter your email address/i);
    // Finds the submit button by its text.
    const submitButton = screen.getByRole('button', { name: /subscribe/i });

    // Simulates a user typing an email address into the input field.
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });

    // Asserts that the input field's value has been updated.
    expect(emailInput.value).toBe('test@example.com');

    // Simulates clicking the submit button.
    fireEvent.click(submitButton);

    // Asserts that the alert function was called (optional, confirms handleSubmit logic ran).
    expect(window.alert).toHaveBeenCalled();

    // Asserts that the input field's value was cleared after submission.
    expect(emailInput.value).toBe('');

    // Restores the original window.alert function after the test.
    window.alert.mockRestore();
  });

});