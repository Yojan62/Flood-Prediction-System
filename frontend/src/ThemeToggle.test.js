// Imports testing utilities from React Testing Library and Jest-DOM.
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// Imports the component to be tested.
import ThemeToggle from './ThemeToggle';
// Imports the specific icons to check for their presence.
import { FaSun, FaMoon } from 'react-icons/fa';

// Mocks the react-icons library to simplify testing.
// We replace the complex icon components with simple text.
jest.mock('react-icons/fa', () => ({
  // Creates a mock FaMoon component that just renders "Moon Icon".
  FaMoon: () => <span data-testid="moon-icon">Moon Icon</span>,
  // Creates a mock FaSun component that just renders "Sun Icon".
  FaSun: () => <span data-testid="sun-icon">Sun Icon</span>,
}));

// Defines a test suite for the ThemeToggle component.
describe('ThemeToggle', () => {

  // Creates a mock function (a "spy") for the toggleTheme prop.
  // This allows tracking if the function was called.
  const mockToggleTheme = jest.fn();

  // Clears the mock function's call history before each test.
  beforeEach(() => {
    mockToggleTheme.mockClear();
  });

  // Test case 1: Checks if the Moon icon renders when the theme is 'light'.
  test('renders the Moon icon when theme is light', () => {
    // Renders the component with the theme set to 'light'.
    render(<ThemeToggle theme="light" toggleTheme={mockToggleTheme} />);

    // Checks if the "Moon Icon" (our mock) is present in the document.
    expect(screen.getByTestId('moon-icon')).toBeInTheDocument();
    // Checks that the "Sun Icon" (our mock) is NOT present.
    expect(screen.queryByTestId('sun-icon')).not.toBeInTheDocument();
  });

  // Test case 2: Checks if the Sun icon renders when the theme is 'dark'.
  test('renders the Sun icon when theme is dark', () => {
    // Renders the component with the theme set to 'dark'.
    render(<ThemeToggle theme="dark" toggleTheme={mockToggleTheme} />);

    // Checks if the "Sun Icon" is present in the document.
    expect(screen.getByTestId('sun-icon')).toBeInTheDocument();
    // Checks that the "Moon Icon" is NOT present.
    expect(screen.queryByTestId('moon-icon')).not.toBeInTheDocument();
  });

  // Test case 3: Checks if the toggle function is called on click.
  test('calls toggleTheme function on button click', () => {
    // Renders the component.
    render(<ThemeToggle theme="light" toggleTheme={mockToggleTheme} />);

    // Finds the main button element (the switch track) by its role.
    const toggleButton = screen.getByRole('button');

    // Simulates a user clicking the button.
    fireEvent.click(toggleButton);

    // Asserts that the mock toggle function was called exactly one time.
    expect(mockToggleTheme).toHaveBeenCalledTimes(1);
  });

});