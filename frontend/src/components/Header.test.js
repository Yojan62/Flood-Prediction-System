// Imports necessary functions from React Testing Library and Jest-DOM.
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Imports the Header component that this file will test.
import Header from './Header';

// Defines a test suite, grouping tests related to the Header component.
describe('Header Component', () => {

  // Defines a specific test case within the suite.
  test('renders the correct title', () => {
    // Renders the Header component into a virtual DOM for testing.
    render(<Header />);

    // Finds an element in the rendered output whose text content matches
    // the regular expression /Flood Forecasting Dashboard/i (case-insensitive).
    // Stores the found element in the titleElement variable.
    const titleElement = screen.getByText(/Flood Forecasting Dashboard/i);

    // Asserts that the titleElement was successfully found within the rendered component.
    expect(titleElement).toBeInTheDocument();
  });

});