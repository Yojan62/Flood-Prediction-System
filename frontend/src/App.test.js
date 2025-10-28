import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the Flood Forecasting Dashboard title', () => {
  render(<App />);
  const titleElement = screen.getByText(/Flood Forecasting Dashboard/i); // Case-insensitive match
  expect(titleElement).toBeInTheDocument();
});