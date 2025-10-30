import React from 'react';
// Imports testing utilities from React Testing Library.
import { render, screen, within } from '@testing-library/react';
// Imports Jest-DOM for extended matchers like .toBeInTheDocument().
import '@testing-library/jest-dom';
// Imports the MapCard component to be tested.
import MapCard from './MapCard';
// Imports the Leaflet library object to access the mocked 'divIcon'.
import L from 'leaflet';

// --- Mocking External Libraries ---

// Mocks the react-leaflet library to prevent actual map rendering in the test environment (JSDOM).
// Replaces actual components with simplified divs containing data-testid attributes for querying.
jest.mock('react-leaflet', () => ({
  MapContainer: ({ children, center, zoom }) => (
    // Mock MapContainer passes center and zoom props to data attributes.
    <div data-testid="mock-map-container" data-center={JSON.stringify(center)} data-zoom={zoom}>
      {children}
    </div>
  ),
  TileLayer: ({ url, attribution }) => (
    // Mock TileLayer passes url and attribution props to data attributes.
    <div data-testid="mock-tile-layer" data-url={url} data-attribution={attribution} />
  ),
  // Mock Marker renders children and attempts to render icon HTML for testing icon content.
  Marker: ({ children, position, icon }) => {
    // Extracts the HTML content from the expected structure of the mocked L.divIcon.
    const iconHtmlContent = icon?.options?.html || null;
    return (
      <div data-testid="mock-marker" data-position={JSON.stringify(position)}>
        {/* Renders the icon HTML directly if it was correctly passed. */}
        {iconHtmlContent && <div dangerouslySetInnerHTML={{ __html: iconHtmlContent }} />}
        {children} {/* Renders nested components like Popup. */}
      </div>
    );
  },
  Popup: ({ children, className }) => (
    // Mock Popup passes className and renders children.
    <div data-testid="mock-popup" className={className}>
      {children}
    </div>
  ),
  // Mocks react-leaflet hooks to prevent errors, returning empty functions or basic values.
  useMap: jest.fn(() => ({})),
  useMapEvent: jest.fn(() => jest.fn()),
  useMapEvents: jest.fn(() => jest.fn()),
}));

// Mocks the Leaflet library itself, specifically replacing L.divIcon with a Jest mock function.
jest.mock('leaflet', () => {
    // Retains other exports from the actual Leaflet library.
    const actualLeaflet = jest.requireActual('leaflet');
    return {
        ...actualLeaflet,
        // Replaces L.divIcon with a Jest mock function.
        // This mock returns an object containing the options passed to it,
        // allowing inspection of the 'html' option in tests.
        divIcon: jest.fn((options) => ({ options })),
        // Mocks the default Leaflet icon setup to prevent errors in JSDOM environment.
        Icon: {
            Default: jest.fn().mockImplementation(() => ({
                options: {},
                _getIconUrl: jest.fn(() => 'mock-marker-icon.png'),
            })),
        },
        // Provides a minimal mock for L.marker if needed.
        marker: jest.fn(() => ({ setIcon: jest.fn(), bindPopup: jest.fn(), addTo: jest.fn() })),
    };
});

// --- Test Suite for MapCard ---
describe('MapCard', () => {
  // Clears any previous calls to mocked functions before each test runs.
  beforeEach(() => {
    L.divIcon.mockClear();
  });

  // Defines default props to pass to MapCard in tests.
  const defaultProps = {
    theme: 'light',
  };

  test('renders the map card with the correct title', () => {
    // Renders the MapCard component with default props.
    render(<MapCard theme={defaultProps.theme} />);
    // Asserts that the expected title text is present in the document.
    expect(screen.getByText(/Monitored Location: Dhaka, Bangladesh/i)).toBeInTheDocument();
  });

  test('renders the mock MapContainer with correct center and zoom', () => {
    // Renders the MapCard component.
    render(<MapCard theme={defaultProps.theme} />);
    // Finds the mocked MapContainer via its test ID.
    const mapContainer = screen.getByTestId('mock-map-container');
    // Asserts the mock container exists.
    expect(mapContainer).toBeInTheDocument();
    // Asserts the mock container received the correct 'center' prop (as a stringified attribute).
    expect(mapContainer).toHaveAttribute('data-center', JSON.stringify([23.8103, 90.4125]));
    // Asserts the mock container received the correct 'zoom' prop.
    expect(mapContainer).toHaveAttribute('data-zoom', "11");
  });

  test('renders the mock TileLayer with the correct light theme URL', () => {
    // Renders the MapCard specifically with the light theme.
    render(<MapCard theme="light" />);
    // Finds the mocked TileLayer via its test ID.
    const tileLayer = screen.getByTestId('mock-tile-layer');
    // Asserts the mock layer exists.
    expect(tileLayer).toBeInTheDocument();
    // Asserts the mock layer received the correct URL for the light theme.
    expect(tileLayer).toHaveAttribute('data-url', 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png');
  });

   test('renders the mock TileLayer with the correct dark theme URL', () => {
    // Renders the MapCard specifically with the dark theme.
    render(<MapCard theme="dark" />);
    // Finds the mocked TileLayer.
    const tileLayer = screen.getByTestId('mock-tile-layer');
    // Asserts the mock layer exists.
    expect(tileLayer).toBeInTheDocument();
    // Asserts the mock layer received the correct URL for the dark theme.
    expect(tileLayer).toHaveAttribute('data-url', 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png');
  });


  test('renders the mock Marker at the correct position', () => {
    // Renders the MapCard component.
    render(<MapCard theme={defaultProps.theme} />);
    // Finds the mocked Marker via its test ID.
    const marker = screen.getByTestId('mock-marker');
    // Asserts the mock marker exists.
    expect(marker).toBeInTheDocument();
    // Asserts the mock marker received the correct 'position' prop (as a stringified attribute).
    expect(marker).toHaveAttribute('data-position', JSON.stringify([23.8103, 90.4125]));
  });

  test('renders the Popup with the correct text', () => {
    // Renders the MapCard component.
    render(<MapCard theme={defaultProps.theme} />);
    // Finds the mocked Popup via its test ID.
    const popup = screen.getByTestId('mock-popup');
    // Asserts the mock popup exists.
    expect(popup).toBeInTheDocument();
    // Asserts the popup contains the expected text content.
    expect(popup).toHaveTextContent("Dhaka, Bangladesh");
    expect(popup).toHaveTextContent("Monitoring Station Alpha.");
  });

  // Tests verifying the component's internal logic for selecting the icon based on theme.
  test('calls L.divIcon with correct emoji HTML for light theme', () => {
    // Renders the MapCard with the light theme.
    render(<MapCard theme="light" />);
    // Asserts that the mocked L.divIcon function was called exactly once.
    expect(L.divIcon).toHaveBeenCalledTimes(1);
    // Asserts that L.divIcon was called with an options object containing the correct HTML string for the light theme emoji.
    expect(L.divIcon).toHaveBeenCalledWith(expect.objectContaining({
        html: '<span style="font-size: 24px;">📍</span>'
    }));
  });

   test('calls L.divIcon with correct emoji HTML for dark theme', () => {
    // Renders the MapCard with the dark theme.
    render(<MapCard theme="dark" />);
    // Asserts that the mocked L.divIcon function was called exactly once.
    expect(L.divIcon).toHaveBeenCalledTimes(1);
    // Asserts that L.divIcon was called with an options object containing the correct HTML string for the dark theme emoji.
    expect(L.divIcon).toHaveBeenCalledWith(expect.objectContaining({
        html: '<span style="font-size: 24px;">🌟</span>'
    }));
  });
});