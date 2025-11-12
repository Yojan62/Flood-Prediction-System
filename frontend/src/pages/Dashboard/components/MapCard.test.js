// src/pages/Dashboard/components/MapCard.test.js
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import MapCard from './MapCard';
import L from 'leaflet';

// --- Mocking External Libraries (This part is unchanged and correct) ---
jest.mock('react-leaflet', () => ({
  MapContainer: ({ children, center, zoom }) => (
    <div data-testid="mock-map-container" data-center={JSON.stringify(center)} data-zoom={zoom}>
      {children}
    </div>
  ),
  TileLayer: ({ url, attribution }) => (
    <div data-testid="mock-tile-layer" data-url={url} data-attribution={attribution} />
  ),
  Marker: ({ children, position, icon, eventHandlers }) => {
    const iconHtmlContent = icon?.options?.html || null;
    return (
      <div 
        data-testid="mock-marker" 
        data-position={JSON.stringify(position)}
        onClick={eventHandlers.click} // Pass the click handler
      >
        {iconHtmlContent && <div dangerouslySetInnerHTML={{ __html: iconHtmlContent }} />}
        {children}
      </div>
    );
  },
  Popup: ({ children, className }) => (
    <div data-testid="mock-popup" className={className}>
      {children}
    </div>
  ),
  useMap: jest.fn(() => ({})),
  useMapEvent: jest.fn(() => jest.fn()),
  useMapEvents: jest.fn(() => jest.fn()),
}));

jest.mock('leaflet', () => {
  const actualLeaflet = jest.requireActual('leaflet');
  return {
    ...actualLeaflet,
    divIcon: jest.fn((options) => ({ options })),
    Icon: {
      Default: jest.fn().mockImplementation(() => ({
        options: {},
        _getIconUrl: jest.fn(() => 'mock-marker-icon.png'),
      })),
    },
    marker: jest.fn(() => ({ setIcon: jest.fn(), bindPopup: jest.fn(), addTo: jest.fn() })),
  };
});
// --- End of Mocks ---


// --- Test Suite for MapCard (Updated) ---
describe('MapCard', () => {
  beforeEach(() => {
    L.divIcon.mockClear();
  });

  // --- 1. Define Mock Data ---
  const mockLocations = [
    { location_id: 1, name: 'Dhaka', latitude: 23.81, longitude: 90.41 },
    { location_id: 2, name: 'Chittagong', latitude: 22.35, longitude: 91.83 }
  ];
  const mockMapCenter = [23.81, 90.41];
  const mockOnMarkerClick = jest.fn(); // Mock function to track clicks

  // --- 2. Update Tests to Use Mock Data ---

  test('renders the map card with the correct title', () => {
    render(<MapCard theme="light" />);
    // Test for the NEW title
    expect(screen.getByText(/Monitored Locations/i)).toBeInTheDocument();
  });

  test('renders the mock MapContainer with correct center and zoom', () => {
    render(<MapCard theme="light" mapCenter={mockMapCenter} />);
    const mapContainer = screen.getByTestId('mock-map-container');
    
    expect(mapContainer).toBeInTheDocument();
    expect(mapContainer).toHaveAttribute('data-center', JSON.stringify(mockMapCenter));
    // Test for the NEW zoom level
    expect(mapContainer).toHaveAttribute('data-zoom', "7");
  });

  test('renders the mock TileLayer with the correct light/dark URLs', () => {
    // Test light theme
    const { rerender } = render(<MapCard theme="light" />);
    const tileLayer = screen.getByTestId('mock-tile-layer');
    expect(tileLayer).toHaveAttribute('data-url', 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png');

    // Re-render with dark theme
    rerender(<MapCard theme="dark" />);
    expect(tileLayer).toHaveAttribute('data-url', 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png');
  });

  test('renders the correct number of markers with correct popups', () => {
    render(<MapCard theme="light" locations={mockLocations} />);
    
    // Check for the correct NUMBER of markers
    const markers = screen.getAllByTestId('mock-marker');
    expect(markers).toHaveLength(2);

    // Check that the popups have the correct text from the props
    expect(screen.getByText('Dhaka')).toBeInTheDocument();
    expect(screen.getByText('Chittagong')).toBeInTheDocument();
  });

  test('calls onMarkerClick with the correct ID when a marker is clicked', () => {
    render(
      <MapCard 
        theme="light" 
        locations={mockLocations} 
        onMarkerClick={mockOnMarkerClick} 
      />
    );

    // Find the first marker (Dhaka)
    const firstMarker = screen.getAllByTestId('mock-marker')[0];
    
    // Simulate a click
    fireEvent.click(firstMarker);

    // Assert that our mock function was called with the correct ID
    expect(mockOnMarkerClick).toHaveBeenCalledTimes(1);
    expect(mockOnMarkerClick).toHaveBeenCalledWith(1); // 1 is the location_id for Dhaka
  });

  test('highlights the selected marker', () => {
    render(
      <MapCard 
        theme="light" 
        locations={mockLocations} 
        selectedLocationId={2} // <-- Chittagong is selected
      />
    );
    
    // The createEmojiIcon mock will add a 'leaflet-emoji-icon-selected' class
    // We check that the icon HTML contains that class
    const chittagongIcon = L.divIcon.mock.results[1].value; // Get the 2nd icon
    expect(chittagongIcon.options.html).toContain('span');
    expect(chittagongIcon.options.className).toContain('leaflet-emoji-icon-selected');

    // Check that the first icon (Dhaka) is NOT selected
    const dhakaIcon = L.divIcon.mock.results[0].value;
    expect(dhakaIcon.options.className).not.toContain('leaflet-emoji-icon-selected');
  });

});