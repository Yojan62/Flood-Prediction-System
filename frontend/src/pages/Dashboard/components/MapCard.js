import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Helper function to create a Leaflet divIcon
const createEmojiIcon = (emoji, isSelected = false) => {
  // Add a special class if the marker is selected
  const selectedClass = isSelected ? 'leaflet-emoji-icon-selected' : '';
  
  return L.divIcon({
    html: `<span style="font-size: 24px;">${emoji}</span>`,
    className: `leaflet-emoji-icon ${selectedClass}`, // Add the new class
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -24]
  });
};

// Define the boundaries for Bangladesh
const bounds = [
  [20.34, 88.01], // Southwest corner
  [26.65, 92.67]  // Northeast corner
]

// Defines the MapCard component.
// --- IT NOW ACCEPTS NEW PROPS ---
function MapCard({ theme, locations = [], mapCenter = [23.8103, 90.4125], onMarkerClick = () => {}, selectedLocationId }) {
  
  const mapZoom = 7;

  const lightMapUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  const lightMapAttribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
  const darkMapUrl = "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png";
  const darkMapAttribution = '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

  const mapUrl = theme === 'dark' ? darkMapUrl : lightMapUrl;
  const mapAttribution = theme === 'dark' ? darkMapAttribution : lightMapAttribution;

  // --- THIS LOGIC IS NEW ---
  // Function to get the right icon (selected or default)
  const getIcon = (location) => {
    const isSelected = location.location_id === selectedLocationId;
    const emoji = theme === 'dark' ? '🌟' : '📍';
    return createEmojiIcon(emoji, isSelected);
  };

  return (
    <div className="card" id="map-card">
      <h2>Monitored Locations</h2>
      <MapContainer key={theme + mapCenter.toString()} center={mapCenter} zoom={mapZoom} style={{ height: '60vh', width: '100%' }}>
        <TileLayer
          attribution={mapAttribution}
          url={mapUrl}
        />
        
        {locations.map(location => (
          <Marker
            key={location.location_id}
            position={[location.latitude, location.longitude]}
            icon={getIcon(location)} // Use the new function to set the icon
            
            // --- THIS IS THE CLICK HANDLER ---
            // When a marker is clicked, call the 'onMarkerClick' function
            // that was passed down from Dashboard.js
            eventHandlers={{
              click: () => {
                onMarkerClick(location.location_id);
              },
            }}
            // --- END OF CLICK HANDLER ---
          >
            <Popup className={`custom-popup ${theme}-theme-popup`}>
              <strong>{location.name}</strong>
              <br />
              Lat: {location.latitude}, Lng: {location.longitude}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}

export default MapCard;