import React from 'react';
// Import necessary components from react-leaflet
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is imported

import L from 'leaflet';
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});


function MapCard() {
  // Coordinates for Dhaka, Bangladesh
  const position = [23.8103, 90.4125];

  return (
    <div className="card" id="map-card">
      <h2>Monitored Location: Dhaka, Bangladesh</h2>
      {/* The MapContainer component renders the map */}
      <MapContainer center={position} zoom={11} style={{ height: '400px', width: '100%' }}>
        {/* TileLayer provides the map background image */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {/* Marker places a pin at the specified position */}
        <Marker position={position}>
          {/* Popup shows information when the marker is clicked */}
          <Popup>
            Dhaka, Bangladesh <br /> Monitoring Station Alpha.
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  );
}

export default MapCard;