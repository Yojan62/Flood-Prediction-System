import React from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Bangladesh bounds
const bounds = [
  [20.34, 88.01], // SW
  [26.65, 92.67], // NE
];

// Helper to build a coloured dot icon
const createDotIcon = (color, isSelected = false) =>
  L.divIcon({
    html: `<span class="map-marker-dot ${
      isSelected ? "map-marker-dot-selected" : ""
    }" style="background:${color};"></span>`,
    className: "map-marker-wrapper",
    iconSize: [32, 32], // bigger click target
    iconAnchor: [16, 32],
    popupAnchor: [0, -28],
  });

function MapCard({
  theme,
  locations = [],
  mapCenter = [23.8103, 90.4125],
  onMarkerClick = () => {},
  selectedLocationId,
  loading = false,
}) {
  const mapZoom = 7;

  const lightMapUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  const darkMapUrl =
    "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png";

  const mapUrl = theme === "dark" ? darkMapUrl : lightMapUrl;
  const mapAttribution =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

  // --- COLOR FIX ---
  // Use the specific hex codes for your theme
  // Forest Green (#15803d) for Light, Electric Blue (#00BFFF) for Dark
  const getIcon = (location) => {
    const isSelected = location.location_id === selectedLocationId;
    const color = theme === "dark" ? "#00BFFF" : "#15803d"; 
    return createDotIcon(color, isSelected);
  };

  return (
    <div className="card" id="map-card">
      <h2>
        Monitored Locations{" "}
        {locations.length ? `(${locations.length})` : loading ? "(…)" : "(0)"}
      </h2>

      <MapContainer
        key={theme + mapCenter.toString()}
        center={mapCenter}
        zoom={mapZoom}
        minZoom={6}
        maxZoom={12}
        maxBounds={bounds}
        maxBoundsViscosity={1.0}
        style={{ height: "60vh", width: "100%" }}
        scrollWheelZoom={true}
      >
        <TileLayer attribution={mapAttribution} url={mapUrl} />

        {locations.map((location) => (
          <Marker
            key={location.location_id}
            position={[location.latitude, location.longitude]}
            icon={getIcon(location)}
            eventHandlers={{
              click: () => onMarkerClick(location.location_id),
            }}
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