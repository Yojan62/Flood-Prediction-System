import React from 'react';
// Imports necessary components from react-leaflet library.
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
// Imports Leaflet's CSS for map styling.
import 'leaflet/dist/leaflet.css';
// Imports the main Leaflet library object.
import L from 'leaflet';

// Helper function to create a Leaflet divIcon using an emoji.
const createEmojiIcon = (emoji) => {
    // Returns a Leaflet divIcon configured to display the provided emoji.
    return L.divIcon({
        html: `<span style="font-size: 24px;">${emoji}</span>`, // The HTML content (the emoji wrapped in a span).
        className: 'leaflet-emoji-icon', // Assigns a CSS class for custom styling (e.g., removing default background).
        iconSize: [24, 24], // Sets the pixel size of the icon container.
        iconAnchor: [12, 24], // Sets the anchor point of the icon relative to its top-left corner (bottom-center).
        popupAnchor: [0, -24] // Sets the point where the popup opens relative to the iconAnchor (centered above).
    });
};

// Defines the MapCard component, which displays the interactive map.
// It accepts the current 'theme' ('light' or 'dark') as a prop.
function MapCard({ theme }) {
    // Sets the geographical coordinates (latitude, longitude) for Dhaka.
    const position = [23.8103, 90.4125];

    // Defines the URL and attribution text for the light theme map tiles (OpenStreetMap).
    const lightMapUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    const lightMapAttribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    // Defines the URL and attribution text for the dark theme map tiles (Stadia Maps).
    const darkMapUrl = "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png";
    const darkMapAttribution = '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

    // Selects the appropriate map URL based on the current theme prop.
    const mapUrl = theme === 'dark' ? darkMapUrl : lightMapUrl;
    // Selects the appropriate attribution text based on the current theme prop.
    const mapAttribution = theme === 'dark' ? darkMapAttribution : lightMapAttribution;

    // Creates the appropriate marker icon (emoji) based on the current theme prop.
    const markerIcon = theme === 'dark' ? createEmojiIcon('🌟') : createEmojiIcon('📍');

    // Returns the JSX structure for the map card.
    return (
        // Uses the generic 'card' class and a specific ID for styling.
        <div className="card" id="map-card">
            <h2>Monitored Location: Dhaka, Bangladesh</h2>
            {/* Renders the Leaflet map container. */}
            {/* The 'key' prop forces a re-render when the theme changes, ensuring tile updates. */}
            <MapContainer key={theme} center={position} zoom={11} style={{ height: '400px', width: '100%' }}>
                {/* Adds the background map tiles using the selected URL and attribution. */}
                <TileLayer
                    attribution={mapAttribution}
                    url={mapUrl}
                />
                {/* Places a marker on the map at the specified position using the custom icon. */}
                <Marker position={position} icon={markerIcon}>
                    {/* Creates a popup that appears when the marker is clicked. */}
                    {/* Applies custom CSS classes for theme-aware styling. */}
                    <Popup className={`custom-popup ${theme}-theme-popup`}>
                        Dhaka, Bangladesh <br /> Monitoring Station Alpha.
                    </Popup>
                </Marker>
            </MapContainer>
        </div>
    );
}

// Exports the MapCard component for use in App.js.
export default MapCard;