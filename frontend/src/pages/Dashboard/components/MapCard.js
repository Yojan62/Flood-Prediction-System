import React from 'react';
// Imports necessary components from the react-leaflet library.
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
// Imports Leaflet's CSS for correct map styling.
import 'leaflet/dist/leaflet.css';
// Imports the main Leaflet library object.
import L from 'leaflet';

// Helper function to create a Leaflet divIcon using an emoji.
const createEmojiIcon = (emoji) => {
    // Returns a Leaflet divIcon configured to display the provided emoji.
    return L.divIcon({
        html: `<span style="font-size: 24px;">${emoji}</span>`, // The HTML content (the emoji).
        className: 'leaflet-emoji-icon', // Assigns a CSS class for custom styling.
        iconSize: [24, 24], // Sets the pixel size of the icon container.
        iconAnchor: [12, 24], // Sets the anchor point of the icon (bottom-center).
        popupAnchor: [0, -24] // Sets the point where the popup opens (centered above).
    });
};

// Defines the MapCard component.
// It accepts the current 'theme', the list of 'locations', and the 'mapCenter' coordinates as props.
function MapCard({ theme, locations, mapCenter }) {
    // Sets a default zoom level.
    const mapZoom = 11; // Use 11 for a city-level zoom, 7 for a wider view

    // Defines the URL and attribution text for the light theme map tiles.
    const lightMapUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    const lightMapAttribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    // Defines the URL and attribution text for the dark theme map tiles.
    const darkMapUrl = "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png";
    const darkMapAttribution = '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

    // Selects the appropriate map style based on the current theme prop.
    const mapUrl = theme === 'dark' ? darkMapUrl : lightMapUrl;
    // Selects the appropriate attribution text based on the current theme prop.
    const mapAttribution = theme === 'dark' ? darkMapAttribution : lightMapAttribution;

    // Creates the appropriate marker icon (emoji) based on the current theme prop.
    const markerIcon = theme === 'dark' ? createEmojiIcon('🌟') : createEmojiIcon('📍');

    // Returns the JSX structure for the map card.
    return (
        // Uses the generic 'card' class and a specific ID for styling.
        <div className="card" id="map-card">
            <h2>Monitored Locations</h2>
            {/* Renders the Leaflet map container. */}
            {/* The 'key' prop forces a re-render when theme or mapCenter changes. */}
            {/* The 'center' prop is now dynamic, controlled by state in App.js. */}
            <MapContainer key={theme + mapCenter.toString()} center={mapCenter} zoom={mapZoom} style={{ height: '400px', width: '100%' }}>
                {/* Adds the background map tiles. */}
                <TileLayer
                    attribution={mapAttribution}
                    url={mapUrl}
                />
                
                {/* Loops over the 'locations' prop array (fetched from the backend). */}
                {/* For each location, creates a Marker component. */}
                {locations.map(location => (
                    <Marker
                        key={location.location_id} // Uses the unique ID for the React key.
                        position={[location.latitude, location.longitude]} // Uses the location's coordinates.
                        icon={markerIcon} // Uses the theme-appropriate icon.
                    >
                        {/* Creates a popup that appears when the marker is clicked. */}
                        <Popup className={`custom-popup ${theme}-theme-popup`}>
                            {/* Displays the location's name in the popup. */}
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

// Sets default props in case 'locations' or 'mapCenter' are not provided.
MapCard.defaultProps = {
    locations: [],
    mapCenter: [23.8103, 90.4125] // Default to Dhaka if no prop is passed.
};

// Exports the MapCard component for use in App.js.
export default MapCard;