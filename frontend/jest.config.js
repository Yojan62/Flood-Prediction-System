module.exports = {
  // Use the default create-react-app test environment
  testEnvironment: 'jsdom',

  // Setup file for Jest DOM extensions (like .toBeInTheDocument())
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.js'],

  // Define how to transform files.
  // This tells Jest to use babel-jest for JS/JSX files
  // and for modules that are typically ignored.
  transform: {
    "^.+\\.(js|jsx|mjs|cjs|ts|tsx)$": "<rootDir>/node_modules/babel-jest",
    "^.+\\.css$": "<rootDir>/config/jest/cssTransform.js", // If you have CSS modules, add this
    "^(?!.*\\.(js|jsx|mjs|cjs|ts|tsx|css|json)$)": "<rootDir>/config/jest/fileTransform.js", // For other assets
  },

  // Crucial: Specify which modules Jest *should* transform in node_modules
  transformIgnorePatterns: [
    "node_modules/(?!(react-leaflet|leaflet|@react-leaflet)/)",
    // If other node_modules packages cause issues, add them here:
    // "node_modules/(?!other-problematic-package)/"
  ],

  // Module file extensions for importing
  moduleFileExtensions: ['js', 'json', 'jsx', 'node', 'mjs', 'ts', 'tsx'],

  // Optional: If you use absolute imports (e.g., import Component from 'components/Component'), configure here
  // moduleNameMapper: {
  //   "^@src/(.*)$": "<rootDir>/src/$1"
  // },

  // Optional: Coverage settings
  // collectCoverageFrom: ['src/**/*.{js,jsx,ts,tsx}', '!src/**/*.d.ts'],
  // coveragePathIgnorePatterns: ['/node_modules/', '/build/'],
};