const { override, addJestBabelPlugin } = require('customize-cra');

module.exports = override(
  // This is a common override for Jest in CRA to handle ES module imports
  // in node_modules, specifically for libraries like react-leaflet.
  // It essentially adds the babel-plugin-transform-es2015-modules-commonjs
  // for specific problematic node_modules packages.
  // This is *not* directly setting transformIgnorePatterns, but configuring
  // Babel to transform specific modules when Jest runs.

  // We don't directly manipulate transformIgnorePatterns here because customize-cra
  // has its own way of handling Jest config overrides.
  // Instead, we use a Babel plugin to transform the problematic modules.

  // As of customize-cra, directly overriding Jest's transformIgnorePatterns
  // can be tricky. A more reliable way is to ensure Babel processes the modules.
  // However, addJestBabelPlugin is more for custom babel plugins.

  // Let's try the direct Jest config modification that customize-cra offers.
  (config) => {
    // Find the Jest configuration part of the CRA webpack config
    // This is usually found under config.jest
    // CRA v5 stores Jest config in its default config.
    // We'll try to directly modify its transformIgnorePatterns

    // If CRA's internal Jest config is not directly exposed for modification
    // in this way by customize-cra, this might still fail.

    // The correct way to modify Jest config with customize-cra is usually through
    // a function that returns the Jest config.
    // Let's assume 'config' passed here *is* the Jest config for now,
    // or a part of the webpack config that includes it.

    if (!config.transformIgnorePatterns) {
      config.transformIgnorePatterns = [];
    }

    // Add the react-leaflet related paths to be transformed (i.e., NOT ignored)
    config.transformIgnorePatterns.push(
      "node_modules/(?!(react-leaflet|leaflet|@react-leaflet)/)"
    );

    // A better pattern for react-leaflet specifically.
    // This tells Jest to *not ignore* the following modules for transformation.
    // It means these modules *will* be run through Babel.
    config.transformIgnorePatterns = [
      "/node_modules/(?!(react-leaflet|leaflet|@react-leaflet)/)",
    ];


    return config;
  }
);