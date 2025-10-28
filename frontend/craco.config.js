module.exports = {
  jest: {
    configure: (jestConfig, { env, paths }) => {
      // Log the original transformIgnorePatterns to debug
      console.log("Original Jest transformIgnorePatterns:", jestConfig.transformIgnorePatterns);

      // Ensure it's an array for manipulation
      if (!Array.isArray(jestConfig.transformIgnorePatterns)) {
        // If it's a string, convert it to an array. If undefined, initialize.
        jestConfig.transformIgnorePatterns = jestConfig.transformIgnorePatterns
          ? [jestConfig.transformIgnorePatterns]
          : [];
      }

      // Filter out the default node_modules ignore, then add our more specific one.
      // This ensures that we don't accidentally re-add a generic /node_modules/ ignore.
      // The goal is to INCLUDE react-leaflet for transformation.
      // Regex breakdown:
      // /node_modules/ - Matches node_modules directory
      // (?!...)        - Negative lookahead (do NOT match if followed by...)
      // (react-leaflet|leaflet|@react-leaflet) - specific packages to EXCLUDE from being ignored
      // /              - Matches the / after the package name (e.g., node_modules/react-leaflet/)
      // So, it *ignores* node_modules UNLESS it's one of react-leaflet, leaflet, @react-leaflet.

      jestConfig.transformIgnorePatterns = [
        "/node_modules/(?!react-leaflet|leaflet|@react-leaflet)/",
      ];

      // For good measure, explicitly define the transform for js/jsx/ts/tsx if it's missing or modified.
      // CRA's default typically uses babel-jest for these.
      if (!jestConfig.transform) {
          jestConfig.transform = {};
      }
      if (!jestConfig.transform['^.+\\.(js|jsx|mjs|cjs|ts|tsx)$']) {
        jestConfig.transform['^.+\\.(js|jsx|mjs|cjs|ts|tsx)$'] = require.resolve('babel-jest');
      }

      console.log("Modified Jest transformIgnorePatterns:", jestConfig.transformIgnorePatterns);
      return jestConfig;
    },
  },
};