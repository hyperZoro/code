const { searchFlights: searchMockFlights } = require("./mockFlightProvider");
const { searchFlights: searchSerpApiFlights, isSerpApiConfigured } = require("./serpApiFlightProvider");

async function searchFlights(request) {
  const requestedProvider = String(process.env.FLIGHT_PROVIDER || "mock").toLowerCase();

  if (requestedProvider === "serpapi") {
    if (!isSerpApiConfigured()) {
      return {
        provider: "mock",
        requestedProvider: "serpapi",
        warning: "SERPAPI_API_KEY is missing, so mock results were used.",
        itineraries: await searchMockFlights(request),
        searchMeta: { datePairsSearched: 0, partialErrors: [] }
      };
    }

    return searchSerpApiFlights(request);
  }

  return {
    provider: "mock",
    requestedProvider,
    itineraries: await searchMockFlights(request),
    searchMeta: { datePairsSearched: 0, partialErrors: [] }
  };
}

function providerHealth() {
  const requestedProvider = String(process.env.FLIGHT_PROVIDER || "mock").toLowerCase();
  return {
    activeProvider: requestedProvider === "serpapi" && isSerpApiConfigured() ? "serpapi" : "mock",
    requestedProvider,
    serpApiConfigured: isSerpApiConfigured()
  };
}

module.exports = {
  providerHealth,
  searchFlights
};
