const { addDays, computeTravelWindow, daysBetweenDates, formatDate, parseDate } = require("../domain");

const SERPAPI_ENDPOINT = "https://serpapi.com/search";
const MULTI_AIRPORT_CODES = {
  LON: "LHR,LGW,STN,LTN,LCY",
  NYC: "JFK,LGA,EWR",
  PAR: "CDG,ORY",
  MIL: "MXP,LIN,BGY",
  ROM: "FCO,CIA",
  SHA: "PVG,SHA",
  BJS: "PEK,PKX"
};

function isSerpApiConfigured() {
  return Boolean(process.env.SERPAPI_API_KEY);
}

function clampInteger(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.floor(number)));
}

function daysBetween(startDate, endDate) {
  return Math.floor((parseDate(endDate, "endDate").getTime() - parseDate(startDate, "startDate").getTime()) / 86400000);
}

function uniqueDatePairs(pairs) {
  const seen = new Set();
  return pairs.filter((pair) => {
    const key = `${pair.outboundDate}:${pair.inboundDate}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return pair.outboundDate < pair.inboundDate;
  });
}

function expandAirportCode(code) {
  const normalized = String(code || "").trim().toUpperCase();
  return MULTI_AIRPORT_CODES[normalized] || normalized;
}

function generateDatePairs(travelWindow) {
  if (!travelWindow.hasOverlap) {
    return [];
  }

  const start = parseDate(travelWindow.startDate, "travelWindow.startDate");
  const end = parseDate(travelWindow.endDate, "travelWindow.endDate");
  const spanDays = daysBetween(travelWindow.startDate, travelWindow.endDate);
  const maxPairs = clampInteger(process.env.SERPAPI_MAX_DATE_PAIRS, 4, 1, 12);
  const minTripDays = Math.max(1, clampInteger(travelWindow.minTripDays, 14, 1, 90));
  const maxTripDays = Math.max(minTripDays, clampInteger(travelWindow.maxTripDays, minTripDays + 7, minTripDays, 120));
  const tripLengths = [...new Set([minTripDays, Math.round((minTripDays + maxTripDays) / 2), maxTripDays])];
  const offsets = spanDays <= 10 ? [0, 1, 2] : spanDays <= 24 ? [0, 2, 4, 6] : [1, 3, 5, 7, 10, 14];

  const pairs = [];
  for (const offset of offsets) {
    for (const length of tripLengths) {
      const outbound = addDays(start, offset);
      const inbound = addDays(outbound, length);
      if (inbound <= end) {
        pairs.push({
          outboundDate: formatDate(outbound),
          inboundDate: formatDate(inbound)
        });
      }
    }
  }

  if (pairs.length === 0 && spanDays >= minTripDays) {
    pairs.push({
      outboundDate: formatDate(start),
      inboundDate: formatDate(addDays(start, minTripDays))
    });
  }

  return uniqueDatePairs(pairs).slice(0, maxPairs);
}

function searchParamsForPair(request, pair, extraParams = {}) {
  const params = new URLSearchParams({
    engine: "google_flights",
    type: "1",
    departure_id: expandAirportCode(request.origin),
    arrival_id: expandAirportCode(request.destination),
    outbound_date: pair.outboundDate,
    return_date: pair.inboundDate,
    currency: process.env.SERPAPI_CURRENCY || "GBP",
    gl: process.env.SERPAPI_GOOGLE_COUNTRY || "uk",
    hl: process.env.SERPAPI_LANGUAGE || "en",
    adults: String(Math.max(1, Number(request.adults || 1))),
    children: String(Array.isArray(request.childrenAges) ? request.childrenAges.length : 0),
    stops: process.env.SERPAPI_STOPS || "2",
    sort_by: process.env.SERPAPI_SORT_BY || "2",
    api_key: process.env.SERPAPI_API_KEY
  });

  for (const [key, value] of Object.entries(extraParams)) {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, value);
    }
  }

  if (process.env.SERPAPI_DEEP_SEARCH === "true") {
    params.set("deep_search", "true");
  }

  if (process.env.SERPAPI_MAX_DURATION_MINUTES) {
    params.set("max_duration", process.env.SERPAPI_MAX_DURATION_MINUTES);
  }

  if (process.env.SERPAPI_LAYOVER_DURATION) {
    params.set("layover_duration", process.env.SERPAPI_LAYOVER_DURATION);
  }

  return params;
}

async function fetchSerpApiPair(request, pair) {
  const url = `${SERPAPI_ENDPOINT}?${searchParamsForPair(request, pair)}`;
  const response = await fetch(url);
  const payload = await response.json();

  if (!response.ok || payload.error) {
    throw new Error(payload.error || `SerpApi returned HTTP ${response.status}`);
  }

  return payload;
}

async function fetchSerpApiReturnOptions(request, pair, departureToken) {
  const params = new URLSearchParams({
    engine: "google_flights",
    type: "1",
    departure_token: departureToken,
    return_date: pair.inboundDate,
    currency: process.env.SERPAPI_CURRENCY || "GBP",
    gl: process.env.SERPAPI_GOOGLE_COUNTRY || "uk",
    hl: process.env.SERPAPI_LANGUAGE || "en",
    api_key: process.env.SERPAPI_API_KEY
  });
  const url = `${SERPAPI_ENDPOINT}?${params}`;
  const response = await fetch(url);
  const payload = await response.json();

  if (!response.ok || payload.error) {
    throw new Error(payload.error || `SerpApi return details returned HTTP ${response.status}`);
  }

  return payload;
}

async function fetchReverseReturnSearch(request, itinerary) {
  const params = new URLSearchParams({
    engine: "google_flights",
    type: "2",
    departure_id: expandAirportCode(request.destination),
    arrival_id: expandAirportCode(request.origin),
    outbound_date: itinerary.inboundDate,
    currency: process.env.SERPAPI_CURRENCY || "GBP",
    gl: process.env.SERPAPI_GOOGLE_COUNTRY || "uk",
    hl: process.env.SERPAPI_LANGUAGE || "en",
    adults: String(Math.max(1, Number(request.adults || 1))),
    children: String(Array.isArray(request.childrenAges) ? request.childrenAges.length : 0),
    stops: process.env.SERPAPI_STOPS || "2",
    sort_by: process.env.SERPAPI_SORT_BY || "2",
    api_key: process.env.SERPAPI_API_KEY
  });

  const response = await fetch(`${SERPAPI_ENDPOINT}?${params}`);
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `SerpApi reverse return search returned HTTP ${response.status}`);
  }
  return payload;
}

function firstLegDate(result, fallbackDate) {
  const time = result.flights?.[0]?.departure_airport?.time;
  return time ? time.slice(0, 10) : fallbackDate;
}

function airlineSummary(result) {
  const airlines = [...new Set((result.flights || []).map((flight) => flight.airline).filter(Boolean))];
  return airlines.slice(0, 3).join(", ");
}

function normalizeSegments(flights = []) {
  return flights.map((flight) => ({
    from: flight.departure_airport?.id || "",
    fromName: flight.departure_airport?.name || "",
    to: flight.arrival_airport?.id || "",
    toName: flight.arrival_airport?.name || "",
    departTime: flight.departure_airport?.time || "",
    arriveTime: flight.arrival_airport?.time || "",
    airline: flight.airline || "",
    flightNumber: flight.flight_number || "",
    durationMinutes: Number(flight.duration || 0)
  }));
}

function normalizeSerpApiResult(result, pair, index) {
  const stops = Math.max(0, (result.flights?.length || 1) - 1);
  const longestLayover = Math.max(0, ...(result.layovers || []).map((layover) => Number(layover.duration || 0)));
  const firstLayover = result.layovers?.[0];
  const totalDuration = Number(result.total_duration || 0);

  return {
    id: `serpapi-${pair.outboundDate}-${pair.inboundDate}-${index}`,
    price: Number(result.price),
    currency: process.env.SERPAPI_CURRENCY || "GBP",
    outboundDate: firstLegDate(result, pair.outboundDate),
    inboundDate: pair.inboundDate,
    isDirect: stops === 0,
    stops,
    tripDays: daysBetweenDates(pair.outboundDate, pair.inboundDate),
    totalDurationMinutes: totalDuration || 999,
    directBaselineDurationMinutes: Number(process.env.SERPAPI_DIRECT_BASELINE_MINUTES || totalDuration || 999),
    connectionAirport: firstLayover?.id || null,
    layoverDurationMinutes: longestLayover,
    airlineSummary: airlineSummary(result),
    outboundSegments: normalizeSegments(result.flights || []),
    returnSegments: [],
    returnDurationMinutes: null,
    source: "Google Flights via SerpApi",
    bookingToken: result.booking_token || null,
    departureToken: result.departure_token || null
  };
}

function returnOptionsFromPayload(payload) {
  return [
    ...(payload.best_flights || []),
    ...(payload.other_flights || []),
    ...(payload.return_flights || []),
    ...(payload.flights || [])
  ]
    .filter((result) => Array.isArray(result.flights) && result.flights.length > 0)
    .sort((a, b) => Number(a.price || Infinity) - Number(b.price || Infinity));
}

function oneWayOptionsFromPayload(payload) {
  return [...(payload.best_flights || []), ...(payload.other_flights || [])]
    .filter((result) => Array.isArray(result.flights) && result.flights.length > 0)
    .sort((a, b) => Number(a.price || Infinity) - Number(b.price || Infinity));
}

function shouldUseReverseReturnFallback() {
  return process.env.SERPAPI_RETURN_REVERSE_FALLBACK !== "false";
}

async function enrichReturnDetails(request, itineraries, partialErrors) {
  const limit = clampInteger(process.env.SERPAPI_RETURN_DETAILS_LIMIT, 3, 0, 10);
  const candidates = [...itineraries]
    .filter((itinerary) => itinerary.departureToken)
    .sort((a, b) => a.price - b.price)
    .slice(0, limit);

  for (const itinerary of candidates) {
    try {
      let returnOption = null;

      try {
        const payload = await fetchSerpApiReturnOptions(
          request,
          { outboundDate: itinerary.outboundDate, inboundDate: itinerary.inboundDate },
          itinerary.departureToken
        );
        returnOption = returnOptionsFromPayload(payload)[0] || null;
      } catch (error) {
        if (!shouldUseReverseReturnFallback()) {
          throw error;
        }
      }

      if (returnOption) {
        itinerary.returnSegments = normalizeSegments(returnOption.flights || []);
        itinerary.returnDurationMinutes = Number(returnOption.total_duration || 0) || null;
        if (Number.isFinite(Number(returnOption.price))) {
          itinerary.price = Number(returnOption.price);
        }
      } else {
        const payload = await fetchReverseReturnSearch(request, itinerary);
        const oneWayReturn = oneWayOptionsFromPayload(payload)[0];
        if (oneWayReturn) {
          itinerary.returnSegments = normalizeSegments(oneWayReturn.flights || []);
          itinerary.returnDurationMinutes = Number(oneWayReturn.total_duration || 0) || null;
        }
      }
    } catch (error) {
      partialErrors.push({
        outboundDate: itinerary.outboundDate,
        inboundDate: itinerary.inboundDate,
        message: `Return detail lookup failed: ${error.message}`
      });
    }
  }
}

function normalizeSerpApiPayload(payload, pair) {
  const results = [...(payload.best_flights || []), ...(payload.other_flights || [])]
    .filter((result) => Number.isFinite(Number(result.price)))
    .slice(0, clampInteger(process.env.SERPAPI_RESULTS_PER_PAIR, 5, 1, 10));

  return results.map((result, index) => normalizeSerpApiResult(result, pair, index));
}

async function searchFlights(request) {
  const travelWindow = computeTravelWindow(request);
  const minTripDays = Math.max(1, clampInteger(request.minTripDays, 14, 1, 90));
  const tripFlexDays = clampInteger(request.tripFlexDays, 7, 0, 120);
  travelWindow.minTripDays = minTripDays;
  travelWindow.maxTripDays = minTripDays + tripFlexDays;
  const datePairs = generateDatePairs(travelWindow);
  const partialErrors = [];
  const itineraries = [];

  for (const pair of datePairs) {
    try {
      const payload = await fetchSerpApiPair(request, pair);
      itineraries.push(...normalizeSerpApiPayload(payload, pair));
    } catch (error) {
      partialErrors.push({
        outboundDate: pair.outboundDate,
        inboundDate: pair.inboundDate,
        message: error.message
      });
    }
  }

  await enrichReturnDetails(request, itineraries, partialErrors);

  return {
    provider: "serpapi",
    requestedProvider: "serpapi",
    itineraries,
    searchMeta: {
      datePairs,
      datePairsSearched: datePairs.length,
      partialErrors
    }
  };
}

module.exports = {
  expandAirportCode,
  generateDatePairs,
  isSerpApiConfigured,
  normalizeSerpApiPayload,
  oneWayOptionsFromPayload,
  searchFlights
};
