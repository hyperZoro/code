const DAY_MS = 24 * 60 * 60 * 1000;

function parseDate(dateString, fieldName) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(dateString))) {
    throw Object.assign(new Error(`${fieldName} must be a YYYY-MM-DD date.`), { statusCode: 400 });
  }

  const date = new Date(`${dateString}T00:00:00.000Z`);
  if (Number.isNaN(date.getTime())) {
    throw Object.assign(new Error(`${fieldName} is not a valid date.`), { statusCode: 400 });
  }
  return date;
}

function formatDate(date) {
  return date.toISOString().slice(0, 10);
}

function addDays(date, days) {
  return new Date(date.getTime() + days * DAY_MS);
}

function clampNonNegativeInteger(value, fallback = 0) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    return fallback;
  }
  return Math.floor(number);
}

function daysBetweenDates(startDate, endDate) {
  return Math.round((parseDate(endDate, "endDate").getTime() - parseDate(startDate, "startDate").getTime()) / DAY_MS);
}

function computeChildWindow(child, leaveEarlyDays, returnLateDays) {
  const termEnds = parseDate(child.termEnds, `${child.schoolName || child.id || "child"} termEnds`);
  const termStarts = parseDate(child.termStarts, `${child.schoolName || child.id || "child"} termStarts`);

  if (termStarts <= termEnds) {
    throw Object.assign(new Error("Each child's termStarts must be after termEnds."), { statusCode: 400 });
  }

  const start = leaveEarlyDays > 0 ? addDays(termEnds, -leaveEarlyDays) : addDays(termEnds, 1);
  const end = returnLateDays > 0 ? addDays(termStarts, returnLateDays) : addDays(termStarts, -1);

  return {
    childId: child.id,
    schoolName: child.schoolName,
    startDate: formatDate(start),
    endDate: formatDate(end),
    termEnds: formatDate(termEnds),
    termStarts: formatDate(termStarts)
  };
}

function computeTravelWindow(request) {
  if (request.overrideWindow?.enabled) {
    const start = parseDate(request.overrideWindow.startDate, "overrideWindow.startDate");
    const end = parseDate(request.overrideWindow.endDate, "overrideWindow.endDate");

    if (end <= start) {
      throw Object.assign(new Error("Override window end date must be after start date."), { statusCode: 400 });
    }

    return {
      startDate: formatDate(start),
      endDate: formatDate(end),
      childWindows: [],
      hasOverlap: true,
      source: "override"
    };
  }

  const children = Array.isArray(request.children) ? request.children : [];
  if (children.length === 0) {
    throw Object.assign(new Error("At least one child with term dates is required."), { statusCode: 400 });
  }

  const leaveEarlyDays = clampNonNegativeInteger(request.leaveEarlyDays);
  const returnLateDays = clampNonNegativeInteger(request.returnLateDays);
  const childWindows = children.map((child) => computeChildWindow(child, leaveEarlyDays, returnLateDays));
  const startTime = Math.max(...childWindows.map((window) => parseDate(window.startDate, "window startDate").getTime()));
  const endTime = Math.min(...childWindows.map((window) => parseDate(window.endDate, "window endDate").getTime()));

  if (startTime > endTime) {
    return {
      startDate: null,
      endDate: null,
      childWindows,
      hasOverlap: false
    };
  }

  return {
    startDate: formatDate(new Date(startTime)),
    endDate: formatDate(new Date(endTime)),
    childWindows,
    hasOverlap: true,
    source: "school"
  };
}

function isItineraryInsideWindow(itinerary, travelWindow) {
  if (!travelWindow.hasOverlap) {
    return false;
  }

  const tripDays = daysBetweenDates(itinerary.outboundDate, itinerary.inboundDate);
  const minTripDays = clampNonNegativeInteger(travelWindow.minTripDays, 0);
  const maxTripDays = clampNonNegativeInteger(travelWindow.maxTripDays, 9999);

  return (
    itinerary.outboundDate >= travelWindow.startDate &&
    itinerary.inboundDate <= travelWindow.endDate &&
    itinerary.inboundDate >= itinerary.outboundDate &&
    tripDays >= minTripDays &&
    tripDays <= maxTripDays
  );
}

function scoreEligibleItineraries(itineraries) {
  if (itineraries.length === 0) {
    return [];
  }

  const cheapestPrice = Math.min(...itineraries.map((itinerary) => itinerary.price));
  const averageDuration =
    itineraries.reduce((sum, itinerary) => sum + itinerary.totalDurationMinutes, 0) / itineraries.length;

  const scored = itineraries.map((itinerary) => {
    const directBaseline = itinerary.directBaselineDurationMinutes || itinerary.totalDurationMinutes;
    const durationRatio = itinerary.totalDurationMinutes / directBaseline;
    const layover = itinerary.layoverDurationMinutes || 0;
    const priceScore = (itinerary.price / cheapestPrice) * 100;
    const stopPenalty = itinerary.stops > 0 ? 14 : 0;
    const durationPenalty = Math.max(0, durationRatio - 1) * 25;
    const layoverPenalty = layover > 180 ? 22 : layover > 120 ? 10 : itinerary.stops > 0 ? 4 : 0;
    const score = Math.round((priceScore + stopPenalty + durationPenalty + layoverPenalty) * 10) / 10;

    return {
      ...itinerary,
      isDirect: itinerary.stops === 0,
      score,
      reasonFlags: ["school-compatible"]
    };
  });

  scored.sort(
    (a, b) =>
      a.price - b.price ||
      a.totalDurationMinutes - b.totalDurationMinutes ||
      a.score - b.score
  );

  const cheapest = scored.reduce((best, itinerary) => (itinerary.price < best.price ? itinerary : best), scored[0]);
  return scored.map((itinerary, index) => {
    const flags = [...itinerary.reasonFlags];
    if (itinerary.id === cheapest.id) {
      flags.push("cheapest");
    }
    if (itinerary.totalDurationMinutes <= averageDuration) {
      flags.push(index === 0 && itinerary.isDirect ? "faster" : "efficient");
    }
    if (itinerary.stops > 0 && itinerary.layoverDurationMinutes <= 120) {
      flags.push("short-layover");
    }
    if (itinerary.layoverDurationMinutes > 180) {
      flags.push("connection-risk");
    }
    if (index === 0) {
      flags.unshift("recommended");
    }
    return { ...itinerary, reasonFlags: flags };
  });
}

function normalizeRequest(request) {
  const minTripDays = Math.max(1, clampNonNegativeInteger(request.minTripDays, 14));
  const tripFlexDays = clampNonNegativeInteger(request.tripFlexDays, 7);

  const overrideWindow = {
    enabled: Boolean(request.overrideWindow?.enabled),
    startDate: request.overrideWindow?.startDate || "",
    endDate: request.overrideWindow?.endDate || ""
  };

  return {
    origin: String(request.origin || "").trim().toUpperCase(),
    destination: String(request.destination || "").trim().toUpperCase(),
    tripType: "return",
    adults: Math.max(1, clampNonNegativeInteger(request.adults, 2)),
    childrenAges: Array.isArray(request.childrenAges) ? request.childrenAges.map((age) => Number(age)) : [],
    children: Array.isArray(request.children) ? request.children : [],
    leaveEarlyDays: clampNonNegativeInteger(request.leaveEarlyDays),
    returnLateDays: clampNonNegativeInteger(request.returnLateDays),
    minTripDays,
    tripFlexDays,
    maxTripDays: minTripDays + tripFlexDays,
    overrideWindow
  };
}

function buildSearchResponse(request, providerResults, providerMeta = {}) {
  const normalizedRequest = normalizeRequest(request);
  if (!normalizedRequest.origin || !normalizedRequest.destination) {
    throw Object.assign(new Error("Origin and destination are required."), { statusCode: 400 });
  }

  const travelWindow = computeTravelWindow(normalizedRequest);
  travelWindow.minTripDays = normalizedRequest.minTripDays;
  travelWindow.maxTripDays = normalizedRequest.maxTripDays;
  const eligible = providerResults.filter((itinerary) => isItineraryInsideWindow(itinerary, travelWindow));
  const itineraries = scoreEligibleItineraries(eligible);

  return {
    request: normalizedRequest,
    travelWindow,
    provider: providerMeta.provider || "mock",
    requestedProvider: providerMeta.requestedProvider || providerMeta.provider || "mock",
    providerWarning: providerMeta.warning || null,
    searchMeta: providerMeta.searchMeta || null,
    resultCount: itineraries.length,
    itineraries
  };
}

module.exports = {
  addDays,
  buildSearchResponse,
  computeTravelWindow,
  daysBetweenDates,
  formatDate,
  isItineraryInsideWindow,
  parseDate,
  scoreEligibleItineraries
};
