const DAY_MS = 24 * 60 * 60 * 1000;

function parseDate(dateString) {
  return new Date(`${dateString}T00:00:00.000Z`);
}

function formatDate(date) {
  return date.toISOString().slice(0, 10);
}

function addDays(date, days) {
  return new Date(date.getTime() + days * DAY_MS);
}

function inferWindowFromChildren(children) {
  const validChildren = Array.isArray(children)
    ? children.filter((child) => child.termEnds && child.termStarts)
    : [];

  if (validChildren.length === 0) {
    return {
      start: parseDate("2026-08-01"),
      end: parseDate("2026-08-31")
    };
  }

  const start = new Date(Math.max(...validChildren.map((child) => addDays(parseDate(child.termEnds), 1).getTime())));
  const end = new Date(Math.min(...validChildren.map((child) => addDays(parseDate(child.termStarts), -1).getTime())));
  return { start, end };
}

function dateWithinWindow(start, end, ratio) {
  const spanDays = Math.max(1, Math.floor((end.getTime() - start.getTime()) / DAY_MS));
  return addDays(start, Math.max(0, Math.min(spanDays, Math.round(spanDays * ratio))));
}

async function searchFlights(request) {
  const { start, end } = inferWindowFromChildren(request.children);
  const minTripDays = Math.max(1, Number(request.minTripDays || 14));
  const maxTripDays = Math.max(minTripDays, minTripDays + Number(request.tripFlexDays || 7));
  const tripDays = Math.round((minTripDays + maxTripDays) / 2);
  const routeSeed = `${request.origin || "LON"}-${request.destination || "SHA"}`.length;
  const routeAdjustment = (routeSeed % 7) * 25;

  const outboundEarly = dateWithinWindow(start, end, 0.18);
  const outboundMid = dateWithinWindow(start, end, 0.28);
  const outboundLate = dateWithinWindow(start, end, 0.42);
  const inboundMid = addDays(outboundEarly, tripDays);
  const inboundLate = addDays(outboundMid, Math.min(maxTripDays, tripDays + 2));
  const inboundVeryLate = addDays(outboundMid, maxTripDays);

  return [
    {
      id: "mock-direct-best",
      price: 3820 + routeAdjustment,
      currency: "GBP",
      outboundDate: formatDate(outboundEarly),
      inboundDate: formatDate(inboundMid),
      stops: 0,
      totalDurationMinutes: 690,
      directBaselineDurationMinutes: 690,
      connectionAirport: null,
      layoverDurationMinutes: 0
      ,
      outboundSegments: [
        {
          from: request.origin || "LON",
          to: request.destination || "CAN",
          departTime: `${formatDate(outboundEarly)} 12:00`,
          arriveTime: `${formatDate(outboundEarly)} 23:30`,
          airline: "Mock Air",
          flightNumber: "MO 101"
        }
      ],
      returnSegments: [
        {
          from: request.destination || "CAN",
          to: request.origin || "LON",
          departTime: `${formatDate(inboundMid)} 01:30`,
          arriveTime: `${formatDate(inboundMid)} 13:00`,
          airline: "Mock Air",
          flightNumber: "MO 102"
        }
      ]
    },
    {
      id: "mock-one-stop-cheaper",
      price: 3180 + routeAdjustment,
      currency: "GBP",
      outboundDate: formatDate(outboundMid),
      inboundDate: formatDate(inboundLate),
      stops: 1,
      totalDurationMinutes: 875,
      directBaselineDurationMinutes: 690,
      connectionAirport: "DOH",
      layoverDurationMinutes: 95
      ,
      outboundSegments: [
        {
          from: request.origin || "LON",
          to: "DOH",
          departTime: `${formatDate(outboundMid)} 09:00`,
          arriveTime: `${formatDate(outboundMid)} 18:00`,
          airline: "Mock Connect",
          flightNumber: "MC 201"
        },
        {
          from: "DOH",
          to: request.destination || "CAN",
          departTime: `${formatDate(outboundMid)} 19:35`,
          arriveTime: `${formatDate(addDays(outboundMid, 1))} 08:20`,
          airline: "Mock Connect",
          flightNumber: "MC 202"
        }
      ],
      returnSegments: []
    },
    {
      id: "mock-long-layover-cheapest",
      price: 2950 + routeAdjustment,
      currency: "GBP",
      outboundDate: formatDate(outboundMid),
      inboundDate: formatDate(inboundVeryLate),
      stops: 1,
      totalDurationMinutes: 1160,
      directBaselineDurationMinutes: 690,
      connectionAirport: "IST",
      layoverDurationMinutes: 280
    },
    {
      id: "mock-fast-direct",
      price: 4050 + routeAdjustment,
      currency: "GBP",
      outboundDate: formatDate(outboundLate),
      inboundDate: formatDate(inboundLate),
      stops: 0,
      totalDurationMinutes: 675,
      directBaselineDurationMinutes: 675,
      connectionAirport: null,
      layoverDurationMinutes: 0
    },
    {
      id: "mock-before-window",
      price: 2600 + routeAdjustment,
      currency: "GBP",
      outboundDate: formatDate(addDays(start, -4)),
      inboundDate: formatDate(inboundMid),
      stops: 1,
      totalDurationMinutes: 840,
      directBaselineDurationMinutes: 690,
      connectionAirport: "HEL",
      layoverDurationMinutes: 80
    },
    {
      id: "mock-after-window",
      price: 2750 + routeAdjustment,
      currency: "GBP",
      outboundDate: formatDate(outboundLate),
      inboundDate: formatDate(addDays(end, 4)),
      stops: 1,
      totalDurationMinutes: 910,
      directBaselineDurationMinutes: 690,
      connectionAirport: "AMS",
      layoverDurationMinutes: 130
    }
  ];
}

module.exports = {
  searchFlights
};
