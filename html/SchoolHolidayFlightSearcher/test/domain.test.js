const test = require("node:test");
const assert = require("node:assert/strict");
const {
  buildSearchResponse,
  computeTravelWindow,
  isItineraryInsideWindow,
  matchesPreferredAirlines,
  scoreEligibleItineraries
} = require("../src/domain");

const children = [
  {
    id: "child-1",
    schoolName: "North School",
    termEnds: "2026-07-17",
    termStarts: "2026-09-03"
  },
  {
    id: "child-2",
    schoolName: "South School",
    termEnds: "2026-07-20",
    termStarts: "2026-09-01"
  }
];

test("computes the intersection of child holiday windows without flexibility", () => {
  const window = computeTravelWindow({ children, leaveEarlyDays: 0, returnLateDays: 0 });

  assert.equal(window.hasOverlap, true);
  assert.equal(window.startDate, "2026-07-21");
  assert.equal(window.endDate, "2026-08-31");
});

test("expands outbound and inbound bounds with flexibility", () => {
  const window = computeTravelWindow({ children, leaveEarlyDays: 3, returnLateDays: 2 });

  assert.equal(window.hasOverlap, true);
  assert.equal(window.startDate, "2026-07-17");
  assert.equal(window.endDate, "2026-09-03");
});

test("reports non-overlapping school windows clearly", () => {
  const window = computeTravelWindow({
    children: [
      { id: "a", termEnds: "2026-07-17", termStarts: "2026-08-01" },
      { id: "b", termEnds: "2026-08-10", termStarts: "2026-09-01" }
    ],
    leaveEarlyDays: 0,
    returnLateDays: 0
  });

  assert.equal(window.hasOverlap, false);
  assert.equal(window.startDate, null);
  assert.equal(window.endDate, null);
});

test("uses override search window instead of school windows", () => {
  const window = computeTravelWindow({
    children,
    leaveEarlyDays: 0,
    returnLateDays: 0,
    overrideWindow: {
      enabled: true,
      startDate: "2026-10-01",
      endDate: "2026-10-09"
    }
  });

  assert.equal(window.source, "override");
  assert.equal(window.hasOverlap, true);
  assert.equal(window.startDate, "2026-10-01");
  assert.equal(window.endDate, "2026-10-09");
  assert.deepEqual(window.childWindows, []);
});

test("filters itineraries outside the shared travel window", () => {
  const window = computeTravelWindow({ children, leaveEarlyDays: 0, returnLateDays: 0 });

  assert.equal(
    isItineraryInsideWindow({ outboundDate: "2026-08-01", inboundDate: "2026-08-20" }, window),
    true
  );
  assert.equal(
    isItineraryInsideWindow({ outboundDate: "2026-07-19", inboundDate: "2026-08-20" }, window),
    false
  );
  assert.equal(
    isItineraryInsideWindow({ outboundDate: "2026-08-01", inboundDate: "2026-09-02" }, window),
    false
  );
});

test("scores cheaper one-stop trips against faster direct trips with clear flags", () => {
  const scored = scoreEligibleItineraries([
    {
      id: "direct",
      price: 4000,
      stops: 0,
      totalDurationMinutes: 690,
      directBaselineDurationMinutes: 690,
      layoverDurationMinutes: 0
    },
    {
      id: "one-stop",
      price: 3100,
      stops: 1,
      totalDurationMinutes: 850,
      directBaselineDurationMinutes: 690,
      layoverDurationMinutes: 95,
      connectionAirport: "DOH"
    },
    {
      id: "long-layover",
      price: 3000,
      stops: 1,
      totalDurationMinutes: 1150,
      directBaselineDurationMinutes: 690,
      layoverDurationMinutes: 300,
      connectionAirport: "IST"
    }
  ]);

  assert.equal(scored[0].id, "long-layover");
  assert.ok(scored[0].reasonFlags.includes("recommended"));
  assert.ok(scored.find((item) => item.id === "one-stop").reasonFlags.includes("short-layover"));
  assert.ok(scored.find((item) => item.id === "long-layover").reasonFlags.includes("connection-risk"));
});

test("builds a complete response from provider results", () => {
  const response = buildSearchResponse(
    {
      origin: "lon",
      destination: "sha",
      adults: 2,
      childrenAges: [9, 11],
      children,
      leaveEarlyDays: 0,
      returnLateDays: 0
    },
    [
      {
        id: "inside",
        price: 3500,
        currency: "GBP",
        outboundDate: "2026-08-01",
        inboundDate: "2026-08-20",
        stops: 0,
        totalDurationMinutes: 690,
        directBaselineDurationMinutes: 690,
        layoverDurationMinutes: 0
      },
      {
        id: "outside",
        price: 2500,
        currency: "GBP",
        outboundDate: "2026-07-01",
        inboundDate: "2026-08-20",
        stops: 0,
        totalDurationMinutes: 690,
        directBaselineDurationMinutes: 690,
        layoverDurationMinutes: 0
      }
    ]
  );

  assert.equal(response.resultCount, 1);
  assert.equal(response.itineraries[0].id, "inside");
  assert.equal(response.request.origin, "LON");
});

test("filters itineraries to preferred airlines when requested", () => {
  const response = buildSearchResponse(
    {
      origin: "LHR",
      destination: "CAN",
      adults: 2,
      childrenAges: [9, 11],
      children,
      preferredAirlines: ["British Airways"],
      leaveEarlyDays: 0,
      returnLateDays: 0
    },
    [
      {
        id: "ba",
        price: 3800,
        currency: "GBP",
        outboundDate: "2026-08-01",
        inboundDate: "2026-08-20",
        airlineSummary: "British Airways",
        outboundSegments: [{ airline: "British Airways" }],
        returnSegments: [{ airline: "British Airways" }],
        stops: 0,
        totalDurationMinutes: 690,
        directBaselineDurationMinutes: 690,
        layoverDurationMinutes: 0
      },
      {
        id: "swiss",
        price: 3200,
        currency: "GBP",
        outboundDate: "2026-08-01",
        inboundDate: "2026-08-20",
        airlineSummary: "SWISS",
        outboundSegments: [{ airline: "SWISS" }],
        returnSegments: [{ airline: "SWISS" }],
        stops: 1,
        totalDurationMinutes: 820,
        directBaselineDurationMinutes: 690,
        layoverDurationMinutes: 100
      }
    ]
  );

  assert.equal(response.resultCount, 1);
  assert.equal(response.itineraries[0].id, "ba");
  assert.equal(response.searchMeta.filteredByAirlines, 1);
});

test("requires all known sector airlines to match preferences", () => {
  assert.equal(
    matchesPreferredAirlines(
      {
        airlineSummary: "British Airways, SWISS",
        outboundSegments: [{ airline: "British Airways" }],
        returnSegments: [{ airline: "SWISS" }]
      },
      ["British Airways"]
    ),
    false
  );
});
