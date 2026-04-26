const test = require("node:test");
const assert = require("node:assert/strict");
const {
  expandAirportCode,
  generateDatePairs,
  normalizeSerpApiPayload,
  oneWayOptionsFromPayload
} = require("../src/providers/serpApiFlightProvider");

test("expands broad city airport codes for Google Flights searches", () => {
  assert.equal(expandAirportCode("LON"), "LHR,LGW,STN,LTN,LCY");
  assert.equal(expandAirportCode("sha"), "PVG,SHA");
  assert.equal(expandAirportCode("CAN"), "CAN");
});

test("generates a small set of date pairs for a short half-term window", () => {
  const pairs = generateDatePairs({
    hasOverlap: true,
    startDate: "2026-05-23",
    endDate: "2026-05-31",
    minTripDays: 7,
    maxTripDays: 9
  });

  assert.ok(pairs.length > 0);
  assert.ok(pairs.length <= 4);
  assert.equal(pairs[0].outboundDate, "2026-05-23");
  assert.equal(pairs[0].inboundDate, "2026-05-30");
});

test("generates an exact date pair when override window equals trip length", () => {
  const pairs = generateDatePairs({
    hasOverlap: true,
    startDate: "2026-12-12",
    endDate: "2026-12-19",
    minTripDays: 7,
    maxTripDays: 7
  });

  assert.deepEqual(pairs, [{ outboundDate: "2026-12-12", inboundDate: "2026-12-19" }]);
});

test("normalizes SerpApi flight results into itinerary summaries", () => {
  const results = normalizeSerpApiPayload(
    {
      best_flights: [
        {
          flights: [
            {
              departure_airport: { id: "LON", time: "2026-07-10 10:00" },
              arrival_airport: { id: "DOH", time: "2026-07-10 19:00" },
              airline: "Qatar Airways",
              duration: 420
            },
            {
              departure_airport: { id: "DOH", time: "2026-07-10 21:00" },
              arrival_airport: { id: "CAN", time: "2026-07-11 11:00" },
              airline: "Qatar Airways",
              duration: 480
            }
          ],
          layovers: [{ id: "DOH", name: "Hamad International Airport", duration: 120 }],
          total_duration: 1020,
          price: 3120,
          type: "Round trip"
        }
      ],
      other_flights: []
    },
    { outboundDate: "2026-07-10", inboundDate: "2026-08-20" }
  );

  assert.equal(results.length, 1);
  assert.equal(results[0].price, 3120);
  assert.equal(results[0].outboundDate, "2026-07-10");
  assert.equal(results[0].inboundDate, "2026-08-20");
  assert.equal(results[0].stops, 1);
  assert.equal(results[0].connectionAirport, "DOH");
  assert.equal(results[0].layoverDurationMinutes, 120);
  assert.equal(results[0].airlineSummary, "Qatar Airways");
});

test("sorts one-way return options by price", () => {
  const options = oneWayOptionsFromPayload({
    best_flights: [
      { price: 900, flights: [{ airline: "A" }] },
      { price: 700, flights: [{ airline: "B" }] }
    ],
    other_flights: [{ price: 800, flights: [{ airline: "C" }] }]
  });

  assert.equal(options[0].price, 700);
  assert.equal(options[1].price, 800);
  assert.equal(options[2].price, 900);
});
