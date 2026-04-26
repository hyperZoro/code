const test = require("node:test");
const assert = require("node:assert/strict");
const { AIRPORTS, ROUTE_PRESETS, airportPayload } = require("../src/airports");

test("airport helper includes frequent family and ski airports", () => {
  const codes = new Set(AIRPORTS.map((airport) => airport.code));

  assert.ok(codes.has("CAN"));
  assert.ok(codes.has("LON"));
  assert.ok(codes.has("GVA"));
  assert.ok(codes.has("ZRH"));
  assert.ok(codes.has("INN"));
});

test("route presets point at known airport codes", () => {
  const codes = new Set(AIRPORTS.map((airport) => airport.code));

  for (const preset of ROUTE_PRESETS) {
    assert.ok(codes.has(preset.origin), `${preset.id} has an unknown origin`);
    assert.ok(codes.has(preset.destination), `${preset.id} has an unknown destination`);
  }
});

test("airport payload exposes airports and route presets", () => {
  const payload = airportPayload();

  assert.equal(payload.airports.length, AIRPORTS.length);
  assert.equal(payload.routePresets.length, ROUTE_PRESETS.length);
});
