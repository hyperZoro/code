const test = require("node:test");
const assert = require("node:assert/strict");
const { airlinePayload } = require("../src/airlines");

test("provides preferred airline groups for family searches", () => {
  const payload = airlinePayload();
  const groups = payload.groups.map((group) => group.id);

  assert.ok(groups.includes("chinese"));
  assert.ok(groups.includes("ba"));
  assert.ok(groups.includes("european-long-haul"));
  assert.ok(groups.includes("european-holiday"));
  assert.ok(payload.airlines.includes("China Southern"));
  assert.ok(payload.airlines.includes("British Airways"));
  assert.ok(payload.airlines.includes("Lufthansa"));
});
