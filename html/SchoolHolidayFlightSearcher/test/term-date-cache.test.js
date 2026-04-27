const test = require("node:test");
const assert = require("node:assert/strict");
const {
  DEFAULT_REFRESH_THRESHOLD_DAYS,
  getFurthestBreakEndDate,
  shouldRefreshCachedTermDates
} = require("../src/schools/termDateCache");

const catalog = {
  holidayWindows: [
    {
      junior: { termStarts: "2026-09-03" },
      senior: { termStarts: "2026-09-03" }
    },
    {
      junior: { termStarts: "2027-06-07" },
      senior: { termStarts: "2027-06-07" }
    }
  ]
};

test("finds the furthest stored break end date", () => {
  assert.equal(getFurthestBreakEndDate(catalog), "2027-06-07");
});

test("uses a half-year default refresh threshold", () => {
  assert.equal(DEFAULT_REFRESH_THRESHOLD_DAYS, 183);
});

test("refreshes when furthest stored break is within half a year", () => {
  const state = shouldRefreshCachedTermDates(catalog, new Date("2026-12-10T00:00:00Z"), 183);

  assert.equal(state.shouldRefresh, true);
  assert.equal(state.reason, "furthest-date-within-threshold");
});

test("keeps cache when furthest stored break is more than half a year away", () => {
  const state = shouldRefreshCachedTermDates(catalog, new Date("2026-11-01T00:00:00Z"), 183);

  assert.equal(state.shouldRefresh, false);
  assert.equal(state.reason, "cache-current");
});
