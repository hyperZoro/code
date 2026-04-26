const test = require("node:test");
const assert = require("node:assert/strict");
const { buildHolidayWindows, parseFreemensTermDates } = require("../src/schools/freemens");

const sampleHtml = `
  <h5>Autumn Term 2025</h5>
  <p>Wednesday 3 September 2025 Term starts</p>
  <p>Friday 17 October 2025 Half term begins at 4pm</p>
  <p>Monday 3 November 2025 Term recommences</p>
  <p>Friday 12 December 2025 Term ends at 3.15pm Junior School</p>
  <p>Term ends at 4pm Senior School</p>
  <h5>Spring Term 2026</h5>
  <p>Tuesday 6 January 2026 Term starts</p>
  <p>Friday 13 February 2026 Half term begins at 4pm</p>
  <p>Monday 23 February 2026 Term recommences</p>
  <p>Friday 27 March 2026 Term ends at 3.15pm for Junior School</p>
  <p>Term ends at 4pm for Senior School</p>
  <h5>Summer Term 2026</h5>
  <p>Thursday 16 April 2026 Term starts</p>
  <p>Friday 22 May 2026 Half term begins at 4pm</p>
  <p>Monday 1 June 2026 Term recommences</p>
  <p>Wednesday 8 July 2026 Term ends at 12pm</p>
  <h5>Autumn Term 2026</h5>
  <p>Thursday 3 September 2026 Term starts</p>
  <p>Friday 16 October 2026 Half term begins at 4pm</p>
  <p>Monday 2 November 2026 Term recommences</p>
  <p>Tuesday 15 December 2026 Term ends at 3.15pm Junior School</p>
  <p>Term ends at 4pm Senior School</p>
`;

test("parses Freemen's term starts and junior/senior end dates", () => {
  const parsed = parseFreemensTermDates(sampleHtml);

  assert.equal(parsed.source, "deterministic");
  assert.equal(parsed.terms.length, 4);
  assert.deepEqual(parsed.terms[0], {
    academicYear: "2025-2026",
    name: "Autumn Term 2025",
    starts: "2025-09-03",
    ends: {
      junior: "2025-12-12",
      senior: "2025-12-12"
    },
    halfTerm: {
      begins: "2025-10-17",
      recommences: "2025-11-03"
    },
    source: "deterministic"
  });
  assert.equal(parsed.terms[2].ends.junior, "2026-07-08");
  assert.equal(parsed.terms[2].ends.senior, "2026-07-08");
  assert.deepEqual(parsed.terms[2].halfTerm, {
    begins: "2026-05-22",
    recommences: "2026-06-01"
  });
});

test("builds half-term and between-term holiday windows", () => {
  const parsed = parseFreemensTermDates(sampleHtml);
  const windows = buildHolidayWindows(parsed.terms);

  assert.equal(windows.length, 6);
  assert.equal(windows[0].label, "Autumn 2025 half term");
  assert.deepEqual(windows[0].junior, {
    termEnds: "2025-10-17",
    termStarts: "2025-11-03"
  });

  const summer2026 = windows.find((window) => window.label === "Summer holiday 2026");
  assert.equal(summer2026.type, "term-holiday");
  assert.deepEqual(summer2026.junior, {
    termEnds: "2026-07-08",
    termStarts: "2026-09-03"
  });
  assert.deepEqual(summer2026.senior, {
    termEnds: "2026-07-08",
    termStarts: "2026-09-03"
  });
});
