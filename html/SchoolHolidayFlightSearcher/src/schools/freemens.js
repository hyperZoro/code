const { extractFreemensTermDatesWithGemini } = require("../providers/geminiExtractor");

const FREEMENS_TERM_DATES_URL = "https://freemens.org/parent-hub/term-dates/";
const MONTHS = {
  january: 0,
  february: 1,
  march: 2,
  april: 3,
  may: 4,
  june: 5,
  july: 6,
  august: 7,
  september: 8,
  october: 9,
  november: 10,
  december: 11
};

function toIsoDate(day, monthName, year) {
  const date = new Date(Date.UTC(Number(year), MONTHS[monthName.toLowerCase()], Number(day)));
  return date.toISOString().slice(0, 10);
}

function stripHtml(html) {
  return String(html)
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|li|h[1-6])>/gi, "\n")
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&#8217;|&rsquo;/g, "'")
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s+/g, "\n")
    .trim();
}

function academicYearFromTerm(termName) {
  const year = Number(termName.match(/\b(20\d{2})\b/)?.[1]);
  if (!year) {
    return "";
  }
  return termName.includes("Autumn") ? `${year}-${year + 1}` : `${year - 1}-${year}`;
}

function normalizeTerms(terms, source = "deterministic") {
  return terms
    .filter((term) => term.starts && (term.ends.junior || term.ends.senior))
    .map((term) => ({
      academicYear: term.academicYear || academicYearFromTerm(term.name),
      name: term.name,
      starts: term.starts,
      ends: {
        junior: term.ends.junior || term.ends.senior,
        senior: term.ends.senior || term.ends.junior
      },
      halfTerm: term.halfTerm
        ? {
            begins: term.halfTerm.begins || null,
            recommences: term.halfTerm.recommences || null
          }
        : null,
      source
    }))
    .sort((a, b) => a.starts.localeCompare(b.starts));
}

function parseFreemensTermDates(html) {
  const text = stripHtml(html);
  const headingMatches = [...text.matchAll(/\b(Autumn|Spring|Summer) Term (20\d{2})\b/gi)];
  const terms = [];

  for (let index = 0; index < headingMatches.length; index += 1) {
    const heading = headingMatches[index];
    const name = heading[0];
    const startIndex = heading.index + heading[0].length;
    const endIndex = headingMatches[index + 1]?.index ?? text.length;
    const section = text.slice(startIndex, endIndex);
    const dateMatches = [...section.matchAll(/\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})\b/gi)].map(
      (match) => ({
        index: match.index,
        date: toIsoDate(match[1], match[2], match[3])
      })
    );

    const dateBefore = (keywordIndex) => {
      const priorDates = dateMatches.filter((match) => match.index < keywordIndex);
      return priorDates.at(-1)?.date || null;
    };

    const startsIndex = section.search(/\bTerm starts\b/i);
    const halfTermBeginsIndex = section.search(/\bHalf term begins\b/i);
    const recommencesIndex = section.search(/\bTerm recommences\b/i);
    const endMatches = [...section.matchAll(/\bTerm ends\b/gi)];
    const juniorEndIndex = section.search(/\bTerm ends[\s\S]{0,80}\bJunior School\b/i);
    const seniorEndIndex = section.search(/\bTerm ends[\s\S]{0,80}\bSenior School\b/i);
    const genericEndIndex = endMatches.at(-1)?.index ?? -1;
    const genericEndDate = genericEndIndex >= 0 ? dateBefore(genericEndIndex) : null;

    terms.push({
      academicYear: academicYearFromTerm(name),
      name,
      starts: startsIndex >= 0 ? dateBefore(startsIndex) : null,
      ends: {
        junior: juniorEndIndex >= 0 ? dateBefore(juniorEndIndex) : genericEndDate,
        senior: seniorEndIndex >= 0 ? dateBefore(seniorEndIndex) : genericEndDate
      },
      halfTerm: {
        begins: halfTermBeginsIndex >= 0 ? dateBefore(halfTermBeginsIndex) : null,
        recommences: recommencesIndex >= 0 ? dateBefore(recommencesIndex) : null
      }
    });
  }

  return {
    source: "deterministic",
    pageText: text,
    terms: normalizeTerms(terms, "deterministic")
  };
}

function holidayName(termName) {
  if (termName.includes("Autumn")) {
    return termName.replace("Autumn Term", "Christmas holiday");
  }
  if (termName.includes("Spring")) {
    return termName.replace("Spring Term", "Easter holiday");
  }
  return termName.replace("Summer Term", "Summer holiday");
}

function buildHolidayWindows(terms) {
  const windows = [];

  for (let index = 0; index < terms.length - 1; index += 1) {
    const term = terms[index];
    const nextTerm = terms[index + 1];

    if (term.halfTerm?.begins && term.halfTerm?.recommences) {
      windows.push({
        id: `${term.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-half-term`,
        type: "half-term",
        label: `${term.name.replace(" Term", "")} half term`,
        fromTerm: term.name,
        toTerm: term.name,
        junior: {
          termEnds: term.halfTerm.begins,
          termStarts: term.halfTerm.recommences
        },
        senior: {
          termEnds: term.halfTerm.begins,
          termStarts: term.halfTerm.recommences
        }
      });
    }

    windows.push({
      id: `${term.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${nextTerm.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
      type: "term-holiday",
      label: holidayName(term.name),
      fromTerm: term.name,
      toTerm: nextTerm.name,
      junior: {
        termEnds: term.ends.junior,
        termStarts: nextTerm.starts
      },
      senior: {
        termEnds: term.ends.senior,
        termStarts: nextTerm.starts
      }
    });
  }

  return windows.sort((a, b) => a.junior.termEnds.localeCompare(b.junior.termEnds));
}

async function fetchFreemensTermDatesLive() {
  const response = await fetch(FREEMENS_TERM_DATES_URL, {
    headers: {
      "user-agent": "SchoolHolidayFlightSearcher/0.1 personal-use"
    }
  });

  if (!response.ok) {
    throw Object.assign(new Error(`Freemen's term page returned ${response.status}.`), { statusCode: 502 });
  }

  const html = await response.text();
  const parsed = parseFreemensTermDates(html);
  let terms = parsed.terms;
  let source = parsed.source;

  if (terms.length < 3) {
    const geminiResult = await extractFreemensTermDatesWithGemini(parsed.pageText);
    if (geminiResult?.terms?.length) {
      terms = normalizeTerms(geminiResult.terms, "gemini");
      source = "gemini";
    }
  }

  if (terms.length < 2) {
    throw Object.assign(new Error("Could not extract enough Freemen's term dates."), { statusCode: 502 });
  }

  return {
    schoolName: "City of London Freemen's School",
    url: FREEMENS_TERM_DATES_URL,
    juniorUntilAge: 11,
    seniorFromAge: 12,
    source,
    geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
    terms,
    holidayWindows: buildHolidayWindows(terms)
  };
}

module.exports = {
  FREEMENS_TERM_DATES_URL,
  buildHolidayWindows,
  fetchFreemensTermDatesLive,
  parseFreemensTermDates,
  stripHtml
};
