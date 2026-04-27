const fs = require("node:fs/promises");
const path = require("node:path");
const { fetchFreemensTermDatesLive } = require("./freemens");

const DAY_MS = 24 * 60 * 60 * 1000;
const DATA_DIR = process.env.DATA_DIR || path.join(process.cwd(), "data");
const FREEMENS_CACHE_PATH = path.join(DATA_DIR, "freemens-term-dates.json");
const DEFAULT_REFRESH_THRESHOLD_DAYS = 183;

function refreshThresholdDays() {
  const configured = Number(process.env.TERM_CACHE_REFRESH_DAYS || DEFAULT_REFRESH_THRESHOLD_DAYS);
  return Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_REFRESH_THRESHOLD_DAYS;
}

function getFurthestBreakEndDate(catalog) {
  const dates = (catalog?.holidayWindows || [])
    .flatMap((window) => [window.junior?.termStarts, window.senior?.termStarts])
    .filter(Boolean)
    .sort();

  return dates.at(-1) || null;
}

function shouldRefreshCachedTermDates(catalog, now = new Date(), thresholdDays = refreshThresholdDays()) {
  const furthestDate = getFurthestBreakEndDate(catalog);
  if (!furthestDate) {
    return {
      shouldRefresh: true,
      furthestBreakEndDate: null,
      daysUntilFurthestBreakEnd: null,
      reason: "missing-cache"
    };
  }

  const furthestTime = new Date(`${furthestDate}T00:00:00Z`).getTime();
  const nowTime = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const daysUntilFurthestBreakEnd = Math.ceil((furthestTime - nowTime) / DAY_MS);

  return {
    shouldRefresh: daysUntilFurthestBreakEnd <= thresholdDays,
    furthestBreakEndDate: furthestDate,
    daysUntilFurthestBreakEnd,
    reason: daysUntilFurthestBreakEnd <= thresholdDays ? "furthest-date-within-threshold" : "cache-current"
  };
}

async function readCachedTermDates() {
  try {
    const text = await fs.readFile(FREEMENS_CACHE_PATH, "utf8");
    return JSON.parse(text);
  } catch (error) {
    if (error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

async function writeCachedTermDates(catalog) {
  await fs.mkdir(DATA_DIR, { recursive: true });
  const payload = {
    ...catalog,
    cache: {
      ...(catalog.cache || {}),
      storedAt: new Date().toISOString()
    }
  };
  await fs.writeFile(FREEMENS_CACHE_PATH, `${JSON.stringify(payload, null, 2)}\n`);
  return payload;
}

async function getFreemensTermDates({ forceRefresh = false } = {}) {
  const cached = await readCachedTermDates();
  const refreshState = shouldRefreshCachedTermDates(cached);

  if (!forceRefresh && cached && !refreshState.shouldRefresh) {
    return {
      ...cached,
      cache: {
        ...(cached.cache || {}),
        servedFromCache: true,
        refreshed: false,
        refreshReason: refreshState.reason,
        furthestBreakEndDate: refreshState.furthestBreakEndDate,
        daysUntilFurthestBreakEnd: refreshState.daysUntilFurthestBreakEnd
      }
    };
  }

  try {
    const fresh = await fetchFreemensTermDatesLive();
    const stored = await writeCachedTermDates({
      ...fresh,
      cache: {
        servedFromCache: false,
        refreshed: true,
        refreshReason: forceRefresh ? "manual-refresh" : refreshState.reason,
        furthestBreakEndDate: getFurthestBreakEndDate(fresh)
      }
    });
    return stored;
  } catch (error) {
    if (cached) {
      return {
        ...cached,
        cache: {
          ...(cached.cache || {}),
          servedFromCache: true,
          refreshed: false,
          refreshFailed: true,
          refreshError: error.message,
          refreshReason: refreshState.reason,
          furthestBreakEndDate: refreshState.furthestBreakEndDate,
          daysUntilFurthestBreakEnd: refreshState.daysUntilFurthestBreakEnd
        }
      };
    }
    throw error;
  }
}

module.exports = {
  DEFAULT_REFRESH_THRESHOLD_DAYS,
  FREEMENS_CACHE_PATH,
  getFreemensTermDates,
  getFurthestBreakEndDate,
  refreshThresholdDays,
  readCachedTermDates,
  shouldRefreshCachedTermDates,
  writeCachedTermDates
};
