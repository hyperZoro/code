const childrenRoot = document.querySelector("#children");
const form = document.querySelector("#search-form");
const resultsRoot = document.querySelector("#results");
const windowCard = document.querySelector("#window-card");
const resultCount = document.querySelector("#result-count");
const serviceStatus = document.querySelector("#service-status");
const statusPanel = document.querySelector(".status-panel");
const submitButton = form.querySelector("button[type='submit']");
const loadFreemensButton = document.querySelector("#load-freemens");
const refreshFreemensButton = document.querySelector("#refresh-freemens");
const applyFreemensButton = document.querySelector("#apply-freemens");
const freemensWindowSelect = document.querySelector("#freemens-window");
const schoolStatus = document.querySelector("#school-status");
const airportOptions = document.querySelector("#airport-options");
const routePresetsRoot = document.querySelector("#route-presets");
const airportSummary = document.querySelector("#airport-summary");
const londonAirportsRoot = document.querySelector("#london-airports");
const homeAirportsButton = document.querySelector("#home-airports");
const holidayAirportsButton = document.querySelector("#holiday-airports");
const airlinePreferencesRoot = document.querySelector("#airline-preferences");
const airlineSummary = document.querySelector("#airline-summary");
const airlineActionsRoot = document.querySelector(".airline-actions");
const overrideEnabledInput = document.querySelector("#override-enabled");
const overrideStartInput = document.querySelector("#override-start");
const overrideEndInput = document.querySelector("#override-end");
const appPath = window.location.pathname.endsWith("/")
  ? window.location.pathname
  : window.location.pathname.replace(/\/[^/]*$/, "/");
const apiBase = appPath === "/" ? "/api" : `${appPath.replace(/\/$/, "")}/api`;

const defaultChildren = [
  {
    id: "child-1",
    label: "Child 1",
    age: 9,
    schoolName: "City of London Freemen's School",
    schoolYear: "Year 5",
    schoolUrl: "https://freemens.org/parent-hub/term-dates/",
    termEnds: "2026-07-08",
    termStarts: "2026-09-03"
  },
  {
    id: "child-2",
    label: "Child 2",
    age: 11,
    schoolName: "City of London Freemen's School",
    schoolYear: "Year 7",
    schoolUrl: "https://freemens.org/parent-hub/term-dates/",
    termEnds: "2026-07-08",
    termStarts: "2026-09-03"
  }
];
let freemensCatalog = null;
let airportCatalog = { airports: [], routePresets: [] };
let airlineCatalog = { groups: [], airlines: [] };
const londonAirportCodes = ["LHR", "LGW", "STN", "LTN", "LCY"];

function formatDate(dateString) {
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric"
  }).format(new Date(`${dateString}T00:00:00Z`));
}

function formatMinutes(minutes) {
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return mins ? `${hours}h ${mins}m` : `${hours}h`;
}

function formatCurrency(amount, currency) {
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency,
    maximumFractionDigits: 0
  }).format(amount);
}

function schoolLevelForAge(age) {
  return Number(age) >= 12 ? "senior" : "junior";
}

function getInput(name) {
  return form.elements[name];
}

function codeFromInputValue(value) {
  const trimmed = String(value || "").trim().toUpperCase();
  const exact = airportCatalog.airports.find((airport) => airport.code === trimmed);
  if (exact) {
    return exact.code;
  }

  const match = trimmed.match(/\b([A-Z]{3})\b/);
  return match ? match[1] : trimmed;
}

function airportLabel(code) {
  if (code.includes(",")) {
    return code
      .split(",")
      .map((part) => airportLabel(part))
      .join(" + ");
  }
  const airport = airportCatalog.airports.find((item) => item.code === code);
  return airport ? `${airport.code} - ${airport.city} (${airport.name})` : code;
}

function selectedLondonAirports() {
  return [...londonAirportsRoot.querySelectorAll("input:checked")].map((input) => input.value);
}

function selectedPreferredAirlines() {
  return [...airlinePreferencesRoot.querySelectorAll("input:checked")].map((input) => input.value);
}

function syncOriginFromAirportChecks() {
  const selected = selectedLondonAirports();
  if (selected.length === londonAirportCodes.length) {
    getInput("origin").value = "LON";
  } else if (selected.length > 0) {
    getInput("origin").value = selected.join(",");
  }
  refreshAirportSummary();
}

function refreshAirportSummary() {
  const origin = codeFromInputValue(getInput("origin").value);
  const destination = codeFromInputValue(getInput("destination").value);
  airportSummary.textContent = `${airportLabel(origin)} to ${airportLabel(destination)}`;
}

function renderAirports(payload) {
  airportCatalog = payload;
  airportOptions.innerHTML = payload.airports
    .map((airport) => `<option value="${airport.code}">${airport.city} - ${airport.name}, ${airport.country}</option>`)
    .join("");

  routePresetsRoot.innerHTML = payload.routePresets
    .map(
      (preset) => `
        <button class="preset-chip" type="button" data-origin="${preset.origin}" data-destination="${preset.destination}" title="${preset.description}">
          ${preset.label}
        </button>
      `
    )
    .join("");

  londonAirportsRoot.innerHTML = londonAirportCodes
    .map(
      (code) => `
        <label class="airport-check">
          <input type="checkbox" value="${code}" ${code === "LHR" || code === "LGW" ? "checked" : ""}>
          <span>${code}</span>
        </label>
      `
    )
    .join("");
  syncOriginFromAirportChecks();

  refreshAirportSummary();
}

async function loadAirports() {
  try {
    const response = await fetch(`${apiBase}/airports`);
    if (!response.ok) {
      throw new Error("Airport list unavailable");
    }
    renderAirports(await response.json());
  } catch {
    airportSummary.textContent = "Airport helper unavailable; IATA codes still work.";
  }
}

function renderAirlines(payload) {
  airlineCatalog = payload;
  airlinePreferencesRoot.innerHTML = payload.groups
    .map(
      (group) => `
        <div class="airline-group" data-airline-group="${group.id}">
          <span>${group.label}</span>
          <div class="airline-checks-row">
            ${group.airlines
              .map(
                (airline) => `
                  <label class="airport-check airline-check">
                    <input type="checkbox" value="${airline}">
                    <span>${airline}</span>
                  </label>
                `
              )
              .join("")}
          </div>
        </div>
      `
    )
    .join("");
  refreshAirlineSummary();
}

async function loadAirlines() {
  try {
    const response = await fetch(`${apiBase}/airlines`);
    if (!response.ok) {
      throw new Error("Airline list unavailable");
    }
    renderAirlines(await response.json());
  } catch {
    airlineSummary.textContent = "Airline helper unavailable; all airlines included.";
  }
}

function setPreferredAirlines(airlines) {
  const selected = new Set(airlines);
  for (const input of airlinePreferencesRoot.querySelectorAll("input")) {
    input.checked = selected.has(input.value);
  }
  refreshAirlineSummary();
}

function airlinesForGroups(groupIds) {
  const ids = new Set(groupIds);
  return airlineCatalog.groups.filter((group) => ids.has(group.id)).flatMap((group) => group.airlines);
}

function refreshAirlineSummary() {
  const selected = selectedPreferredAirlines();
  airlineSummary.textContent = selected.length
    ? `Only showing trips operated by: ${selected.join(", ")}`
    : "All airlines included";
}

function renderChildren() {
  childrenRoot.innerHTML = defaultChildren
    .map(
      (child) => `
        <article class="child-card">
          <h3>${child.label} <span class="level-badge" data-level-for="${child.id}">${schoolLevelForAge(child.age)}</span></h3>
          <div class="child-fields">
            <label>
              <span>Age</span>
              <input name="${child.id}-age" type="number" min="0" max="17" value="${child.age}" required>
            </label>
            <label>
              <span>Year</span>
              <input name="${child.id}-schoolYear" value="${child.schoolYear}">
            </label>
            <label class="wide">
              <span>School</span>
              <input name="${child.id}-schoolName" value="${child.schoolName}" required>
            </label>
            <label class="wide">
              <span>Term source URL</span>
              <input name="${child.id}-schoolUrl" type="url" value="${child.schoolUrl}" placeholder="https://">
            </label>
            <label>
              <span>Term ends</span>
              <input name="${child.id}-termEnds" type="date" value="${child.termEnds}" required>
            </label>
            <label>
              <span>Term starts</span>
              <input name="${child.id}-termStarts" type="date" value="${child.termStarts}" required>
            </label>
          </div>
        </article>
      `
    )
    .join("");
}

function refreshLevelBadges() {
  for (const child of defaultChildren) {
    const age = Number(getInput(`${child.id}-age`)?.value || child.age);
    const badge = document.querySelector(`[data-level-for="${child.id}"]`);
    if (badge) {
      badge.textContent = schoolLevelForAge(age);
    }
  }
}

function collectRequest() {
  const data = new FormData(form);
  const children = defaultChildren.map((child) => ({
    id: child.id,
    age: Number(data.get(`${child.id}-age`)),
    schoolName: data.get(`${child.id}-schoolName`),
    schoolUrl: data.get(`${child.id}-schoolUrl`),
    schoolYear: data.get(`${child.id}-schoolYear`),
    termEnds: data.get(`${child.id}-termEnds`),
    termStarts: data.get(`${child.id}-termStarts`)
  }));

  return {
    origin: codeFromInputValue(data.get("origin")),
    destination: codeFromInputValue(data.get("destination")),
    tripType: "return",
    adults: Number(data.get("adults")),
    childrenAges: children.map((child) => child.age),
    children,
    leaveEarlyDays: Number(data.get("leaveEarlyDays")),
    returnLateDays: Number(data.get("returnLateDays")),
    minTripDays: Number(data.get("minTripDays")),
    tripFlexDays: Number(data.get("tripFlexDays")),
    preferredAirlines: selectedPreferredAirlines(),
    overrideWindow: {
      enabled: data.get("overrideEnabled") === "on",
      startDate: data.get("overrideStartDate"),
      endDate: data.get("overrideEndDate")
    }
  };
}

function flagLabel(flag) {
  return flag
    .split("-")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

function renderWindow(travelWindow) {
  if (!travelWindow.hasOverlap) {
    windowCard.className = "error-state";
    windowCard.textContent = "The selected school windows do not overlap. Adjust the dates or flexibility.";
    return;
  }

  windowCard.className = "window-range";
  windowCard.innerHTML = `
    <div class="date-band">
      <span>${travelWindow.source === "override" ? "Override search dates" : "Eligible return-trip dates"}</span>
      <strong>${formatDate(travelWindow.startDate)} to ${formatDate(travelWindow.endDate)}</strong>
    </div>
    ${travelWindow.childWindows
      .map(
        (childWindow) => `
          <div class="child-window">
            <strong>${childWindow.schoolName || childWindow.childId}</strong><br>
            ${formatDate(childWindow.startDate)} to ${formatDate(childWindow.endDate)}
          </div>
        `
      )
      .join("")}
  `;
}

function addDaysToIsoDate(dateString, days) {
  if (!dateString) {
    return "";
  }
  const date = new Date(`${dateString}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

function syncOverrideControls() {
  const enabled = overrideEnabledInput.checked;
  overrideStartInput.disabled = !enabled;
  overrideEndInput.disabled = !enabled;
  if (enabled && overrideStartInput.value && !overrideEndInput.value) {
    overrideEndInput.value = addDaysToIsoDate(overrideStartInput.value, 1);
  }
}

function renderResults(itineraries) {
  resultCount.textContent = itineraries.length;

  if (itineraries.length === 0) {
    resultsRoot.innerHTML = '<div class="empty-state">No recommended flights fit this search yet.</div>';
    return;
  }

  resultsRoot.innerHTML = itineraries
    .map((itinerary) => {
      const stopText = itinerary.isDirect ? "Direct" : `${itinerary.stops} stop via ${itinerary.connectionAirport}`;
      const layoverText = itinerary.isDirect ? "No layover" : formatMinutes(itinerary.layoverDurationMinutes);
      const airlineText = itinerary.airlineSummary ? `<div><span>Airlines</span><strong>${itinerary.airlineSummary}</strong></div>` : "";
      const tripDays = itinerary.tripDays ? `<div><span>Trip length</span><strong>${itinerary.tripDays} days</strong></div>` : "";

      return `
        <article class="result-card">
          <div>
            <div class="price">${formatCurrency(itinerary.price, itinerary.currency)}</div>
            <div class="score">${itinerary.source || "Flight result"}</div>
          </div>
          <div class="itinerary-main">
            <div class="date-pair">
              <div class="date-cell">
                <span>Outbound</span>
                <strong>${formatDate(itinerary.outboundDate)}</strong>
              </div>
              <div class="date-cell">
                <span>Inbound</span>
                <strong>${formatDate(itinerary.inboundDate)}</strong>
              </div>
            </div>
            <div class="meta-row">
              <div><span>Stops</span><strong>${stopText}</strong></div>
              <div><span>Journey time</span><strong>${formatMinutes(itinerary.totalDurationMinutes)}</strong></div>
              <div><span>Layover</span><strong>${layoverText}</strong></div>
              ${tripDays}
              ${airlineText}
            </div>
            ${renderSegments("Outbound", itinerary.outboundSegments)}
            ${renderSegments("Return", itinerary.returnSegments)}
            <div class="tags">
              ${itinerary.reasonFlags
                .map((flag) => `<span class="tag ${flag === "connection-risk" ? "warn" : ""}">${flagLabel(flag)}</span>`)
                .join("")}
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderSegments(label, segments = []) {
  if (!Array.isArray(segments) || segments.length === 0) {
    if (label === "Return") {
      return `
        <div class="segment-list muted-segments">
          <span>Return</span>
          <div class="segment-row">
            <strong>Details pending</strong>
            <span>Return flight times were not available for this result.</span>
            <span></span>
          </div>
        </div>
      `;
    }
    return "";
  }

  return `
    <div class="segment-list">
      <span>${label}</span>
      ${segments
        .map(
          (segment) => `
            <div class="segment-row">
              <strong>${segment.from} -> ${segment.to}</strong>
              <span>${formatSegmentTime(segment.departTime)} -> ${formatSegmentTime(segment.arriveTime)}</span>
              <span>${segment.airline || ""} ${segment.flightNumber || ""}</span>
              <small>${segment.fromName || ""}${segment.toName ? ` to ${segment.toName}` : ""}</small>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function formatSegmentTime(value) {
  if (!value) {
    return "";
  }
  const [date, time] = value.split(" ");
  return `${date || ""} ${time || ""}`.trim();
}

function renderProviderNotice(payload) {
  const pieces = [];
  if (payload.provider) {
    pieces.push(`Provider: ${payload.provider}`);
  }
  if (payload.searchMeta?.datePairsSearched) {
    pieces.push(`${payload.searchMeta.datePairsSearched} date pairs searched`);
  }
  if (payload.travelWindow?.minTripDays) {
    pieces.push(`${payload.travelWindow.minTripDays}-${payload.travelWindow.maxTripDays} trip days`);
  }
  if (payload.providerWarning) {
    pieces.push(payload.providerWarning);
  }
  if (payload.searchMeta?.partialErrors?.length) {
    pieces.push(`${payload.searchMeta.partialErrors.length} date pair errors`);
  }
  if (payload.searchMeta?.airlineFilter?.length) {
    pieces.push(`Airline filter: ${payload.searchMeta.airlineFilter.join(", ")}`);
  }
  if (payload.searchMeta?.filteredByAirlines) {
    pieces.push(`${payload.searchMeta.filteredByAirlines} results hidden by airline preference`);
  }

  if (pieces.length) {
    const notice = document.createElement("div");
    notice.className = payload.providerWarning ? "error-state" : "empty-state";
    notice.textContent = pieces.join(" · ");
    resultsRoot.prepend(notice);
  }

  if (payload.searchMeta?.partialErrors?.length && payload.itineraries?.length === 0) {
    const details = document.createElement("div");
    details.className = "error-state";
    details.innerHTML = payload.searchMeta.partialErrors
      .map((error) => `${error.outboundDate} to ${error.inboundDate}: ${error.message}`)
      .join("<br>");
    resultsRoot.append(details);
  }
}

function renderError(message) {
  resultCount.textContent = "0";
  resultsRoot.innerHTML = `<div class="error-state">${message}</div>`;
}

function renderSchoolStatus(message, isError = false) {
  schoolStatus.textContent = message;
  schoolStatus.classList.toggle("is-error", isError);
}

function populateFreemensWindows(catalog) {
  freemensWindowSelect.innerHTML = catalog.holidayWindows
    .map(
      (window) =>
        `<option value="${window.id}">${window.label} (${window.junior.termEnds} to ${window.junior.termStarts})</option>`
    )
    .join("");

  const summer2026 = catalog.holidayWindows.find((window) => window.label === "Summer holiday 2026");
  if (summer2026) {
    freemensWindowSelect.value = summer2026.id;
  }

  freemensWindowSelect.disabled = false;
  applyFreemensButton.disabled = false;
}

async function loadFreemensDates() {
  loadFreemensButton.disabled = true;
  loadFreemensButton.textContent = "Loading";
  renderSchoolStatus("Loading stored Freemen's term dates...");

  try {
    const response = await fetch(`${apiBase}/schools/freemens/term-dates`);
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Could not load Freemen's dates");
    }

    freemensCatalog = payload;
    populateFreemensWindows(payload);
    renderSchoolStatus(termDateStatusMessage(payload));
  } catch (error) {
    renderSchoolStatus(error.message, true);
  } finally {
    loadFreemensButton.disabled = false;
    loadFreemensButton.textContent = "Load Freemen's dates";
  }
}

async function refreshFreemensDates() {
  refreshFreemensButton.disabled = true;
  refreshFreemensButton.textContent = "Refreshing";
  renderSchoolStatus("Refreshing Freemen's term dates from the school page...");

  try {
    const response = await fetch(`${apiBase}/schools/freemens/term-dates/refresh`, {
      method: "POST"
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Could not refresh Freemen's dates");
    }

    freemensCatalog = payload;
    populateFreemensWindows(payload);
    renderSchoolStatus(termDateStatusMessage(payload));
  } catch (error) {
    renderSchoolStatus(error.message, true);
  } finally {
    refreshFreemensButton.disabled = false;
    refreshFreemensButton.textContent = "Refresh stored dates";
  }
}

function termDateStatusMessage(payload) {
  const cache = payload.cache || {};
  const pieces = [`Loaded ${payload.holidayWindows.length} holiday windows`];
  pieces.push(cache.servedFromCache ? "from storage" : "from school page");
  if (cache.refreshed) {
    pieces.push("stored for next time");
  }
  if (cache.furthestBreakEndDate) {
    pieces.push(`furthest break ends ${cache.furthestBreakEndDate}`);
  }
  if (cache.refreshFailed) {
    pieces.push(`refresh failed, using stored copy: ${cache.refreshError}`);
  }
  return pieces.join(" · ");
}

function applyFreemensDates() {
  if (!freemensCatalog) {
    return;
  }

  const selectedWindow = freemensCatalog.holidayWindows.find((window) => window.id === freemensWindowSelect.value);
  if (!selectedWindow) {
    renderSchoolStatus("Select a holiday window first.", true);
    return;
  }

  for (const child of defaultChildren) {
    const age = Number(getInput(`${child.id}-age`).value || child.age);
    const level = schoolLevelForAge(age);
    const dates = selectedWindow[level];
    getInput(`${child.id}-schoolName`).value = freemensCatalog.schoolName;
    getInput(`${child.id}-schoolUrl`).value = freemensCatalog.url;
    getInput(`${child.id}-termEnds`).value = dates.termEnds;
    getInput(`${child.id}-termStarts`).value = dates.termStarts;
  }

  applyTripLengthDefaults(selectedWindow);
  refreshLevelBadges();
  renderSchoolStatus(`Applied ${selectedWindow.label}. Age 12 and older uses Senior School dates.`);
}

function applyTripLengthDefaults(window) {
  if (window.type === "half-term") {
    getInput("minTripDays").value = 7;
    getInput("tripFlexDays").value = 2;
    return;
  }

  if (window.label.includes("Summer")) {
    getInput("minTripDays").value = 21;
    getInput("tripFlexDays").value = 7;
    return;
  }

  getInput("minTripDays").value = 10;
  getInput("tripFlexDays").value = 4;
}

async function checkHealth() {
  try {
    const response = await fetch(`${apiBase}/health`);
    if (!response.ok) {
      throw new Error("Service unavailable");
    }
    const health = await response.json();
    serviceStatus.textContent = `${health.activeProvider || health.provider} provider ready`;
    statusPanel.classList.add("is-ok");
  } catch {
    serviceStatus.textContent = "Local service unavailable";
    statusPanel.classList.remove("is-ok");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "Searching";

  try {
    const response = await fetch(`${apiBase}/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(collectRequest())
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Search failed");
    }

    renderWindow(payload.travelWindow);
    renderResults(payload.itineraries);
    renderProviderNotice(payload);
  } catch (error) {
    renderError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Search flights";
  }
});

renderChildren();
form.addEventListener("input", (event) => {
  if (event.target.name === "origin" || event.target.name === "destination") {
    refreshAirportSummary();
  }
});
routePresetsRoot.addEventListener("click", (event) => {
  const button = event.target.closest("[data-origin]");
  if (!button) {
    return;
  }

  getInput("origin").value = button.dataset.origin;
  getInput("destination").value = button.dataset.destination;
  if (button.dataset.destination === "CAN" || button.dataset.destination === "HKG" || button.dataset.destination === "SZX") {
    setLondonAirportSelection(["LHR", "LGW"]);
  }
  refreshAirportSummary();
});
londonAirportsRoot.addEventListener("change", syncOriginFromAirportChecks);
homeAirportsButton.addEventListener("click", () => setLondonAirportSelection(["LHR", "LGW"]));
holidayAirportsButton.addEventListener("click", () => setLondonAirportSelection(londonAirportCodes));
airlinePreferencesRoot.addEventListener("change", refreshAirlineSummary);
airlineActionsRoot.addEventListener("click", (event) => {
  const button = event.target.closest("[data-airline-preset]");
  if (!button) {
    return;
  }

  const preset = button.dataset.airlinePreset;
  if (preset === "clear") {
    setPreferredAirlines([]);
  } else if (preset === "european") {
    setPreferredAirlines(airlinesForGroups(["european-long-haul", "european-holiday"]));
  } else {
    setPreferredAirlines(airlinesForGroups([preset]));
  }
});

function setLondonAirportSelection(codes) {
  for (const input of londonAirportsRoot.querySelectorAll("input")) {
    input.checked = codes.includes(input.value);
  }
  syncOriginFromAirportChecks();
}
childrenRoot.addEventListener("input", (event) => {
  if (event.target.name?.endsWith("-age")) {
    refreshLevelBadges();
  }
});
loadFreemensButton.addEventListener("click", loadFreemensDates);
refreshFreemensButton.addEventListener("click", refreshFreemensDates);
applyFreemensButton.addEventListener("click", applyFreemensDates);
overrideEnabledInput.addEventListener("change", syncOverrideControls);
overrideStartInput.addEventListener("change", () => {
  overrideEndInput.value = addDaysToIsoDate(overrideStartInput.value, 1);
});
loadAirports();
loadAirlines();
checkHealth();
