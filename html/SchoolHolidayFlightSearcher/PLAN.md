# Family Flight Search Frontend Plan

## Summary
Build a recommendation-led family flight search app as a working v1 vertical slice. The first implementation uses a plain HTML/CSS/JS frontend, a small Node backend, deterministic school-holiday date logic, and a mock flight provider behind a provider adapter. Real Skyscanner or compatible API integration is deferred until API access is confirmed.

## V1 Scope
- Frontend defaults to `2 adults + 2 children`, with child ages `9` and `11`.
- Return trips only, with explicit `origin` and `destination` inputs.
- Airport fields support IATA autocomplete and route presets for frequent China/family and European ski routes.
- Broad city airport codes are expanded for SerpApi where useful, for example `LON` becomes `LHR,LGW,STN,LTN,LCY`.
- Searches include trip-length controls: minimum trip days plus extra flexible days.
- Results default to cheapest-first sorting and hide internal score details from the UI.
- Two editable child cards from day one:
  - `age`
  - `school name`
  - `school term source URL`
  - `school year/group`
  - manual term boundaries
- Manual school term dates are the source of truth in v1.
- Freemen's term dates and half-term breaks can be fetched from the school's public term-date page and applied to both children.
- Children aged `12` and older use Senior School dates; younger children use Junior School dates.
- School source URLs are collected in the UI and Freemen's is the first allowlisted fetch integration.
- Results are produced by a backend mock flight provider and ranked with real date-window and scoring logic.
- Real flight results can be fetched with SerpApi when `FLIGHT_PROVIDER=serpapi` and `SERPAPI_API_KEY` are configured.

## Deferred Scope
- Live Skyscanner or direct airline/provider integrations beyond SerpApi.
- Automatic school page scraping.
- AI-assisted school/PDF extraction.
- Saved school profiles.
- Multi-city, one-way, nearby-airport, cabin-class, and hotel/car search.

## Product Behavior
- The app computes a shared travel window for all selected children.
- Each child's holiday window is defined as:
  - first travel day: the day after `termEnds`
  - last return day: the day before `termStarts`
- Flexibility expands that holiday window:
  - outbound may be up to `leaveEarlyDays` before `termEnds`
  - inbound may be up to `returnLateDays` after `termStarts`
- The family window is the intersection of every child's expanded window.
- Only return itineraries with outbound and inbound dates inside the shared family window are eligible.
- Results should favor recommendation cards over a raw flight-table UX.
- Each result card explains:
  - total price
  - outbound and inbound dates
  - stop pattern
  - connection airport and layover when present
  - why the option fits the school constraints
  - score/tradeoff labels such as `cheapest`, `faster`, `short layover`, and `connection risk`

## Architecture
- Frontend:
  - `public/index.html`
  - `public/styles.css`
  - `public/app.js`
  - structured sections for route, travelers, school calendars, flexibility, window summary, and ranked results
- Backend:
  - `server.js` serves static files and JSON endpoints
  - `src/domain.js` owns date-window intersection and ranking
- `src/schools/freemens.js` fetches and parses Freemen's allowlisted term-date page, including half-term breaks
  - `src/providers/mockFlightProvider.js` supplies deterministic sample itineraries
  - `src/providers/serpApiFlightProvider.js` searches Google Flights via SerpApi
  - SerpApi return-sector details are enriched for the cheapest few results, controlled by `SERPAPI_RETURN_DETAILS_LIMIT`
  - `src/providers/flightProvider.js` chooses the configured provider and keeps mock fallback behavior
  - `src/providers/geminiExtractor.js` provides an optional Gemini fallback for term-date extraction
  - future real providers must implement the same flight-provider interface
- API:
  - `GET /api/health`
  - `GET /api/airports`
  - `POST /api/search`
  - `GET /api/schools/freemens/term-dates`
- No browser-exposed flight API keys.

## Public Interfaces / Data Shapes
- `ChildProfile`: `id`, `age`, `schoolName`, `schoolUrl`, `schoolYear`, `termEnds`, `termStarts`
- `SearchRequest`: `origin`, `destination`, `tripType=return`, `adults`, `childrenAges[]`, `children[]`, `leaveEarlyDays`, `returnLateDays`
- `TravelWindow`: `startDate`, `endDate`, `childWindows[]`
- `ItinerarySummary`: `id`, `price`, `currency`, `outboundDate`, `inboundDate`, `isDirect`, `stops`, `totalDurationMinutes`, `directBaselineDurationMinutes`, `connectionAirport`, `layoverDurationMinutes`, `score`, `reasonFlags[]`

## Ranking Logic
- Filter out itineraries outside the shared family window.
- Compute a transparent score where lower is better:
  - normalized price is the largest factor
  - add a directness penalty for 1-stop itineraries
  - add a duration penalty relative to route direct baseline
  - add a layover penalty, with heavier penalty for long layovers
- Sort by score, then price.
- Add reason flags after ranking:
  - cheapest eligible result
  - faster-than-average result
  - short layover
  - connection risk for long layovers
  - school-compatible date fit

## Security And Provider Notes
- V1 does not fetch arbitrary school URLs from the backend.
- V1 fetches only the allowlisted Freemen's term-date URL.
- Any future school URL fetch endpoint must include URL validation, redirect limits, timeouts, max response size, content-type checks, and internal-network blocking.
- Gemini fallback is configured with `GEMINI_API_KEY` and defaults to `gemini-2.5-flash-lite`.
- Skyscanner access is partner/commercial-gated and uses asynchronous live-price flows, so provider integration should be implemented only after credentials and workflow expectations are confirmed.

## Test Plan
- Date-window logic:
  - no flexibility
  - outbound allowed a few days before term end
  - inbound allowed a few days after term start
  - only windows valid for both children are returned
  - non-overlapping child windows fail clearly
- Flight filtering/ranking:
  - reject itineraries outside the shared family window
  - direct vs 1-stop when 1-stop is cheaper
  - demote excessive layovers
  - show connection airport and wait time correctly
  - expose score and reason flags
- Frontend UX:
  - defaults load with ages `9` and `11`
  - user can edit schools independently
  - result cards explain price, dates, stop pattern, and school fit
