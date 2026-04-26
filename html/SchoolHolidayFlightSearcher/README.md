# Family Flight Finder

A personal family holiday flight planner. V1 runs as a small Node web service with a static frontend, deterministic school-holiday date logic, and mock flight data.

## Quick Local Run

```bash
npm test
PORT=3010 npm start
```

Open:

```text
http://127.0.0.1:3010/
```

## Docker Run

```bash
docker compose up --build -d
```

Open:

```text
http://YOUR-SERVER-IP:3010/
```

## Optional Gemini Fallback

Freemen's term dates are parsed deterministically first. Gemini is only used as a fallback if that parser cannot extract enough dates.

Create a `.env` file beside `docker-compose.yml`:

```text
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash-lite
```

Do not commit `.env`. The repository ignores it.

## SerpApi Flight Search

To use real Google Flights results via SerpApi, create `.env` beside `docker-compose.yml`:

```text
FLIGHT_PROVIDER=serpapi
SERPAPI_API_KEY=your-key-here
SERPAPI_CURRENCY=GBP
SERPAPI_GOOGLE_COUNTRY=uk
SERPAPI_LANGUAGE=en
SERPAPI_MAX_DATE_PAIRS=4
SERPAPI_RESULTS_PER_PAIR=5
SERPAPI_RETURN_DETAILS_LIMIT=3
SERPAPI_RETURN_REVERSE_FALLBACK=true
SERPAPI_STOPS=2
SERPAPI_DEEP_SEARCH=false
```

`SERPAPI_MAX_DATE_PAIRS` controls the main search cost. `SERPAPI_RETURN_DETAILS_LIMIT` controls how many cheap results get extra return-flight detail lookups. `SERPAPI_RETURN_REVERSE_FALLBACK=true` may use an extra one-way return lookup when Google Flights does not return return-sector details from the round-trip token. Start with `4` and `3`.

## Unraid + Existing Nginx Docker

Use this app as a separate container and let your existing Nginx proxy to it. Do not replace your existing Nginx config. Add only a new location such as `/flight-searcher/`.

Beginner-friendly deployment instructions are in:

```text
docs/UNRAID_DEPLOYMENT.md
```

The sample Nginx snippet is in:

```text
deploy/nginx-location.example.conf
```

Recommended URL:

```text
http://YOUR-SERVER/flight-searcher/
```

## Why This Is Not Static-Only

The page calls backend endpoints:

```text
GET /api/health
GET /api/airports
POST /api/search
```

That backend will also be where future private flight API keys or provider integrations live. Keeping it separate now avoids painting the project into a corner.
