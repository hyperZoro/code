const http = require("node:http");
const path = require("node:path");
const fs = require("node:fs/promises");
const { URL } = require("node:url");
const { loadDotEnv } = require("./src/env");
const { buildSearchResponse } = require("./src/domain");
loadDotEnv();
const { providerHealth, searchFlights } = require("./src/providers/flightProvider");
const { getFreemensTermDates } = require("./src/schools/termDateCache");
const { airportPayload } = require("./src/airports");
const { airlinePayload } = require("./src/airlines");

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || "0.0.0.0";
const PUBLIC_DIR = path.join(__dirname, "public");
const MAX_BODY_BYTES = 1_000_000;

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body)
  });
  res.end(body);
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (Buffer.byteLength(body) > MAX_BODY_BYTES) {
        reject(Object.assign(new Error("Request body is too large."), { statusCode: 413 }));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        reject(Object.assign(new Error("Request body must be valid JSON."), { statusCode: 400 }));
      }
    });
    req.on("error", reject);
  });
}

async function serveStatic(req, res, pathname) {
  const requestedPath = pathname === "/" ? "/index.html" : pathname;
  const decodedPath = decodeURIComponent(requestedPath);
  const filePath = path.normalize(path.join(PUBLIC_DIR, decodedPath));

  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  try {
    const data = await fs.readFile(filePath);
    const ext = path.extname(filePath);
    res.writeHead(200, {
      "content-type": contentTypes[ext] || "application/octet-stream",
      "cache-control": "no-cache"
    });
    res.end(data);
  } catch (error) {
    if (error.code === "ENOENT") {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    throw error;
  }
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

    if (req.method === "GET" && url.pathname === "/api/health") {
      sendJson(res, 200, {
        ok: true,
        service: "school-holiday-flight-searcher",
        ...providerHealth(),
        geminiConfigured: Boolean(process.env.GEMINI_API_KEY)
      });
      return;
    }

    if (req.method === "GET" && url.pathname === "/api/schools/freemens/term-dates") {
      const response = await getFreemensTermDates();
      sendJson(res, 200, response);
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/schools/freemens/term-dates/refresh") {
      const response = await getFreemensTermDates({ forceRefresh: true });
      sendJson(res, 200, response);
      return;
    }

    if (req.method === "GET" && url.pathname === "/api/airports") {
      sendJson(res, 200, airportPayload());
      return;
    }

    if (req.method === "GET" && url.pathname === "/api/airlines") {
      sendJson(res, 200, airlinePayload());
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/search") {
      const request = await readJsonBody(req);
      const providerResult = await searchFlights(request);
      const response = buildSearchResponse(request, providerResult.itineraries, providerResult);
      sendJson(res, 200, response);
      return;
    }

    if (url.pathname.startsWith("/api/")) {
      sendJson(res, 404, { error: "Unknown API endpoint." });
      return;
    }

    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405);
      res.end("Method not allowed");
      return;
    }

    await serveStatic(req, res, url.pathname);
  } catch (error) {
    const statusCode = error.statusCode || 500;
    sendJson(res, statusCode, { error: error.message || "Unexpected server error." });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`School holiday flight searcher listening on http://${HOST}:${PORT}`);
});
