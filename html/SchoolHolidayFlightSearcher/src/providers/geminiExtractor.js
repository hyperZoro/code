const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash-lite";

function extractJson(text) {
  const trimmed = String(text || "").trim();
  if (trimmed.startsWith("{")) {
    return JSON.parse(trimmed);
  }

  const match = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (match) {
    return JSON.parse(match[1]);
  }

  throw new Error("Gemini did not return JSON.");
}

async function extractFreemensTermDatesWithGemini(pageText) {
  if (!process.env.GEMINI_API_KEY) {
    return null;
  }

  const prompt = `
Extract City of London Freemen's School term dates from the text below.
Return only JSON with this exact shape:
{
  "source": "gemini",
  "terms": [
    {
      "academicYear": "2025-2026",
      "name": "Autumn Term 2025",
      "starts": "2025-09-03",
      "ends": { "junior": "2025-12-12", "senior": "2025-12-12" },
      "halfTerm": { "begins": "2025-10-17", "recommences": "2025-11-03" }
    }
  ]
}
Use ISO YYYY-MM-DD dates. Include Junior School and Senior School end dates. If the same date applies to both, repeat it. Include halfTerm when present, otherwise use null values.

TEXT:
${pageText.slice(0, 50000)}
`;

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${process.env.GEMINI_API_KEY}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0,
        response_mime_type: "application/json"
      }
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini extraction failed: ${response.status} ${body.slice(0, 200)}`);
  }

  const payload = await response.json();
  const text = payload.candidates?.[0]?.content?.parts?.[0]?.text;
  return extractJson(text);
}

module.exports = {
  extractFreemensTermDatesWithGemini
};
