const airlineGroups = [
  {
    id: "chinese",
    label: "Chinese airlines",
    airlines: [
      "Air China",
      "China Eastern",
      "China Southern",
      "Hainan Airlines",
      "XiamenAir",
      "Sichuan Airlines",
      "Shenzhen Airlines",
      "Cathay Pacific",
      "Hong Kong Airlines"
    ]
  },
  {
    id: "ba",
    label: "British Airways",
    airlines: ["British Airways"]
  },
  {
    id: "european-long-haul",
    label: "European long haul",
    airlines: [
      "Virgin Atlantic",
      "Lufthansa",
      "SWISS",
      "Austrian",
      "Air France",
      "KLM",
      "Finnair",
      "Iberia",
      "Turkish Airlines"
    ]
  },
  {
    id: "european-holiday",
    label: "European holiday",
    airlines: [
      "easyJet",
      "Ryanair",
      "Jet2",
      "TUI Airways",
      "Wizz Air",
      "Vueling",
      "Eurowings",
      "ITA Airways",
      "Aegean Airlines",
      "Norwegian"
    ]
  }
];

function airlinePayload() {
  const airlines = [...new Set(airlineGroups.flatMap((group) => group.airlines))].sort((a, b) => a.localeCompare(b));
  return {
    groups: airlineGroups,
    airlines
  };
}

module.exports = {
  airlineGroups,
  airlinePayload
};
