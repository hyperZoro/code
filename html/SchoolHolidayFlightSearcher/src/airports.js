const AIRPORTS = [
  { code: "LON", name: "London all airports", city: "London", country: "United Kingdom", tags: ["home", "uk"] },
  { code: "LHR", name: "Heathrow", city: "London", country: "United Kingdom", tags: ["home", "uk"] },
  { code: "LGW", name: "Gatwick", city: "London", country: "United Kingdom", tags: ["home", "uk"] },
  { code: "STN", name: "Stansted", city: "London", country: "United Kingdom", tags: ["uk"] },
  { code: "LTN", name: "Luton", city: "London", country: "United Kingdom", tags: ["uk"] },
  { code: "LCY", name: "London City", city: "London", country: "United Kingdom", tags: ["uk"] },
  { code: "MAN", name: "Manchester", city: "Manchester", country: "United Kingdom", tags: ["uk"] },
  { code: "BHX", name: "Birmingham", city: "Birmingham", country: "United Kingdom", tags: ["uk"] },
  { code: "EDI", name: "Edinburgh", city: "Edinburgh", country: "United Kingdom", tags: ["uk"] },
  { code: "GLA", name: "Glasgow", city: "Glasgow", country: "United Kingdom", tags: ["uk"] },

  { code: "CAN", name: "Guangzhou Baiyun", city: "Guangzhou", country: "China", tags: ["china", "family"] },
  { code: "HKG", name: "Hong Kong International", city: "Hong Kong", country: "Hong Kong", tags: ["china", "family"] },
  { code: "SZX", name: "Shenzhen Bao'an", city: "Shenzhen", country: "China", tags: ["china", "family"] },
  { code: "PVG", name: "Shanghai Pudong", city: "Shanghai", country: "China", tags: ["china"] },
  { code: "SHA", name: "Shanghai Hongqiao", city: "Shanghai", country: "China", tags: ["china"] },
  { code: "PEK", name: "Beijing Capital", city: "Beijing", country: "China", tags: ["china"] },
  { code: "PKX", name: "Beijing Daxing", city: "Beijing", country: "China", tags: ["china"] },
  { code: "CTU", name: "Chengdu Shuangliu", city: "Chengdu", country: "China", tags: ["china"] },
  { code: "TFU", name: "Chengdu Tianfu", city: "Chengdu", country: "China", tags: ["china"] },

  { code: "GVA", name: "Geneva", city: "Geneva", country: "Switzerland", tags: ["ski", "alps"] },
  { code: "ZRH", name: "Zurich", city: "Zurich", country: "Switzerland", tags: ["ski", "alps"] },
  { code: "BSL", name: "EuroAirport Basel Mulhouse Freiburg", city: "Basel", country: "Switzerland/France", tags: ["ski", "alps"] },
  { code: "INN", name: "Innsbruck", city: "Innsbruck", country: "Austria", tags: ["ski", "alps"] },
  { code: "SZG", name: "Salzburg", city: "Salzburg", country: "Austria", tags: ["ski", "alps"] },
  { code: "MUC", name: "Munich", city: "Munich", country: "Germany", tags: ["ski", "alps"] },
  { code: "TRN", name: "Turin", city: "Turin", country: "Italy", tags: ["ski", "alps"] },
  { code: "MXP", name: "Milan Malpensa", city: "Milan", country: "Italy", tags: ["ski", "alps"] },
  { code: "BGY", name: "Milan Bergamo", city: "Bergamo", country: "Italy", tags: ["ski", "alps"] },
  { code: "VRN", name: "Verona", city: "Verona", country: "Italy", tags: ["ski", "alps"] },
  { code: "VCE", name: "Venice Marco Polo", city: "Venice", country: "Italy", tags: ["ski", "alps"] },
  { code: "LYS", name: "Lyon Saint-Exupery", city: "Lyon", country: "France", tags: ["ski", "alps"] },
  { code: "GNB", name: "Grenoble Alpes-Isere", city: "Grenoble", country: "France", tags: ["ski", "alps"] },
  { code: "CMF", name: "Chambery Savoie Mont Blanc", city: "Chambery", country: "France", tags: ["ski", "alps"] }
];

const ROUTE_PRESETS = [
  { id: "home-guangzhou", label: "Home town", origin: "LON", destination: "CAN", description: "London to Guangzhou Baiyun" },
  { id: "hong-kong", label: "Hong Kong", origin: "LON", destination: "HKG", description: "London to Hong Kong" },
  { id: "shenzhen", label: "Shenzhen", origin: "LON", destination: "SZX", description: "London to Shenzhen" },
  { id: "geneva-ski", label: "Geneva ski", origin: "LON", destination: "GVA", description: "London to Geneva" },
  { id: "zurich-ski", label: "Zurich ski", origin: "LON", destination: "ZRH", description: "London to Zurich" },
  { id: "innsbruck-ski", label: "Innsbruck ski", origin: "LON", destination: "INN", description: "London to Innsbruck" },
  { id: "turin-ski", label: "Turin ski", origin: "LON", destination: "TRN", description: "London to Turin" },
  { id: "lyon-ski", label: "Lyon ski", origin: "LON", destination: "LYS", description: "London to Lyon" }
];

function airportPayload() {
  return {
    airports: AIRPORTS,
    routePresets: ROUTE_PRESETS
  };
}

module.exports = {
  AIRPORTS,
  ROUTE_PRESETS,
  airportPayload
};
