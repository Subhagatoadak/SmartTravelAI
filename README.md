
# SmartTravelAPP 🌍✈️

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployed-streamlit-app-url.com)

**SmartTravelAPP** is your AI-powered, interactive travel companion. Choose any destination by typing or clicking on the map, provide your trip details, and instantly receive a comprehensive briefing: safety alerts, official advisories, visa guidance, weather, currency rates, health tips, cultural etiquette, and more—everything distilled by an LLM from live web sources.

---

## 📌 Table of Contents

1. [Features](#features)  
2. [Tech Stack](#tech-stack)  
3. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Configuration](#configuration)  
4. [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [Function Overviews](#function-overviews)  
7. [Requirements (`requirements.txt`)](#requirements-requirementstxt)  
8. [Disclaimer](#disclaimer)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## ✨ Features

- **🗺 Interactive Map & Text Input**  
  Seamlessly select your destination by typing a city/country name or clicking on the global map (geocoded in real time).  
- **🛡 Safety News Analysis**  
  Crawls the web for the latest headlines on war, unrest, kidnapping, etc., then uses an LLM to produce a clear safety advisory.  
- **📋 Official Travel Advisory**  
  Scrapes government advisory pages and asks the LLM to summarize entry rules, restricted areas, and precautions.  
- **🛂 Visa Requirements Guide**  
  Finds live visa-information pages and transforms raw content into a step-by-step application guide tailored to your nationality.  
- **☀️ Local Weather**  
  Displays current weather conditions for your chosen coordinates via OpenWeatherMap.  
- **💱 Currency Conversion**  
  Shows up-to-date exchange rates between your home currency and the destination’s currency.  
- **❤️‍🩹 Health & Vaccination Tips**  
  LLM-generated overview of common travel health considerations and recommended vaccines (with professional disclaimer).  
- **🤝 Cultural Etiquette**  
  Brief AI-curated tips on local customs and manners.  
- **🎒 Packing Checklist**  
  Customizable packing recommendations based on destination, duration, trip type, and weather.  
- **⚡ Caching & Performance**  
  Leverages Streamlit’s caching to minimize repeat crawls and speed up results.  

---

## 🛠 Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)  
- **Mapping**: [Folium](https://python-visualization.github.io/folium/), [streamlit-folium](https://github.com/randyzwitch/streamlit-folium)  
- **Geocoding**: [Geopy](https://github.com/geopy/geopy) (Nominatim)  
- **Web Search**: [duckduckgo-search](https://github.com/deedy5/duckduckgo_search)  
- **Web Scraping**: [Requests](https://docs.python-requests.org/), [Beautiful Soup 4](https://www.crummy.com/software/BeautifulSoup/)  
- **Country/Currency Data**: [pycountry](https://github.com/flyingcircusio/pycountry)  
- **Weather API**: [OpenWeatherMap](https://openweathermap.org/)  
- **Currency API**: [ExchangeRate-API](https://www.exchangerate-api.com/)  
- **LLM Integration**: Your own `llm_service.py` (e.g. OpenAI, Claude, Google Gemini)  
- **Environment**: [python-dotenv](https://github.com/theskumar/python-dotenv)  

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+  
- `pip` package manager  
- Internet access for crawling, APIs & LLM calls  

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/SmartTravelAPP.git
   cd SmartTravelAPP
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate.bat     # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Create a `.env` file** in the project root:  
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   WEATHER_API_KEY=your_openweathermap_key
   CURRENCY_API_KEY=your_exchangerate_api_key
   ```
2. **Provide or implement** `llm_service.py` with at least:
   ```python
   def generate_llm_response(prompt: str) -> str:
       # call your LLM (OpenAI, Claude, etc.) and return the response text
       ...
   ```

---

## ▶️ Usage

Run the app locally:
```bash
streamlit run app.py
```
- **Text input** or **map click** to choose destination  
- **Select** your nationality, home currency, trip duration, trip type  
- Click **“Get Full Travel Briefing”** to display all sections  

---

## 📂 Project Structure

```
SmartTravelAPP/
├── app.py
├── llm_service.py
├── requirements.txt
├── .env.example
└── docs/
    └── demo.png
```

---

## 🔍 Function Overviews

- `fetch_safety_news(country: str) → str`  
  Crawls DuckDuckGo for conflict/unrest headlines, scrapes titles, returns Markdown list.  

- `fetch_travel_advisory(country_code: str) → str`  
  Finds official government advisory pages via web search, scrapes paragraphs, LLM-summarizes.  

- `fetch_visa_requirements(nationality: str, dest_code: str) → str`  
  Searches for and scrapes visa-info sites, returns an LLM-generated step-by-step guide.  

- **Additional utilities**:  
  - Geocoding (`geopy`)  
  - Weather (`OpenWeatherMap`)  
  - Currency (`ExchangeRate-API`)  
  - Caching (`@st.cache_data`)  

---

## 📄 Requirements (`requirements.txt`)

```text
streamlit
streamlit-folium
folium
geopy
duckduckgo-search
requests
beautifulsoup4
pycountry
python-dotenv
openai              # or your LLM SDK
```

---

## ⚠️ Disclaimer

SmartTravelAPP aggregates data from public web sources and AI models. While it strives for accuracy, information may be incomplete or outdated. **Always verify** critical details—visa rules, safety alerts, health advice, emergency contacts—through official government or professional channels before traveling. This tool is for informational purposes only.

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add some feature"`)  
4. Push to branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
```