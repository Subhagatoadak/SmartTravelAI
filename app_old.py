# app.py
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import requests
from requests.exceptions import RequestException, Timeout
from bs4 import BeautifulSoup
import pycountry
import time
from dotenv import load_dotenv
from typing import Tuple, Optional, List, Dict, Union, Any # Added Any
import traceback
import json # For parsing API responses
from datetime import datetime # For weather timestamp

# --- Alternative Search Library ---
from duckduckgo_search import DDGS # Using DuckDuckGo search

# --- LLM Service Import ---
try:
    from llm_service import generate_llm_response
except ImportError:
    st.error("Error: `llm_service.py` not found or `generate_llm_response` function is missing.")
    # You might want to stop execution if the LLM service is critical
    # st.stop()
    # Fallback: Define a dummy function if you want the app to run without LLM features
    def generate_llm_response(prompt: str) -> str:
        print("Warning: LLM Service not available. Returning placeholder response.")
        return f"Placeholder response for prompt: {prompt[:100]}..."

# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY") # Using ExchangeRate-API.com for this example

# Basic validation for API Keys
if not OPENAI_API_KEY:
    # Decide if this is fatal or just disables LLM features
    st.error("Error: OPENAI_API_KEY not found. LLM features will likely fail.")
    # st.stop() # Uncomment if OpenAI key is absolutely required
if not WEATHER_API_KEY:
    st.warning("Warning: WEATHER_API_KEY not found. Weather feature will be disabled.")
if not CURRENCY_API_KEY:
    st.warning("Warning: CURRENCY_API_KEY not found. Currency feature will be disabled.")

# --- Constants ---
DEFAULT_MAP_LOCATION = (20.5937, 78.9629) # Center of India approx.
DEFAULT_ZOOM = 4
REQUEST_TIMEOUT = 10
# Define currency API endpoint (Example for ExchangeRate-API)
CURRENCY_API_ENDPOINT = f"https://v6.exchangerate-api.com/v6/{CURRENCY_API_KEY}/latest/"
# WEATHER_API_ENDPOINT is now constructed inside the fetch_weather function

# --- Initialize Session State ---
if "coords" not in st.session_state:
    st.session_state.coords = DEFAULT_MAP_LOCATION
if "place_name" not in st.session_state:
    st.session_state.place_name = ""
if "map_center" not in st.session_state:
    st.session_state.map_center = DEFAULT_MAP_LOCATION
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = DEFAULT_ZOOM
if "last_known_country" not in st.session_state:
    st.session_state.last_known_country = None
if "last_known_country_code" not in st.session_state:
    st.session_state.last_known_country_code = None
if "place_input_key" not in st.session_state: # Holds the value for the text input widget
    st.session_state.place_input_key = ""
if "place_name_prev" not in st.session_state: # Used for text input change detection
    st.session_state.place_name_prev = ""


# --- Geocoding Setup ---
geolocator = Nominatim(user_agent="safetravelai_app_v8_mapfocus") # Updated user agent slightly

@st.cache_data(ttl=3600) # Cache for 1 hour
def geocode(place_name: str) -> Optional[Tuple[float, float]]:
    """Converts a place name (e.g., 'Paris, France') to coordinates."""
    if not place_name:
        print("geocode: Received empty place_name")
        return None
    print(f"Geocoding: '{place_name}'")
    try:
        # Added timeout adjustment, Nominatim default is 1s which is often too short
        location = geolocator.geocode(place_name, timeout=REQUEST_TIMEOUT + 5)
        if location:
            print(f"Geocode Success: {place_name} -> ({location.latitude}, {location.longitude})")
            return (location.latitude, location.longitude)
        else:
            print(f"Geocoding returned None for: {place_name}")
            return None
    except GeocoderTimedOut:
        st.error(f"Geocoding service timed out for '{place_name}'.")
        print(f"Geocoding timed out for: {place_name}")
        return None
    except GeocoderServiceError as e:
        st.error(f"Geocoding service error for '{place_name}': {e}")
        print(f"Geocoding service error for '{place_name}': {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during geocoding '{place_name}': {e}")
        traceback.print_exc()
        return None

@st.cache_data(ttl=3600) # Cache for 1 hour
def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Converts coordinates to a human-readable address/place name."""
    print(f"Reverse Geocoding: ({lat:.4f}, {lon:.4f})")
    try:
        # Ensure language is set, adjust timeout
        location = geolocator.reverse((lat, lon), language="en", timeout=REQUEST_TIMEOUT + 5)
        if location and location.address:
            print(f"Reverse Geocode Success: ({lat:.4f}, {lon:.4f}) -> {location.address}")
            return location.address
        else:
            print(f"Reverse geocoding returned None or no address for: ({lat:.4f}, {lon:.4f})")
            return None
    except GeocoderTimedOut:
        st.error(f"Reverse geocoding service timed out for ({lat:.4f}, {lon:.4f}).")
        print(f"Reverse geocoding timed out for: ({lat:.4f}, {lon:.4f})")
        return None
    except GeocoderServiceError as e:
        st.error(f"Reverse geocoding service error for ({lat:.4f}, {lon:.4f}): {e}")
        print(f"Reverse geocoding service error for ({lat:.4f}, {lon:.4f}): {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during reverse geocoding ({lat:.4f}, {lon:.4f}): {e}")
        traceback.print_exc()
        return None


def get_country_from_place(place_name: str) -> Optional[str]:
    """Extracts the canonical country name from a place string using pycountry."""
    if not place_name:
        print("get_country_from_place: Received empty place_name")
        return None

    # Split and meticulously strip whitespace from each part
    parts = [part.strip() for part in place_name.split(',')]
    print(f"get_country_from_place: Parsed parts: {parts}")

    if not parts:
        print("get_country_from_place: No parts after splitting.")
        return None

    # Iterate from the last part backwards, as country is usually last
    for i in range(len(parts) - 1, -1, -1):
        potential_country = parts[i]
        if not potential_country: # Skip empty parts (e.g., "City, , Country")
             print(f"get_country_from_place: Skipping empty part at index {i}")
             continue

        print(f"get_country_from_place: Checking part '{potential_country}'...")
        try:
            # Use search_fuzzy which is more robust for slightly different names/cases
            found_countries = pycountry.countries.search_fuzzy(potential_country)

            if found_countries:
                # Important: Return the official name found by pycountry
                # This standardizes the name (e.g., "united states" -> "United States")
                official_name = found_countries[0].name
                print(f"get_country_from_place: Found country '{official_name}' matching part '{potential_country}'")
                return official_name # Return the standardized name
            else:
                 print(f"get_country_from_place: No fuzzy match for part '{potential_country}'")

        except Exception as e:
             # Catch unexpected errors during search
             print(f"get_country_from_place: Error during pycountry search for '{potential_country}': {e}")
             # Optionally add traceback.print_exc() here for deeper debugging
             continue # Try the next part

    # If no part matched
    print(f"get_country_from_place: Could not determine country from any part of '{place_name}'.")
    return None


def get_country_code(country_name: str) -> Optional[str]:
    """Gets the ISO 3166-1 alpha-2 code for a country name using pycountry."""
    if not country_name:
        print("get_country_code: Received empty country_name")
        return None

    print(f"get_country_code: Attempting to find code for '{country_name}'...")
    try:
        # Use search_fuzzy first for robustness
        results = pycountry.countries.search_fuzzy(country_name)
        if results:
            country_code = results[0].alpha_2
            print(f"get_country_code: Found code '{country_code}' for country name '{country_name}' (matched: {results[0].name})")
            return country_code
        else:
            # Try a direct lookup as a fallback (case-sensitive)
            try:
                country = pycountry.countries.get(name=country_name)
                if country:
                    print(f"get_country_code: Found code '{country.alpha_2}' via direct lookup for '{country_name}'")
                    return country.alpha_2
            except KeyError:
                pass # Direct lookup failed, continue

            # If still nothing found
            st.warning(f"Could not find ISO code for country: {country_name}")
            print(f"get_country_code: pycountry search failed for '{country_name}'")
            return None
    except Exception as e:
        st.error(f"An unexpected error occurred getting country code for '{country_name}': {e}")
        traceback.print_exc() # Print full traceback to console
        return None

# --- Callback Functions for State Updates ---

def update_location_from_text():
    """Callback logic when text input potentially changes state."""
    place_from_input = st.session_state.get("place_input_key", "")

    # Only proceed if the text input value is different from the *last known place name*
    # This prevents redundant geocoding if the map click updated the text input
    if place_from_input != st.session_state.get("place_name", ""):
        print(f"\n--- Text Input Changed Detected ---")
        print(f"  Input Key: '{place_from_input}'")
        print(f"  Current Place Name: '{st.session_state.get('place_name')}'")

        st.session_state.place_name = place_from_input # Update the central place name state

        # Attempt to geocode the new place name from text input
        coords = geocode(place_from_input)
        country = get_country_from_place(place_from_input) # Attempt to get country from text

        st.session_state.last_known_country = country
        st.session_state.last_known_country_code = get_country_code(country) if country else None

        if coords:
            st.session_state.coords = coords
            st.session_state.map_center = coords # Update map center to match geocoded text
            st.session_state.map_zoom = 10 if place_from_input else DEFAULT_ZOOM # Zoom in more if specific place found
            print(f"Updated location from TEXT: '{place_from_input}' -> Coords: {coords}")
        else:
            # Geocoding failed for the text input
            # Option 1: Keep previous coords (might be confusing if text doesn't match map)
            # Option 2: Reset coords/map (might be disruptive)
            # Current choice: Keep previous coords but clear country if name is invalid
            print(f"Could not geocode text input: '{place_from_input}'. Keeping previous map coords, but updating country info.")
            if not country: # If we couldn't get a country either, clear related state
                 st.session_state.last_known_country = None
                 st.session_state.last_known_country_code = None

        # Update the tracking variable for the *next* comparison
        st.session_state.place_name_prev = place_from_input


def update_location_from_map(map_data):
    """Update state based on map click. This function now sets the text input state too."""
    if not map_data or not map_data.get("last_clicked"):
        return # No click data

    click_lat = map_data["last_clicked"]["lat"]
    click_lng = map_data["last_clicked"]["lng"]
    new_coords = (click_lat, click_lng)

    # Check if click is significantly different from current coords
    # Reduces redundant updates from minor map shifts or rapid clicks
    lat_diff = abs(new_coords[0] - st.session_state.get("coords", (0,0))[0])
    lon_diff = abs(new_coords[1] - st.session_state.get("coords", (0,0))[1])
    min_diff = 1e-5 # Threshold for detecting a meaningful click location change

    if lat_diff > min_diff or lon_diff > min_diff:
        print(f"\n--- Map Click Detected ---")
        print(f"  Clicked Coords: ({click_lat:.4f}, {click_lng:.4f})")

        st.session_state.coords = new_coords
        st.session_state.map_center = new_coords # Center map exactly on click

        # Attempt reverse geocoding to get a place name
        address = reverse_geocode(click_lat, click_lng)
        print(f"  Reverse Geocoded Address: {address}")

        if address:
            st.session_state.place_name = address # Update main place name state
            country = get_country_from_place(address)
            st.session_state.last_known_country = country
            st.session_state.last_known_country_code = get_country_code(country) if country else None
            # --- CRITICAL: Update the text input state ---
            st.session_state.place_input_key = address
            st.session_state.place_name_prev = address # Sync prev tracker
            print(f"Updated location from MAP: '{address}' -> Coords: {new_coords}")

        else:
            # Reverse geocoding failed, use coordinates as fallback name
            fallback_name = f"Lat: {click_lat:.4f}, Lng: {click_lng:.4f}"
            st.session_state.place_name = fallback_name
            st.session_state.last_known_country = None # Country unknown
            st.session_state.last_known_country_code = None
            # --- CRITICAL: Update the text input state ---
            st.session_state.place_input_key = fallback_name
            st.session_state.place_name_prev = fallback_name
            print(f"Reverse geocode failed. Updated location from MAP -> Coords: {new_coords}")

        # Zoom in slightly on click
        current_zoom = st.session_state.map_zoom
        st.session_state.map_zoom = max(current_zoom, 8) if current_zoom < 14 else current_zoom # Zoom in, but don't go too far


# --- Data Fetching Functions (Existing + New Weather/Currency) ---

# --- NEW Weather Function ---
@st.cache_data(ttl=3600*1) # Cache weather for 1 hour
def fetch_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetches current weather data from OpenWeatherMap using data/2.5/weather endpoint."""
    if not WEATHER_API_KEY:
        print("fetch_weather: API key missing.")
        return {"error": "Weather API key not configured."}
    if lat is None or lon is None:
        print("fetch_weather: Invalid coordinates.")
        return {"error": "Invalid coordinates for weather."}

    # Use the standard current weather endpoint
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY,
        'units': 'metric' # Use Celsius and m/s
    }
    print(f"Fetching weather for ({lat:.4f}, {lon:.4f})")

    try:
        response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # --- Extract relevant info from data/2.5/weather response structure ---
        if response.status_code == 200 and data:
            weather_main = data.get("weather", [{}])[0] # Weather is a list
            main_data = data.get("main", {})
            wind_data = data.get("wind", {})
            sys_data = data.get("sys", {})

            weather_info = {
                "description": weather_main.get("description", "N/A").capitalize(),
                "temperature": main_data.get("temp"),
                "feels_like": main_data.get("feels_like"),
                "humidity": main_data.get("humidity"),
                "pressure": main_data.get("pressure"), # Added pressure
                "wind_speed": wind_data.get("speed"), # meter/sec
                "wind_deg": wind_data.get("deg"),     # Added wind direction
                "city_name": data.get("name"),          # City name from API response
                "country": sys_data.get("country"),   # Country code from API response
                "icon": weather_main.get("icon"),       # Weather icon code
                "timestamp": data.get("dt"),            # Observation time (Unix UTC)
                "timezone": data.get("timezone")        # Shift in seconds from UTC
            }
            print(f"Weather fetch success: {weather_info.get('description')} at {weather_info.get('city_name')}")
            return weather_info
        else:
             print(f"Weather API returned status {response.status_code} or empty data.")
             return {"error": f"Weather data not available (Status: {response.status_code})"}

    except RequestException as e:
        st.error(f"Weather API request failed: {e}")
        print(f"Weather API request exception: {e}")
        return {"error": f"Could not connect to weather service."}
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse weather API response: {e}")
        print(f"Weather API JSON decode error: {e}")
        return {"error": "Invalid weather data received."}
    except Exception as e:
        st.error(f"Error processing weather data: {type(e).__name__}")
        traceback.print_exc()
        return {"error": f"Unexpected error processing weather data."}


# --- Currency Function (Unchanged, seems okay) ---
@st.cache_data(ttl=3600*6) # Cache currency for 6 hours
def fetch_currency_info(home_currency: str, dest_country_code_iso2: str) -> Optional[Dict[str, Any]]:
    """Fetches local currency code, symbol and conversion rate."""
    if not CURRENCY_API_KEY:
        return {"error": "Currency API key not configured."}
    if not home_currency:
        return {"error": "Home currency not selected."}
    if not dest_country_code_iso2:
        return {"error": "Destination country code missing."}

    print(f"Fetching currency: {home_currency} to {dest_country_code_iso2}")

    # 1. Get local currency code using pycountry
    local_currency_code = None
    try:
        country = pycountry.countries.get(alpha_2=dest_country_code_iso2)
        if country and hasattr(country, 'numeric'):
            # Find currency by country's numeric code (often more reliable)
            currencies = pycountry.currencies.get(numeric=country.numeric)
            if currencies:
                 local_currency_code = currencies.alpha_3
                 print(f"Found local currency code via pycountry: {local_currency_code}")

    except Exception as e:
        print(f"Could not get currency code via pycountry for {dest_country_code_iso2}: {e}")
        # Continue, will rely solely on API or fail later if code not found

    if not local_currency_code:
         print(f"Could not determine local currency code for {dest_country_code_iso2} via pycountry.")
         # Attempting API without local code is unlikely to work with ExchangeRate-API's basic endpoint structure
         # It needs the target currency code.
         # We could try guessing based on common currencies or use a different API if this fails often.
         st.warning(f"Could not reliably determine local currency for {dest_country_code_iso2}. Conversion may fail.")
         return {"error": f"Could not determine local currency for {pycountry.countries.get(alpha_2=dest_country_code_iso2).name}."}


    # 2. Fetch conversion rate from API (Example: ExchangeRate-API)
    url = f"{CURRENCY_API_ENDPOINT}{home_currency}" # Gets rates relative to home_currency
    print(f"Requesting currency URL: {url}")
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("result") == "success":
            rates = data.get("conversion_rates", {})
            conversion_rate = rates.get(local_currency_code) # Look for the specific local currency code

            if conversion_rate:
                 print(f"Found conversion rate: 1 {home_currency} = {conversion_rate} {local_currency_code}")
                 return {
                     "home_currency": home_currency,
                     "local_currency_code": local_currency_code,
                     # Symbol lookup is unreliable, just use the code
                     "local_currency_symbol": local_currency_code,
                     "rate": conversion_rate, # 1 Home Currency = X Local Currency
                     "last_update_utc": data.get("time_last_update_utc"),
                     "next_update_utc": data.get("time_next_update_utc")
                 }
            else:
                 print(f"Rate for target currency {local_currency_code} not found in API response.")
                 st.warning(f"Conversion rate from {home_currency} to {local_currency_code} not available from API.")
                 return {"error": f"Rate for {local_currency_code} not found in API response."}
        else:
            # Handle API-specific errors
            error_type = data.get("error-type", "Unknown API Error")
            st.error(f"Currency API Error: {error_type}")
            print(f"Currency API returned error: {error_type}")
            return {"error": f"Currency API error: {error_type}"}

    except RequestException as e:
        st.error(f"Currency API request failed: {e}")
        print(f"Currency API request exception: {e}")
        return {"error": f"Could not connect to currency service."}
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse currency API response: {e}")
        print(f"Currency API JSON decode error: {e}")
        return {"error": "Invalid currency data received."}
    except Exception as e:
        st.error(f"Error processing currency data: {type(e).__name__}")
        traceback.print_exc()
        return {"error": f"Unexpected error processing currency data."}


# --- Safety News Fetching (Unchanged) ---
def fetch_safety_news(country: str) -> Union[List[Dict[str, str]], str]:
    if not country: return "Error: Country name is required for safety news search."
    query = f'"{country}" safety conflict unrest kidnapping protest security alert news'
    st.write(f"Searching news for: {country} (using DuckDuckGo)...")
    news_items = []
    max_results = 7 # Limit news items displayed
    print(f"Fetching safety news with query: {query}")
    try:
        with DDGS() as ddgs:
            # Using region='wt-wt' for worldwide results
            results = list(ddgs.news(keywords=query, region='wt-wt', safesearch='off', max_results=max_results))

        if not results:
            print(f"No specific safety news found via DuckDuckGo News for {country}.")
            return [] # Return empty list, not an error string initially

        for r in results:
            title = r.get('title', 'No Title Provided')
            url = r.get('url', '')
            source = r.get('source', 'Unknown Source')
            date_str = r.get('date', '') # DDG provides date string
            snippet = r.get('body', '') # 'body' usually contains the snippet

            # Basic filtering for relevance (optional)
            # if "safety" not in title.lower() and "alert" not in title.lower() and "unrest" not in title.lower():
            #    continue

            if url: # Only include items with a URL
                 news_item = {
                     'title': title,
                     'url': url,
                     'snippet': snippet[:250] + ('...' if len(snippet) > 250 else '') if snippet else 'No snippet available.',
                     'source': source,
                     'date': date_str if date_str else '' # Keep original date string format
                 }
                 news_items.append(news_item)
        print(f"Found {len(news_items)} relevant news items for {country}.")
        if not news_items:
             print(f"Filtered out all initial DDG results for {country}. None seemed relevant.")
             return [] # Return empty list if filtering removed everything


    except Exception as e:
        st.error(f"DuckDuckGo News search failed for safety news: {type(e).__name__} - {e}")
        traceback.print_exc()
        return f"Error: Could not perform news search using DuckDuckGo. Details: {e}" # Return error string

    return news_items

# --- Travel Advisory Fetching (Unchanged) ---
def fetch_travel_advisory(country_code: str) -> str:
    if not country_code: return "Error: Country code is required for advisory search."
    try:
        country = pycountry.countries.get(alpha_2=country_code).name
    except Exception as e:
        st.error(f"Error looking up country code {country_code}: {e}")
        return "Error: Could not find country name from code."

    # Refined query targeting official government domains
    query = f'"{country}" official government travel advisory OR travel advice site:.gov OR site:.gc.ca OR site:.gov.uk OR site:.govt.nz OR site:.gov.au OR site:europa.eu/consular_protection'
    st.write(f"Searching official advisories for: {country} (using DuckDuckGo)...")
    print(f"Fetching advisories with query: {query}")
    urls_to_scrape = []
    try:
        with DDGS() as ddgs:
            # Fetch slightly more results initially to increase chances of finding good ones
            results = list(ddgs.text(keywords=query, region='wt-wt', safesearch='off', max_results=15))

        if not results:
            print(f"No potential advisory sites found via DuckDuckGo for {country}.")
            return f"No potential government advisory sites found via search for {country}. Check official government websites directly."

        urls_seen = set()
        count = 0
        # Filter results more carefully for likely official sources
        official_domains = ['.gov', '.gc.ca', 'govt', 'europa.eu', 'state.gov', '.gov.uk', '.govt.nz', '.gov.au', 'foreignaffairs', 'international.gc.ca']
        for r in results:
            url = r.get('href')
            if url and url not in urls_seen:
                # Check if the URL contains any of the official domain keywords
                if any(kw in url for kw in official_domains):
                    # Optional: Add more filtering, e.g., avoid PDF links directly if scraping struggles
                    # if '.pdf' in url: continue
                    urls_seen.add(url)
                    urls_to_scrape.append(url)
                    count += 1
                    print(f"  Added potential advisory URL: {url}")
                    if count >= 3: # Limit to scraping top 3 relevant URLs
                        break # Stop after finding enough potential URLs

    except Exception as e:
        st.error(f"DDG search failed for advisories: {e}")
        traceback.print_exc()
        return "Error: Could not perform advisory search using DuckDuckGo."

    if not urls_to_scrape:
        print(f"Found DDG results, but none matched the filtered official domains for {country}.")
        return f"Found search results, but could not identify likely official advisory URLs for {country}. Please check your government's official travel advice website."

    advisories_raw_text = []
    st.write(f"Scraping content from up to {len(urls_to_scrape)} potential sources...")
    # (Scraping loop - unchanged logic, added print for which URL is being scraped)
    for url in urls_to_scrape:
        print(f"  Attempting to scrape: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'} # Be polite, identify as a bot
            # Increased timeout slightly for potentially slower gov sites
            r = requests.get(url, timeout=REQUEST_TIMEOUT + 5, headers=headers, allow_redirects=True)
            r.raise_for_status() # Check for HTTP errors

            # Check content type - avoid trying to parse images, PDFs etc.
            content_type = r.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                print(f"  Skipping non-HTML content ({content_type}): {url}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # Try common tags for main content, falling back to body
            main_content = soup.find('main') or soup.find('article') or soup.find('div', id='content') or soup.find('div', class_='content') or soup.body

            text_parts = []
            if main_content:
                # Extract text more robustly, joining paragraphs
                # Consider adding other tags like 'li' if lists are important
                paragraphs = main_content.find_all("p", limit=20) # Limit paragraphs to avoid excessive length
                for p in paragraphs:
                    p_text = p.get_text(" ", strip=True)
                    # Filter out short/boilerplate paragraphs
                    if len(p_text) > 100 and "cookie policy" not in p_text.lower() and "privacy policy" not in p_text.lower():
                        text_parts.append(p_text)

            if text_parts:
                full_text = "\n".join(text_parts)
                # Limit total text length per source sent to LLM
                advisories_raw_text.append(f"Source: {url}\n" + full_text[:5000]) # Limit to 5000 chars per source
                print(f"  Successfully scraped and added text from: {url}")
            else:
                 print(f"  Could not extract sufficient text content from: {url}")

        except requests.exceptions.Timeout:
             print(f"  Timeout scraping advisory: {url}")
        except requests.exceptions.RequestException as e:
             print(f"  Error scraping advisory {url}: {e}")
        except Exception as e:
             print(f"  Unexpected error processing {url}: {e}")
        time.sleep(0.75) # Be slightly more polite between requests

    if not advisories_raw_text:
        print(f"Found potential URLs for {country}, but couldn't scrape useful content.")
        return f"Found potential advisory URLs for {country}, but couldn't scrape readable content. Please check official sites directly."

    # Combine scraped text
    raw_blob = "\n\n---\n\n".join(advisories_raw_text)
    # Limit total blob size before sending to LLM
    max_llm_input_length = 15000
    raw_blob_truncated = raw_blob[:max_llm_input_length]
    if len(raw_blob) > max_llm_input_length:
        print(f"Truncated advisory text from {len(raw_blob)} to {max_llm_input_length} chars for LLM.")

    # (LLM Prompt - unchanged, but ensure `generate_llm_response` is available)
    prompt = (f"Based *only* on the scraped text below from potential government travel advisory pages regarding **{country}**, synthesize a concise summary for a tourist. Use Markdown format with these headings: **Overall Risk Level** (if mentioned), **Key Safety & Security Concerns**, **Entry & Exit Requirements** (if mentioned), **Local Laws & Customs Highlights** (if mentioned), **Health Recommendations** (if mentioned), and **Emergency Contact Info** (if mentioned). If information for a heading is missing in the text, state 'Not specified in provided text'. Prioritize official-sounding warnings and advice. Be factual and neutral. Mention source URLs if helpful for context, but focus on the advice itself.\n\n--- Scraped Text Start ---\n{raw_blob_truncated}\n--- Scraped Text End ---\n\n**Travel Advisory Summary for {country}:**")
    st.write("Summarizing advisory info using AI...")
    print("Sending advisory text to LLM for summarization.")
    try:
        summary = generate_llm_response(prompt)
        print("LLM advisory summary received.")
        return summary
    except NameError:
        st.error("LLM function 'generate_llm_response' is not available. Cannot generate summary.")
        return "Error: LLM service not configured, cannot generate advisory summary."
    except Exception as e:
        st.error(f"LLM advisory summary failed: {e}")
        traceback.print_exc()
        return f"Error generating advisory summary using AI. Raw text might be available but unsummarized."


# --- Visa Requirements Fetching (Unchanged) ---
def fetch_visa_requirements(nationality: str, dest_country_code: str) -> str:
    if not nationality: return "Error: Nationality required for visa search."
    if not dest_country_code: return "Error: Destination country code required."
    try:
        dest_country = pycountry.countries.get(alpha_2=dest_country_code).name
    except Exception as e:
        st.error(f"Error looking up dest country code {dest_country_code}: {e}")
        return "Error: Could not find destination country name from code."

    # More specific query targeting embassy/consulate/official immigration sites
    query = f'"{dest_country}" visa requirements for "{nationality}" citizens official source OR embassy OR consulate OR immigration OR "ministry of foreign affairs"'
    st.write(f"Searching visa info for {nationality} citizen to {dest_country} (using DuckDuckGo)...")
    print(f"Fetching visa info with query: {query}")
    urls_to_scrape = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, region='wt-wt', safesearch='off', max_results=15))

        if not results:
            print(f"No potential visa info sites found via DDG for {nationality} to {dest_country}.")
            return f"No potential visa information sites found via search for {nationality} citizens traveling to {dest_country}. Check the official embassy/consulate website of {dest_country} in your country, or your country's foreign affairs ministry."

        urls_seen = set()
        count = 0
        # Filter for likely official sources (embassy, consulate, gov, mfa, etc.)
        visa_keywords = ['embassy', 'consulate', '.gov', 'govt', 'immigration', 'visa', 'foreignaffairs', 'mfa', 'government', 'border', 'visit']
        for r in results:
            url = r.get('href')
            title = r.get('title', '').lower()
            snippet = r.get('body', '').lower()
            # Check URL, Title, and Snippet for relevance
            if url and url not in urls_seen:
                # Prioritize URLs containing keywords
                if any(kw in url for kw in visa_keywords):
                     # Extra check: Does title or snippet mention visa or the nationality?
                    if 'visa' in title or 'visa' in snippet or nationality.lower() in title or nationality.lower() in snippet:
                        urls_seen.add(url)
                        urls_to_scrape.append(url)
                        count += 1
                        print(f"  Added potential visa URL: {url}")
                        if count >= 3: # Limit scraping
                            break

    except Exception as e:
        st.error(f"DDG search failed for visa info: {e}")
        traceback.print_exc()
        return "Error: Could not perform visa search using DuckDuckGo."

    if not urls_to_scrape:
        print(f"Found DDG results, but none seemed like relevant official visa info URLs for {nationality} to {dest_country}.")
        return f"Found search results, but could not identify likely official visa information URLs. Please consult the official embassy or consulate website of {dest_country} for {nationality} citizens."

    visa_info_snippets = []
    st.write(f"Scraping content from up to {len(urls_to_scrape)} potential visa sources...")
     # (Scraping loop - unchanged logic, added print)
    for url in urls_to_scrape:
        print(f"  Attempting to scrape visa info: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
            r = requests.get(url, timeout=REQUEST_TIMEOUT + 5, headers=headers, allow_redirects=True)
            r.raise_for_status()

            content_type = r.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                print(f"  Skipping non-HTML content ({content_type}): {url}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            main_content = soup.find('main') or soup.find('article') or soup.find('div', id='content') or soup.find('div', class_='content') or soup.body

            text_parts = []
            if main_content:
                 # Look for paragraphs containing relevant keywords
                paragraphs = main_content.find_all("p", limit=25) # Limit paragraphs
                for p in paragraphs:
                    p_text = p.get_text(" ", strip=True)
                    p_text_lower = p_text.lower()
                    # Filter for longer paragraphs potentially containing visa details
                    if len(p_text) > 75 and ('visa' in p_text_lower or nationality.lower() in p_text_lower or 'passport' in p_text_lower or 'apply' in p_text_lower or 'fee' in p_text_lower or 'document' in p_text_lower):
                         # Avoid common boilerplate
                        if "cookie policy" not in p_text_lower and "privacy policy" not in p_text_lower and "terms of service" not in p_text_lower:
                            text_parts.append(p_text)

            if text_parts:
                full_text = "\n".join(text_parts)
                visa_info_snippets.append(f"Source: {url}\n" + full_text[:5000]) # Limit text per source
                print(f"  Successfully scraped and added visa text from: {url}")
            else:
                 print(f"  Could not extract sufficient relevant visa text from: {url}")

        except requests.exceptions.Timeout:
             print(f"  Timeout scraping visa info: {url}")
        except requests.exceptions.RequestException as e:
             print(f"  Error scraping visa info {url}: {e}")
        except Exception as e:
             print(f"  Unexpected error processing {url}: {e}")
        time.sleep(0.75) # Be polite

    if not visa_info_snippets:
        print(f"Found potential visa URLs, but couldn't scrape useful content.")
        return f"Found potential visa URLs, but couldn't scrape specific requirements. Please check official embassy/consulate sites directly for {nationality} citizens traveling to {dest_country}."

    raw_blob = "\n\n---\n\n".join(visa_info_snippets)
    # Limit total blob size before sending to LLM
    max_llm_input_length = 15000
    raw_blob_truncated = raw_blob[:max_llm_input_length]
    if len(raw_blob) > max_llm_input_length:
        print(f"Truncated visa text from {len(raw_blob)} to {max_llm_input_length} chars for LLM.")

    # (LLM Prompt - unchanged, ensure `generate_llm_response` is available)
    prompt = (f"Based *only* on the scraped text below regarding visa requirements for **{nationality}** passport holders traveling to **{dest_country}** (likely for tourism/short stay unless specified otherwise), synthesize a step-by-step guide. Use Markdown format. Prioritize official-sounding information found in the text. Explicitly state if information is unclear, conflicting, or not found in the provided text. Use these headings exactly: \n1. **Visa Requirement:** (e.g., Visa required, Visa-free for X days, eVisa available, etc.)\n2. **Application Process:** (Where and how to apply, based on text? Embassy, online portal?)\n3. **Required Documents:** (List documents mentioned in the text)\n4. **Processing Time & Fees:** (Mention any details found in the text)\n5. **Important Notes:** (Any other relevant points from the text, e.g., validity, specific conditions).\nIf the text is contradictory or insufficient for a section, state that clearly.\n\n--- Scraped Text Start ---\n{raw_blob_truncated}\n--- Scraped Text End ---\n\n**Visa Guide for {nationality} Citizen to {dest_country} (Based on Scraped Info):**")
    st.write("Synthesizing visa guide using AI...")
    print("Sending visa text to LLM for synthesis.")
    try:
        guide = generate_llm_response(prompt)
        print("LLM visa guide received.")
        # Append a disclaimer about verification
        guide += "\n\n**Disclaimer:** This guide is generated based on potentially incomplete scraped text. **You MUST verify all visa requirements with the official embassy or consulate of {dest_country} well in advance of your travel.** Rules can change frequently."
        return guide
    except NameError:
        st.error("LLM function 'generate_llm_response' is not available. Cannot generate guide.")
        return "Error: LLM service not configured, cannot generate visa guide."
    except Exception as e:
        st.error(f"LLM visa guide failed: {e}")
        traceback.print_exc()
        return f"Error generating visa guide using AI. Raw text might be available but unsynthesized."


# --- LLM-Based Feature Functions (Unchanged logic, added print statements) ---

def fetch_emergency_numbers_llm(country: str) -> str:
    """Uses LLM to get common emergency numbers. Needs strong verification disclaimer."""
    if not country: return "Error: Country name required."
    st.write(f"Asking AI for emergency numbers in {country}...")
    print(f"Requesting emergency numbers for {country} from LLM.")
    # More explicit prompt for common numbers
    prompt = (
        f"List the *primary, most common* emergency telephone numbers for {country}. "
        "Specifically, provide the numbers typically used nationwide for:\n"
        "- Police\n"
        "- Ambulance / Medical Emergency\n"
        "- Fire Department\n\n"
        "Format clearly, like: 'Police: XXX, Ambulance: YYY, Fire: ZZZ'. "
        "If a single number (like 112 or 911) covers all or multiple services, state that clearly (e.g., 'General Emergency (Police, Ambulance, Fire): 112'). "
        "If different numbers are common, list them separately. "
        "Do NOT include non-emergency numbers, tourist police (unless it's the primary contact), or regional variations unless they are critical. "
        "Provide only the numbers and their service, no extra conversation or explanations."
    )
    try:
        response = generate_llm_response(prompt)
        print(f"LLM response for emergency numbers received.")
        # Add a mandatory disclaimer
        response += "\n\n**⚠️ Verify these numbers!** This information is AI-generated and may not be accurate or complete. Always confirm emergency numbers with official local sources upon arrival."
        return response
    except NameError:
        st.error("LLM function 'generate_llm_response' is not available.")
        return "Error: LLM service not configured."
    except Exception as e:
        st.error(f"LLM failed for emergency numbers: {e}")
        traceback.print_exc()
        return "Error: Could not retrieve emergency numbers via AI."

def fetch_cultural_tips_llm(country: str) -> str:
    """Uses LLM to get brief cultural tips."""
    if not country: return "Error: Country name required."
    st.write(f"Asking AI for cultural tips for {country}...")
    print(f"Requesting cultural tips for {country} from LLM.")
    prompt = (
        f"Provide 3-5 brief, practical, and essential cultural etiquette tips for a first-time tourist visiting {country}. "
        "Focus on potential major differences or faux pas related to:\n"
        "- **Greetings:** (e.g., handshake, bow, formality)\n"
        "- **Dining:** (e.g., using hands/utensils, table manners, sharing food)\n"
        "- **Tipping:** (Is it expected? How much? Where?)\n"
        "- **Public Behavior/Dress:** (e.g., appropriate clothing for public spaces or religious sites, public displays of affection)\n"
        "- **Communication:** (e.g., directness vs indirectness, sensitive topics, important gestures to use or avoid)\n"
        "Keep each tip concise (1-2 sentences). Use Markdown bullet points. Start directly with the tips."
        "Example format:\n* **Greetings:** A firm handshake is standard for introductions.\n* **Tipping:** Tipping 10-15% is customary in restaurants..."
    )
    try:
        response = generate_llm_response(prompt)
        print(f"LLM response for cultural tips received.")
         # Add a reminder
        response += "\n\n*Note: These are general tips. Local customs can vary. Being observant and respectful is always key.*"
        return response
    except NameError:
        st.error("LLM function 'generate_llm_response' is not available.")
        return "Error: LLM service not configured."
    except Exception as e:
        st.error(f"LLM failed for cultural tips: {e}")
        traceback.print_exc()
        return "Error: Could not retrieve cultural tips via AI."

def fetch_health_info_llm(country: str) -> str:
    """Uses LLM to get GENERAL health info summary. Needs strong disclaimer."""
    if not country: return "Error: Country name required."
    st.write(f"Asking AI for general health info summary for {country}...")
    print(f"Requesting health info for {country} from LLM.")
    # Prompt emphasizing generality and need for professional advice
    prompt = (
        f"Provide a *brief, general* overview of potential health considerations for a typical tourist planning a trip to {country}. Structure the response using these Markdown headings:\n"
        "1.  **Routine Vaccinations:** Briefly state that standard vaccinations (like MMR, Tdap, Polio, Flu) should generally be up-to-date.\n"
        "2.  **Common Travel Vaccinations/Medications:** Mention 1-3 *examples* of vaccines or preventative medications *commonly considered* for travel to this region (e.g., Hepatitis A, Typhoid, Malaria prophylaxis, Yellow Fever *if applicable and widely known*). Acknowledge that requirements vary.\n"
        "3.  **Key Health Risks:** Briefly mention 2-4 significant health risks travelers might encounter (e.g., food and waterborne illness, vector-borne diseases like Malaria/Dengue, altitude sickness, sun exposure, specific local infectious diseases if prominent). Keep descriptions very brief.\n"
        "4.  **Water & Food Safety:** Add a short general reminder about being cautious with tap water and street food.\n\n"
        "**Crucially, conclude with a prominent disclaimer:** State very clearly that this is **general information only**, not medical advice, and that travelers **MUST consult a doctor, travel clinic, or refer to official health organizations (like CDC, WHO, or their national health body) at least 4-6 weeks before departure** for personalized advice based on their specific health status, itinerary, and activities.\n\n"
        "Keep the entire response concise (around 150-200 words)."
    )
    try:
        response = generate_llm_response(prompt)
        print(f"LLM response for health info received.")
        # Ensure the disclaimer is prominent and clear (though it's requested in the prompt)
        # We can add it again just to be safe if needed, or check if it exists.
        disclaimer_keywords = ["consult a doctor", "travel clinic", "cdc", "who", "official health", "personalized advice", "not medical advice"]
        response_lower = response.lower()
        if not any(keyword in response_lower for keyword in disclaimer_keywords):
             print("LLM health response missing key disclaimer elements. Appending mandatory disclaimer.")
             response += "\n\n**VERY IMPORTANT DISCLAIMER:** This is general information only and NOT medical advice. Health risks and recommendations change. You **MUST** consult your doctor or a specialized travel health clinic (referencing official sources like CDC, WHO, or your national health authority) well before your trip for personalized advice tailored to your health history and travel plans."
        return response
    except NameError:
        st.error("LLM function 'generate_llm_response' is not available.")
        return "Error: LLM service not configured."
    except Exception as e:
        st.error(f"LLM failed for health info: {e}")
        traceback.print_exc()
        return "Error: Could not retrieve health info via AI."

def generate_packing_list_llm(destination: str, country: str, duration_days: int, trip_type: str, weather_summary: str) -> str:
    """Uses LLM to generate a suggested packing list."""
    if not destination or not country: return "Error: Destination details required."
    st.write(f"Asking AI to generate packing list for {destination}...")
    print(f"Requesting packing list for {destination} ({duration_days} days, {trip_type}, Weather: {weather_summary}) from LLM.")
    # Detailed prompt for a structured packing list
    prompt = (
        f"Generate a suggested packing list in Markdown checklist format (`- [ ] Item Name`) for a trip with these details:\n"
        f"- **Destination:** {destination}, {country}\n"
        f"- **Duration:** {duration_days} days\n"
        f"- **Primary Trip Type:** {trip_type}\n"
        f"- **Brief Weather Outlook:** {weather_summary}\n\n"
        "Organize the list into logical categories (e.g., **Documents & Essentials**, **Clothing**, **Toiletries**, **Medications & First Aid**, **Electronics**, **Miscellaneous**). "
        "Include standard essentials like passport, visa (mention 'if required'), travel insurance details, flight/hotel confirmations, credit/debit cards, and some local currency. "
        "Suggest clothing items appropriate for the duration, weather, and trip type (e.g., swimwear for beach, rain jacket for rain, comfortable walking shoes, maybe one smarter outfit). Add quantity suggestions where appropriate (e.g., 'T-shirts (x{duration_days//2 + 1})'). "
        "Include basic toiletries, personal medications plus a small first-aid kit, necessary chargers and adapters. "
        "Add items relevant to the trip type (e.g., specific gear for 'Adventure', laptop/notebook for 'Business'). "
        "Do not add any introductory or concluding sentences, just the categorized Markdown checklist."
    )
    try:
        response = generate_llm_response(prompt)
        print(f"LLM response for packing list received.")
        # Basic check for list format
        if not response.strip().startswith("- [ ]") and not response.strip().startswith("*"):
             print("LLM packing list response not in expected format. Returning basic placeholder.")
             return """
- [ ] Passport
- [ ] Visa (if required)
- [ ] Flight/Hotel confirmations
- [ ] Travel Insurance details
- [ ] Credit/Debit Cards & some Cash
- [ ] (AI failed to generate detailed list - Check essentials based on your trip)
"""
        # Add a reminder note
        response += "\n\n*Note: This is a general suggestion. Adjust based on your personal needs, specific activities, and baggage limits.*"
        return response
    except NameError:
        st.error("LLM function 'generate_llm_response' is not available.")
        return "Error: LLM service not configured."
    except Exception as e:
        st.error(f"LLM failed for packing list: {e}")
        traceback.print_exc()
        return "Error: Could not generate packing list via AI."


# --- Streamlit UI ---
st.set_page_config(page_title="SafeTravelAI", layout="wide", initial_sidebar_state="expanded")
st.title("🌎 SafeTravelAI: Your Enhanced AI Travel Advisor")

# --- Inputs moved to Sidebar ---
st.sidebar.header("🌍 Trip Details")

# --- Destination Input ---
st.sidebar.subheader("📍 Destination")
# Text input - its value is controlled by place_input_key state
# It gets updated by map clicks via update_location_from_map
# Manual changes trigger update_location_from_text via the logic below
st.session_state.place_input_key = st.sidebar.text_input(
    "Enter destination OR click map below",
    value=st.session_state.get("place_input_key", ""), # Set value from state
    key="place_input_display_key", # Use a different key for the widget itself if needed, but place_input_key should work
    placeholder="e.g., Paris, France or click map"
)

# --- Logic to handle manual text input changes ---
# Compare the current widget value with the last known *processed* value (place_name_prev)
# This runs on every script rerun
if st.session_state.place_input_key != st.session_state.get("place_name_prev", ""):
    # If the text input value has changed since the last update (either text or map)
    # Trigger the text update logic
    update_location_from_text()
    # We don't rerun here, state changes will trigger it if needed


st.sidebar.subheader("👤 Your Identity")
# Nationality
# Using a predefined list, consider making this searchable if needed
common_nationalities = sorted([
    "United States", "India", "United Kingdom", "Canada", "Australia",
    "Germany", "France", "China", "Brazil", "South Africa", "Nigeria", "Japan",
    "Mexico", "Philippines", "Pakistan", "Bangladesh", "Russia", "Indonesia",
    "Egypt", "Vietnam", "Turkey", "Iran", "Thailand", "Italy", "Spain", "Colombia", "Argentina"
])
# Attempt to find a default index, fallback to 0
default_nationality = "India"
try: default_nationality_index = common_nationalities.index(default_nationality)
except ValueError: default_nationality_index = 0

nationality = st.sidebar.selectbox(
    "Your Nationality (Passport)",
    common_nationalities,
    index=st.session_state.get("nationality_index", default_nationality_index), # Use state to preserve selection
    key="nationality_key",
    help="Select the nationality of the passport you will be traveling with."
)
# Store selected index in session state to preserve selection across reruns
st.session_state.nationality_index = common_nationalities.index(nationality)


# Home Currency (Using ISO Codes)
currency_options = {
    "USD": "US Dollar", "EUR": "Euro", "JPY": "Japanese Yen", "GBP": "British Pound",
    "AUD": "Australian Dollar", "CAD": "Canadian Dollar", "CHF": "Swiss Franc",
    "CNY": "Chinese Yuan Renminbi", "INR": "Indian Rupee", "BRL": "Brazilian Real",
    "RUB": "Russian Ruble", "ZAR": "South African Rand", "NGN": "Nigerian Naira",
    "MXN": "Mexican Peso", "PKR": "Pakistani Rupee", "BDT": "Bangladeshi Taka",
    "PHP": "Philippine Peso", "IDR": "Indonesian Rupiah", "EGP": "Egyptian Pound",
    "VND": "Vietnamese Dong", "TRY": "Turkish Lira", "THB": "Thai Baht",
    "ITL": "Italian Lira (Use EUR)", "ESP": "Spanish Peseta (Use EUR)", # Add note for Eurozone
    "COP": "Colombian Peso", "ARS": "Argentine Peso"
    # Add more as needed
}
sorted_currency_codes = sorted(currency_options.keys())

# Attempt to find default index, fallback to 0
default_currency = "INR"
try: default_currency_index = sorted_currency_codes.index(default_currency)
except ValueError: default_currency_index = 0

home_currency = st.sidebar.selectbox(
    "Your Home Currency",
    options=sorted_currency_codes,
    format_func=lambda x: f"{x} ({currency_options.get(x, 'Unknown')})",
    index=st.session_state.get("home_currency_index", default_currency_index), # Use state
    key="home_currency_key",
    help="Select your primary home currency for conversion rates."
)
# Store selected index
st.session_state.home_currency_index = sorted_currency_codes.index(home_currency)


st.sidebar.subheader("📅 Trip Planning")
# Trip Duration
duration = st.sidebar.slider(
    "Trip Duration (Days)",
    min_value=1,
    max_value=90, # Increased max duration
    value=st.session_state.get("duration_days", 7), # Use state
    key="duration_key",
    help="Estimate the total duration of your trip in days."
)
st.session_state.duration_days = duration # Store value

# Trip Type
trip_types = [
    "Leisure / Vacation", "Business", "Adventure / Outdoors", "Beach / Resort",
    "Cultural / Sightseeing", "Visiting Friends/Family", "Backpacking / Budget Travel", "Other"
]
trip_type = st.sidebar.selectbox(
    "Primary Trip Type",
    options=trip_types,
    index=st.session_state.get("trip_type_index", 0), # Use state
    key="trip_type_key",
    help="Select the main purpose of your trip to tailor suggestions."
)
st.session_state.trip_type_index = trip_types.index(trip_type) # Store index


# --- Action Button (in sidebar) ---
if st.sidebar.button("🔍 Get Full Travel Briefing", type="primary", use_container_width=True):

    # --- Get current state values at the time of button click ---
    print("\n--- Get Full Travel Briefing Button Clicked ---")
    current_place = st.session_state.get("place_name", "")
    target_country = st.session_state.get("last_known_country", None)
    target_country_code = st.session_state.get("last_known_country_code", None)
    current_coords = st.session_state.get("coords", None)
    current_nationality = st.session_state.get("nationality_key", None)
    current_home_currency = st.session_state.get("home_currency_key", None)
    current_duration = st.session_state.get("duration_key", 7)
    current_trip_type = st.session_state.get("trip_type_key", "Leisure / Vacation")

    print(f"  State Values Read:")
    print(f"    Place Name: {current_place}")
    print(f"    Target Country: {target_country}")
    print(f"    Target Country Code: {target_country_code}")
    print(f"    Coordinates: {current_coords}")
    print(f"    Nationality: {current_nationality}")
    print(f"    Home Currency: {current_home_currency}")
    print(f"    Duration: {current_duration}")
    print(f"    Trip Type: {current_trip_type}")


    # --- Input Validation ---
    valid_request = True
    if not current_place or current_place.startswith("Lat:"): # Check if place name is valid
        st.warning("⚠️ Please select a valid destination by clicking the map or entering a place name (e.g., 'City, Country').")
        valid_request = False
    if not current_nationality:
        st.warning("⚠️ Please select your nationality.")
        valid_request = False
    if not target_country or not target_country_code:
        # Try one last time to extract from place name if country state is missing
        target_country = get_country_from_place(current_place)
        if target_country:
            target_country_code = get_country_code(target_country)
            st.session_state.last_known_country = target_country
            st.session_state.last_known_country_code = target_country_code
            print(f"  Inferred country/code just before API calls: {target_country} / {target_country_code}")
        else:
            st.warning(f"⚠️ Could not determine the country from '{current_place}'. Please use 'City, Country' format or click the map accurately.")
            valid_request = False
    if not current_home_currency:
        st.warning("⚠️ Please select your home currency.")
        valid_request = False
    # Check for valid coordinates if weather is needed
    if not current_coords or current_coords == DEFAULT_MAP_LOCATION:
        st.info("ℹ️ Map location not set. Weather information might be unavailable or for default location.")
        # Allow proceeding, but weather will likely fail or be wrong.

    # --- Proceed only if validation passes ---
    if valid_request:
        st.markdown(f"### Travel Briefing for: {current_place}")
        st.markdown("---")
        # Use a single spinner for the whole process
        with st.spinner(f"🕵️‍♀️ Gathering comprehensive insights for {target_country}... This may take a minute..."):
            results = {} # Dictionary to store results

            # --- Fetch Weather ---
            weather_data = None
            weather_summary_for_llm = "Weather data not available."
            if WEATHER_API_KEY and current_coords and current_coords != DEFAULT_MAP_LOCATION:
                print("--- Fetching Weather ---")
                weather_data = fetch_weather(current_coords[0], current_coords[1])
                results["weather"] = weather_data # Store result/error
                if weather_data and not weather_data.get("error"):
                    temp = weather_data.get('temperature', 'N/A')
                    desc = weather_data.get('description', 'N/A')
                    weather_summary_for_llm = f"Currently {desc} with temperatures around {temp}°C."
                    print(f"  Weather Summary: {weather_summary_for_llm}")
                elif weather_data and weather_data.get("error"):
                     weather_summary_for_llm = f"Could not fetch weather ({weather_data['error']})."
                     print(f"  Weather Fetch Error: {weather_data['error']}")
            elif not WEATHER_API_KEY:
                 print("--- Skipping Weather (No API Key) ---")
                 results["weather"] = {"error": "Weather API key not configured."}
            else:
                 print("--- Skipping Weather (Invalid Coords) ---")
                 results["weather"] = {"error": "Valid coordinates not available."}


            # --- Fetch Currency ---
            currency_data = None
            if CURRENCY_API_KEY and current_home_currency and target_country_code:
                print("--- Fetching Currency ---")
                currency_data = fetch_currency_info(current_home_currency, target_country_code)
                results["currency"] = currency_data # Store result/error
                if currency_data and currency_data.get("error"):
                    print(f"  Currency Fetch Error: {currency_data['error']}")
            elif not CURRENCY_API_KEY:
                print("--- Skipping Currency (No API Key) ---")
                results["currency"] = {"error": "Currency API key not configured."}
            else:
                print("--- Skipping Currency (Missing Inputs) ---")
                results["currency"] = {"error": "Home currency or destination country code missing."}


            # --- Fetch Other Features (Use target_country and target_country_code) ---
            print("--- Fetching Safety News ---")
            results["news"] = fetch_safety_news(target_country)

            print("--- Fetching Travel Advisory ---")
            results["advisory"] = fetch_travel_advisory(target_country_code)

            print("--- Fetching Visa Requirements ---")
            results["visa"] = fetch_visa_requirements(current_nationality, target_country_code)

            print("--- Fetching Emergency Numbers (LLM) ---")
            results["emergency"] = fetch_emergency_numbers_llm(target_country)

            print("--- Fetching Cultural Tips (LLM) ---")
            results["culture"] = fetch_cultural_tips_llm(target_country)

            print("--- Fetching Health Info (LLM) ---")
            results["health"] = fetch_health_info_llm(target_country)

            print("--- Generating Packing List (LLM) ---")
            # Pass the weather summary obtained earlier to the packing list generator
            results["packing"] = generate_packing_list_llm(
                destination=current_place, # Use the specific place name
                country=target_country,
                duration_days=current_duration,
                trip_type=current_trip_type,
                weather_summary=weather_summary_for_llm
            )

        # --- Display Results ---
        st.success("✅ Briefing generated!")

        # Use columns for better layout
        col_results1, col_results2 = st.columns(2)

        with col_results1: # Left Column
            # --- Visa ---
            st.subheader("🛂 Visa & Entry Requirements")
            visa_result = results.get("visa", "Data not available.")
            with st.expander("View Details", expanded=False):
                if isinstance(visa_result, str) and "Error:" in visa_result: st.warning(visa_result)
                elif isinstance(visa_result, str): st.markdown(visa_result)
                else: st.info("Visa information could not be retrieved.")

            # --- Advisory ---
            st.subheader("📋 Official Travel Advisory Summary")
            advisory_result = results.get("advisory", "Data not available.")
            with st.expander("View Details", expanded=False):
                if isinstance(advisory_result, str) and "Error:" in advisory_result: st.warning(advisory_result)
                elif isinstance(advisory_result, str): st.markdown(advisory_result)
                else: st.info("Advisory information could not be retrieved.")

            # --- Safety News ---
            st.subheader("🛡️ Recent Safety News & Analysis")
            news_items_list = results.get("news") # Can be list or error string
            with st.expander("View Details", expanded=False):
                safety_llm_summary = "No news analysis available."
                llm_news_input = ""

                if isinstance(news_items_list, list) and news_items_list:
                    # Prepare input for LLM summary
                    llm_news_input = "\n".join([ f"- Title: {item['title']} (Source: {item['source']}, Date: {item.get('date', 'N/A')})\n  Snippet: {item['snippet']}\n  URL: {item['url']}\n" for item in news_items_list])
                    prompt_news_summary = (f"Analyze the following recent news headlines/snippets regarding **{target_country}** (context: travel planned to '{current_place}'). Summarize in 2-3 bullet points the most significant safety and security concerns relevant to a tourist (e.g., widespread unrest, specific crime warnings, travel disruptions, major security incidents). Focus on severity and relevance. Ignore minor or isolated local news unless critical. If no major concerns are apparent, state that.\n\n--- News Items Start ---\n{llm_news_input[:8000]}\n--- News Items End ---\n\n**Safety News Analysis:**") # Limit input size
                    try:
                        print("--- Generating News Summary (LLM) ---")
                        safety_llm_summary = generate_llm_response(prompt_news_summary)
                    except NameError: safety_llm_summary = "LLM service unavailable for news summary."
                    except Exception as e: safety_llm_summary = f"Error summarizing news via AI: {e}"

                elif isinstance(news_items_list, str): # Error message from fetch function
                    safety_llm_summary = f"Could not fetch news: {news_items_list}"
                elif news_items_list == []: # Empty list returned
                    safety_llm_summary = f"No specific safety-related news found via search for {target_country}. This does not guarantee safety; check official advisories."
                else: # Unexpected type
                    safety_llm_summary = "Could not process news data."

                # Display LLM summary or fetch status
                if "Error" in safety_llm_summary or "Could not fetch" in safety_llm_summary or "unavailable" in safety_llm_summary: st.warning(safety_llm_summary)
                elif "No specific safety-related news found" in safety_llm_summary: st.info(safety_llm_summary)
                else: st.markdown(safety_llm_summary)

                # Display Raw News Feed if available
                if isinstance(news_items_list, list) and news_items_list:
                    st.markdown("---")
                    st.subheader("Raw News Feed (from DuckDuckGo)")
                    for item in news_items_list:
                        with st.container(border=True):
                             st.markdown(f"##### [{item['title']}]({item['url']})")
                             st.caption(f"Source: {item['source']} | Date: {item.get('date', 'N/A')}")
                             st.markdown(f"*{item['snippet']}*")
                elif isinstance(news_items_list, list) and not news_items_list:
                    st.markdown("---")
                    # st.info("Raw news feed is empty.") # Info message already displayed above

            # --- Health Info ---
            st.subheader("❤️‍🩹 General Health Information")
            health_result = results.get("health", "Data not available.")
            with st.expander("View Details", expanded=False):
                 if isinstance(health_result, str) and "Error:" in health_result: st.error(health_result)
                 elif isinstance(health_result, str): st.markdown(health_result) # Disclaimer is now added within the fetch function
                 else: st.info("Health information could not be retrieved.")


        with col_results2: # Right Column
            # --- Weather ---
            st.subheader("☀️ Local Weather")
            weather_info = results.get("weather")
            with st.expander("Current Conditions", expanded=True): # Keep expanded
                if weather_info and not weather_info.get("error"):
                    col_w_icon, col_w_data = st.columns([0.2, 0.8])
                    with col_w_icon:
                         icon_code = weather_info.get("icon")
                         if icon_code:
                             st.image(f"http://openweathermap.org/img/wn/{icon_code}@2x.png", width=70)
                         else:
                             st.write(" ") # Placeholder
                    with col_w_data:
                         st.metric(label="Temperature", value=f"{weather_info.get('temperature', '?')}°C", delta=f"Feels like {weather_info.get('feels_like', '?')}°C")
                         st.write(f"**Condition:** {weather_info.get('description', 'N/A')}")
                         st.write(f"**Humidity:** {weather_info.get('humidity', '?')}%")
                         # Display wind speed in km/h for better readability (m/s * 3.6)
                         wind_mps = weather_info.get('wind_speed')
                         wind_kph = f"{wind_mps * 3.6:.1f} km/h" if wind_mps is not None else '?'
                         st.write(f"**Wind:** {wind_kph}")
                         ts = weather_info.get("timestamp")
                         tz_offset = weather_info.get("timezone", 0)
                         if ts:
                             local_time = datetime.utcfromtimestamp(ts + tz_offset)
                             st.caption(f"Observed at: {local_time.strftime('%Y-%m-%d %H:%M')} (Local Time)")
                         api_city = weather_info.get('city_name')
                         api_country = weather_info.get('country')
                         if api_city: st.caption(f"Location: {api_city}{f', {api_country}' if api_country else ''}")


                elif weather_info and weather_info.get("error"):
                     st.warning(f"Weather Update: {weather_info['error']}")
                else:
                     st.info("Weather data could not be retrieved. Ensure location is set and API key is valid.")

            # --- Currency ---
            st.subheader("💰 Currency & Conversion")
            currency_info = results.get("currency")
            with st.expander("Conversion Rate", expanded=True): # Keep expanded
                 if currency_info and not currency_info.get("error"):
                     rate = currency_info.get('rate')
                     home = currency_info.get('home_currency', 'N/A')
                     local_sym = currency_info.get('local_currency_symbol', 'N/A')
                     if rate is not None:
                         st.metric(label=f"1 {home} ≈", value=f"{rate:.3f} {local_sym}") # Show more decimal places for currency
                     else:
                         st.warning("Conversion rate value missing.")
                     update_time = currency_info.get('last_update_utc')
                     if update_time: st.caption(f"Rate updated: {update_time}")
                 elif currency_info and currency_info.get("error"):
                     st.warning(f"Currency Update: {currency_info['error']}")
                 else:
                     st.info("Currency data requires API key, home currency selection, and a valid destination country.")

            # --- Emergency Numbers ---
            st.subheader("🆘 Local Emergency Numbers")
            emergency_result = results.get("emergency", "Data not available.")
            with st.expander("View Numbers", expanded=False):
                 if isinstance(emergency_result, str) and "Error:" in emergency_result: st.error(emergency_result)
                 elif isinstance(emergency_result, str): st.markdown(emergency_result) # Disclaimer added in fetch function
                 else: st.info("Emergency numbers could not be retrieved.")

            # --- Cultural Tips ---
            st.subheader("🤝 Cultural Etiquette Tips")
            culture_result = results.get("culture", "Data not available.")
            with st.expander("View Tips", expanded=False):
                 if isinstance(culture_result, str) and "Error:" in culture_result: st.error(culture_result)
                 elif isinstance(culture_result, str): st.markdown(culture_result) # Note added in fetch function
                 else: st.info("Cultural tips could not be retrieved.")

            # --- Packing List ---
            st.subheader("🎒 Suggested Packing List")
            packing_result = results.get("packing", "Data not available.")
            with st.expander("View List", expanded=False):
                 if isinstance(packing_result, str) and "Error:" in packing_result: st.error(packing_result)
                 elif isinstance(packing_result, str): st.markdown(packing_result) # Note added in fetch function
                 else: st.info("Packing list could not be generated.")

    else:
         # This else corresponds to `if valid_request:`
         st.error("❌ Briefing generation cancelled due to missing inputs or errors. Please check warnings above.")


# --- Map Display (Main Area) ---
st.markdown("---")
st.subheader("🗺️ Select Destination on Map")
st.caption("Click on the map to select your destination. The coordinates and address will update automatically.")

# Create the map centered on the current session state center, with the current zoom
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    control_scale=True, # Adds a scale bar
    tiles='OpenStreetMap' # Specify base tiles
)

# Add a marker only if a valid location (not default) has been selected
if st.session_state.coords != DEFAULT_MAP_LOCATION:
   tooltip_text = st.session_state.place_name if st.session_state.place_name else "Selected Location"
   # Ensure coords are valid floats before creating marker
   try:
        lat, lon = float(st.session_state.coords[0]), float(st.session_state.coords[1])
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"<b>{tooltip_text}</b><br>({lat:.4f}, {lon:.4f})", max_width=250),
            tooltip=tooltip_text,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
   except (ValueError, TypeError, IndexError):
        print(f"Invalid coords in state for marker: {st.session_state.coords}")
        st.warning("Marker could not be placed due to invalid coordinates.")


# Display the map using streamlit-folium
# Pass the current center and zoom from session state
map_output = st_folium(
    m,
    key="map_key", # Consistent key
    center=st.session_state.map_center,
    zoom=st.session_state.map_zoom,
    width='100%',
    height=450, # Adjusted height slightly
    returned_objects=["last_clicked"] # Only need last_clicked event
)

# --- Handle Map Click ---
# Check if the map_output exists and contains 'last_clicked' data
if map_output and map_output.get("last_clicked"):
    # Extract clicked coordinates
    clicked_lat = map_output["last_clicked"]["lat"]
    clicked_lng = map_output["last_clicked"]["lng"]
    clicked_coords = (clicked_lat, clicked_lng)

    # Check if the click location is significantly different from the current coords in state
    # This prevents triggering updates if the user just clicks near the existing marker
    lat_diff = abs(clicked_coords[0] - st.session_state.coords[0])
    lon_diff = abs(clicked_coords[1] - st.session_state.coords[1])

    # Only update if the click is different enough OR if there's no valid place name yet
    if (lat_diff > 1e-5 or lon_diff > 1e-5) or not st.session_state.place_name or st.session_state.place_name.startswith("Lat:"):
        # Call the function to update state based on map click
        update_location_from_map(map_output)
        # Force Streamlit to rerun the script immediately to reflect the state changes in the UI
        # (updates the text input, map center/zoom, marker position)
        st.rerun()


# --- Disclaimer ---
st.markdown("---")
st.caption("Disclaimer: Information gathered via web search (DuckDuckGo), APIs (OpenWeatherMap, ExchangeRate-API), and AI models (e.g., OpenAI via `llm_service`). Data may be incomplete, inaccurate, or outdated. AI-generated content requires careful verification. **ALWAYS verify critical information (visas, safety alerts, health requirements, emergency numbers) with official government or relevant organizational sources before making travel decisions.** This tool is for informational purposes only.")