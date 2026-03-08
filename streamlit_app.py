# streamlit_app.py
import os
import asyncio
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

import streamlit as st
from serpapi import GoogleSearch
from google import genai

# -------------------------
# Helpers
# -------------------------
TAG_RE = re.compile(r"<[^>]*>")

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")
    s = s.replace("＜", "<").replace("＞", ">")
    s = TAG_RE.sub("", s)
    s = s.replace("<", "").replace(">", "")
    return s.strip()

def normalize_date(date_str: str) -> str:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError("Invalid date. Use YYYY-MM-DD.")

def _extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.close()
        except Exception:
            pass

@st.cache_data(ttl=60 * 30, show_spinner=False)
def serpapi_search_cached(params: dict) -> dict:
    return GoogleSearch(params).get_dict()

async def serpapi_search(params: dict) -> dict:
    return await asyncio.to_thread(lambda: serpapi_search_cached(params))

async def gemini_call(client, model: str, prompt: str, retries: int = 5) -> str:
    def _call() -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return _extract_text(resp)

    for i in range(retries):
        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ["503", "overloaded", "unavailable", "429", "quota"]):
                wait = (2 ** i) + random.uniform(0, 0.8)
                await asyncio.sleep(wait)
                continue
            return f"⚠️ AI error: {e}"
    return "⚠️ AI is busy / quota-limited. Please try again shortly."

def google_flights_search_url(origin, destination, out_date, ret_date):
    q = f"Flights from {origin} to {destination} on {out_date} returning {ret_date}"
    return "https://www.google.com/travel/flights?q=" + quote_plus(q)

def google_hotels_search_url(location, check_in, check_out):
    q = f"Hotels in {location} from {check_in} to {check_out}"
    return "https://www.google.com/travel/hotels?q=" + quote_plus(q)

def render_star_rating(rating: float) -> str:
    """Return a visual star rating string (half-star precision)."""
    if rating <= 0:
        return "—"
    full = int(rating // 2)
    half = 1 if (rating % 2) >= 1 else 0
    empty = 5 - full - half
    stars = "★" * full + ("½" if half else "") + "☆" * empty
    score_5 = rating / 2.0
    return f"{stars} {score_5:.1f}/5"

# -------------------------
# SerpAPI fetchers
# -------------------------
# ── City name → IATA lookup ───────────────────────────────────────────────────
CITY_TO_IATA = {
    # Southeast Asia
    "singapore":"SIN","kuala lumpur":"KUL","kl":"KUL","bangkok":"BKK",
    "phuket":"HKT","chiang mai":"CNX","ho chi minh":"SGN","saigon":"SGN",
    "hanoi":"HAN","da nang":"DAD","bali":"DPS","denpasar":"DPS",
    "jakarta":"CGK","surabaya":"SUB","manila":"MNL","cebu":"CEB",
    "yangon":"RGN","phnom penh":"PNH","siem reap":"REP","vientiane":"VTE",
    "colombo":"CMB","male":"MLE","kathmandu":"KTM","dhaka":"DAC",
    # East Asia
    "tokyo":"NRT","osaka":"KIX","kyoto":"KIX","sapporo":"CTS","fukuoka":"FUK",
    "seoul":"ICN","busan":"PUS","beijing":"PEK","shanghai":"PVG",
    "guangzhou":"CAN","shenzhen":"SZX","chengdu":"CTU","hong kong":"HKG",
    "taipei":"TPE","taichung":"RMQ","macau":"MFM","ulaanbaatar":"ULN",
    # South Asia
    "mumbai":"BOM","bombay":"BOM","delhi":"DEL","new delhi":"DEL",
    "bangalore":"BLR","bengaluru":"BLR","chennai":"MAA","hyderabad":"HYD",
    "kolkata":"CCU","goa":"GOI","ahmedabad":"AMD","pune":"PNQ",
    "islamabad":"ISB","karachi":"KHI","lahore":"LHE",
    # Middle East
    "dubai":"DXB","abu dhabi":"AUH","doha":"DOH","kuwait":"KWI",
    "riyadh":"RUH","jeddah":"JED","muscat":"MCT","bahrain":"BAH",
    "amman":"AMM","beirut":"BEY","tel aviv":"TLV",
    # Europe
    "london":"LHR","heathrow":"LHR","gatwick":"LGW","stansted":"STN",
    "paris":"CDG","amsterdam":"AMS","frankfurt":"FRA","madrid":"MAD",
    "barcelona":"BCN","rome":"FCO","milan":"MXP","munich":"MUC",
    "vienna":"VIE","zurich":"ZRH","geneva":"GVA","brussels":"BRU",
    "lisbon":"LIS","athens":"ATH","istanbul":"IST","dublin":"DUB",
    "copenhagen":"CPH","stockholm":"ARN","oslo":"OSL","helsinki":"HEL",
    "warsaw":"WAW","prague":"PRG","budapest":"BUD","bucharest":"OTP",
    # Americas
    "new york":"JFK","nyc":"JFK","los angeles":"LAX","la":"LAX",
    "chicago":"ORD","miami":"MIA","san francisco":"SFO","seattle":"SEA",
    "boston":"BOS","washington":"IAD","atlanta":"ATL","dallas":"DFW",
    "houston":"IAH","denver":"DEN","las vegas":"LAS","honolulu":"HNL",
    "toronto":"YYZ","vancouver":"YVR","montreal":"YUL","calgary":"YYC",
    "mexico city":"MEX","cancun":"CUN","sao paulo":"GRU","rio de janeiro":"GIG",
    "buenos aires":"EZE","lima":"LIM","bogota":"BOG","santiago":"SCL",
    # Africa & Oceania
    "nairobi":"NBO","johannesburg":"JNB","cape town":"CPT","cairo":"CAI",
    "casablanca":"CMN","lagos":"LOS","accra":"ACC","addis ababa":"ADD",
    "sydney":"SYD","melbourne":"MEL","brisbane":"BNE","perth":"PER",
    "auckland":"AKL","christchurch":"CHC","nadi":"NAN",
}

# Cities with multiple major airports — user gets to pick
MULTI_AIRPORT_CITIES = {
    "bangkok":        [("BKK", "BKK — Suvarnabhumi (main international)"),
                       ("DMK", "DMK — Don Mueang (budget/AirAsia)")],
    "london":         [("LHR", "LHR — Heathrow (main international)"),
                       ("LGW", "LGW — Gatwick"),
                       ("STN", "STN — Stansted (budget)"),
                       ("LTN", "LTN — Luton (budget)"),
                       ("LCY", "LCY — City Airport")],
    "new york":       [("JFK", "JFK — John F. Kennedy"),
                       ("EWR", "EWR — Newark"),
                       ("LGA", "LGA — LaGuardia (domestic)")],
    "nyc":            [("JFK", "JFK — John F. Kennedy"),
                       ("EWR", "EWR — Newark"),
                       ("LGA", "LGA — LaGuardia (domestic)")],
    "tokyo":          [("NRT", "NRT — Narita (main international)"),
                       ("HND", "HND — Haneda (closer to city)")],
    "paris":          [("CDG", "CDG — Charles de Gaulle (main)"),
                       ("ORY", "ORY — Orly (shorter flights)")],
    "milan":          [("MXP", "MXP — Malpensa (main international)"),
                       ("LIN", "LIN — Linate (closer to city)"),
                       ("BGY", "BGY — Bergamo/Orio (budget)")],
    "chicago":        [("ORD", "ORD — O'Hare (main)"),
                       ("MDW", "MDW — Midway (budget/Southwest)")],
    "los angeles":    [("LAX", "LAX — Los Angeles International"),
                       ("BUR", "BUR — Burbank (north LA)"),
                       ("LGB", "LGB — Long Beach"),
                       ("ONT", "ONT — Ontario (inland)")],
    "la":             [("LAX", "LAX — Los Angeles International"),
                       ("BUR", "BUR — Burbank"),
                       ("LGB", "LGB — Long Beach")],
    "washington":     [("IAD", "IAD — Dulles International"),
                       ("DCA", "DCA — Reagan National (city)"),
                       ("BWI", "BWI — Baltimore/Washington")],
    "houston":        [("IAH", "IAH — George Bush Intercontinental"),
                       ("HOU", "HOU — Hobby (Southwest/domestic)")],
    "dallas":         [("DFW", "DFW — Dallas/Fort Worth (main)"),
                       ("DAL", "DAL — Dallas Love Field (Southwest)")],
    "miami":          [("MIA", "MIA — Miami International"),
                       ("FLL", "FLL — Fort Lauderdale (budget/closer)")],
    "san francisco":  [("SFO", "SFO — San Francisco International"),
                       ("OAK", "OAK — Oakland (budget)"),
                       ("SJC", "SJC — San Jose")],
    "kuala lumpur":   [("KUL", "KUL — KLIA (main international)"),
                       ("KUL", "KUL — KLIA2 (AirAsia hub)")],
    "kl":             [("KUL", "KUL — KLIA (main)"),
                       ("KUL", "KUL — KLIA2 (AirAsia)")],
    "jakarta":        [("CGK", "CGK — Soekarno-Hatta (main)"),
                       ("HLP", "HLP — Halim Perdanakusuma (domestic)")],
    "seoul":          [("ICN", "ICN — Incheon (main international)"),
                       ("GMP", "GMP — Gimpo (domestic/Japan/China)")],
    "osaka":          [("KIX", "KIX — Kansai (main international)"),
                       ("ITM", "ITM — Itami (domestic)")],
    "rome":           [("FCO", "FCO — Fiumicino (main)"),
                       ("CIA", "CIA — Ciampino (budget/Ryanair)")],
    "barcelona":      [("BCN", "BCN — El Prat (main)"),
                       ("GRO", "GRO — Girona (budget, 1hr away)")],
    "amsterdam":      [("AMS", "AMS — Schiphol (main)"),
                       ("EIN", "EIN — Eindhoven (budget, 1.5hr)")],
    "moscow":         [("SVO", "SVO — Sheremetyevo (main)"),
                       ("DME", "DME — Domodedovo"),
                       ("VKO", "VKO — Vnukovo")],
    "toronto":        [("YYZ", "YYZ — Pearson International (main)"),
                       ("YTZ", "YTZ — Billy Bishop (downtown)")],
    "sao paulo":      [("GRU", "GRU — Guarulhos (main international)"),
                       ("CGH", "CGH — Congonhas (domestic/closer)")],
    "buenos aires":   [("EZE", "EZE — Ezeiza (international)"),
                       ("AEP", "AEP — Aeroparque (domestic/regional)")],
    "sydney":         [("SYD", "SYD — Kingsford Smith (only major airport)")],
    "melbourne":      [("MEL", "MEL — Tullamarine (main)"),
                       ("AVV", "AVV — Avalon (budget/Jetstar)")],
    "istanbul":       [("IST", "IST — Istanbul Airport (main, new)"),
                       ("SAW", "SAW — Sabiha Gökçen (Asian side/budget)")],
    "stockholm":      [("ARN", "ARN — Arlanda (main)"),
                       ("BMA", "BMA — Bromma (domestic)"),
                       ("NYO", "NYO — Skavsta (budget, 100km away)")],
    "dubai":          [("DXB", "DXB — Dubai International (main)"),
                       ("DWC", "DWC — Al Maktoum (some budget)")],
}

def get_airport_options(city_input: str):
    """Return list of (IATA, label) if city has multiple airports, else None."""
    key = city_input.strip().lower()
    if key in MULTI_AIRPORT_CITIES and len(MULTI_AIRPORT_CITIES[key]) > 1:
        return MULTI_AIRPORT_CITIES[key]
    # partial match
    for city, options in MULTI_AIRPORT_CITIES.items():
        if city in key and len(options) > 1:
            return options
    return None

def resolve_to_iata(name: str, gemini_client=None, gemini_model: str = "") -> str:
    """
    Convert a city/airport name to IATA code.
    1. If already a 3-letter IATA code → return uppercase
    2. Look up in CITY_TO_IATA table
    3. Fallback: ask Gemini (sync, called inside run_async)
    """
    s = name.strip()
    # Already IATA?
    if len(s) == 3 and s.isalpha():
        return s.upper()
    key = s.lower()
    if key in CITY_TO_IATA:
        return CITY_TO_IATA[key]
    # Partial match (e.g. "Phuket, Thailand" → "phuket")
    for city, iata in CITY_TO_IATA.items():
        if city in key:
            return iata
    # Gemini fallback
    if gemini_client:
        try:
            prompt = (
                f"What is the main commercial airport IATA code for: {s}? "
                "Reply with ONLY the 3-letter uppercase IATA code, nothing else."
            )
            resp = gemini_client.models.generate_content(model=gemini_model, contents=prompt)
            code = (getattr(resp, "text", "") or "").strip().upper()
            if len(code) == 3 and code.isalpha():
                return code
        except Exception:
            pass
    # Last resort: return as-is (SerpAPI may still handle it)
    return s.upper()


async def get_flights(serp_key, origin, destination, out_date, ret_date, currency,
                      gemini_client=None, gemini_model=""):
    origin_iata = resolve_to_iata(origin, gemini_client, gemini_model)
    dest_iata   = resolve_to_iata(destination, gemini_client, gemini_model)

    params = {
        "engine": "google_flights",
        "api_key": serp_key,
        "departure_id": origin_iata,
        "arrival_id":   dest_iata,
        "outbound_date": out_date,
        "return_date": ret_date,
        "currency": currency,
        "hl": "en",
        "gl": "us",
    }
    results = await serpapi_search(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    return results.get("best_flights", []) or []

async def get_price_for_date(serp_key, origin, destination, out_date, trip_length_days, currency):
    """Fetch the cheapest flight price for a single outbound date. Returns (date_str, price_float)."""
    from datetime import date, timedelta
    try:
        ret = (date.fromisoformat(out_date) + timedelta(days=trip_length_days)).strftime("%Y-%m-%d")
        params = {
            "engine": "google_flights",
            "api_key": serp_key,
            "departure_id": resolve_to_iata(origin),
            "arrival_id":   resolve_to_iata(destination),
            "outbound_date": out_date,
            "return_date": ret,
            "currency": currency,
            "hl": "en",
            "gl": "us",
        }
        results = await serpapi_search(params)
        best = results.get("best_flights", []) or []
        if not best:
            return (out_date, None)
        price = best[0].get("price")
        return (out_date, float(price) if price else None)
    except Exception:
        return (out_date, None)

async def get_hotels(serp_key, location, check_in, check_out, currency, min_rating):
    params = {
        "engine": "google_hotels",
        "api_key": serp_key,
        "q": location,
        "check_in_date": check_in,
        "check_out_date": check_out,
        "currency": currency,
        "hl": "en",
        "gl": "us",
    }
    results = await serpapi_search(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    props = results.get("properties", []) or []
    if min_rating > 0:
        # SerpAPI overall_rating is out of 10; user picks 1-5 stars → multiply by 2
        threshold = float(min_rating) * 2.0
        props = [p for p in props if float(p.get("overall_rating", 0) or 0) >= threshold]
    return props

# -------------------------
# Formatting
# -------------------------
async def get_weather(destination_iata: str, start_date: str, end_date: str) -> dict:
    """
    Fetch weather forecast via Open-Meteo (free, no key).
    First geocodes the IATA city name via Open-Meteo geocoding,
    then fetches daily max temp + precipitation probability.
    Returns dict with keys: city, daily (list of dicts).
    """
    import json
    # IATA → city name mapping for common hubs; fallback to raw string
    IATA_CITY = {
        "NRT":"Tokyo","HND":"Tokyo","KIX":"Osaka","ITM":"Osaka","CTS":"Sapporo",
        "SIN":"Singapore","KUL":"Kuala Lumpur","BKK":"Bangkok","DMK":"Bangkok",
        "HKG":"Hong Kong","ICN":"Seoul","GMP":"Seoul","PEK":"Beijing","PVG":"Shanghai",
        "SYD":"Sydney","MEL":"Melbourne","BNE":"Brisbane","AKL":"Auckland",
        "LHR":"London","LGW":"London","CDG":"Paris","AMS":"Amsterdam","FRA":"Frankfurt",
        "DXB":"Dubai","AUH":"Abu Dhabi","DOH":"Doha","BOM":"Mumbai","DEL":"Delhi",
        "JFK":"New York","LAX":"Los Angeles","SFO":"San Francisco","ORD":"Chicago",
        "MIA":"Miami","YYZ":"Toronto","YVR":"Vancouver","GRU":"São Paulo",
        "SCL":"Santiago","EZE":"Buenos Aires","NBO":"Nairobi","CPT":"Cape Town",
        "MNL":"Manila","CGK":"Jakarta","SGN":"Ho Chi Minh City","HAN":"Hanoi",
        "RGN":"Yangon","CMB":"Colombo","DAC":"Dhaka","KTM":"Kathmandu",
    }
    # Try IATA lookup first; if not found treat the input as a city name directly
    _key = destination_iata.strip().upper()
    city = IATA_CITY.get(_key, destination_iata.strip())

    try:
        # Step 1: geocode
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote_plus(city)}&count=1&language=en&format=json"
        geo_res = await asyncio.to_thread(lambda: __import__('urllib.request', fromlist=['urlopen']).urlopen(geo_url, timeout=5).read())
        geo_data = json.loads(geo_res)
        results_geo = geo_data.get("results", [])
        if not results_geo:
            return {}
        lat = results_geo[0]["latitude"]
        lon = results_geo[0]["longitude"]
        city_name = results_geo[0].get("name", city)

        # Step 2: forecast
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
            f"&start_date={start_date}&end_date={end_date}"
            f"&timezone=auto&temperature_unit=celsius"
        )
        wx_res = await asyncio.to_thread(lambda: __import__('urllib.request', fromlist=['urlopen']).urlopen(wx_url, timeout=5).read())
        wx_data = json.loads(wx_res)
        daily = wx_data.get("daily", {})
        dates     = daily.get("time", [])
        tmax      = daily.get("temperature_2m_max", [])
        tmin      = daily.get("temperature_2m_min", [])
        precip    = daily.get("precipitation_probability_max", [])
        wcode     = daily.get("weathercode", [])

        WMO = {0:"☀️",1:"🌤",2:"⛅",3:"☁️",45:"🌫",48:"🌫",51:"🌦",53:"🌦",55:"🌧",
               61:"🌧",63:"🌧",65:"🌧",71:"🌨",73:"🌨",75:"❄️",80:"🌦",81:"🌧",
               82:"⛈",85:"🌨",86:"❄️",95:"⛈",96:"⛈",99:"⛈"}

        day_list = []
        for i, d in enumerate(dates):
            day_list.append({
                "date": d,
                "tmax": tmax[i] if i < len(tmax) else None,
                "tmin": tmin[i] if i < len(tmin) else None,
                "precip": precip[i] if i < len(precip) else None,
                "icon":  WMO.get(wcode[i] if i < len(wcode) else 0, "🌡"),
            })
        return {"city": city_name, "daily": day_list}
    except Exception:
        return {}

def fmt_flights(best_flights, origin, destination, out_date, ret_date):
    out = []
    fallback_link = google_flights_search_url(origin, destination, out_date, ret_date)
    for f in best_flights[:12]:
        legs = f.get("flights") or []
        if not legs:
            continue
        leg0 = legs[0]
        dep = leg0.get("departure_airport", {}) or {}
        arr = leg0.get("arrival_airport", {}) or {}
        stops = "Nonstop" if len(legs) == 1 else f"{len(legs)-1} stop(s)"
        link = f.get("link") or f.get("booking_link") or fallback_link
        out.append({
            "airline": clean_text(leg0.get("airline", "Unknown")),
            "airline_logo": clean_text(leg0.get("airline_logo", "")),
            "price": clean_text(f.get("price", "N/A")),
            "duration_min": f.get("total_duration", None),
            "stops": clean_text(stops),
            "depart_code": clean_text(dep.get("id", "")),
            "depart_time": clean_text(dep.get("time", "")),
            "arrive_code": clean_text(arr.get("id", "")),
            "arrive_time": clean_text(arr.get("time", "")),
            "class": clean_text(leg0.get("travel_class", "Economy")),
            "link": clean_text(link),
            "raw": f,
        })
    return out

def _is_gps(v: str) -> bool:
    """Return True if string looks like raw GPS coordinates — never show these."""
    import re as _re
    return bool(_re.match(r"^-?\d+\.\d+,\s*-?\d+\.\d+$", v.strip()))

def _pick_hotel_area(raw):
    candidates = [
        raw.get("neighborhood"),
        raw.get("area"),
        raw.get("gps_coordinates_address"),  # SerpAPI sometimes puts address here
        raw.get("address"),
        raw.get("location"),
        raw.get("formatted_address"),
        raw.get("city"),
        raw.get("district"),
        raw.get("suburb"),
    ]
    for v in candidates:
        v = clean_text(v)
        if v and not _is_gps(v):
            return v
    # Check nested address objects
    for addr_key in ["address_info", "address_details", "location_info"]:
        addr = raw.get(addr_key) or {}
        if isinstance(addr, dict):
            for k in ["neighborhood", "district", "suburb", "area",
                       "address", "formatted_address", "city", "region"]:
                v = clean_text(addr.get(k))
                if v and not _is_gps(v):
                    return v
    return ""

def _pick_hotel_amenities(raw):
    am = raw.get("amenities") or raw.get("top_features") or raw.get("features") or []
    if isinstance(am, list):
        am = [clean_text(x) for x in am if clean_text(x)][:4]
        return am
    v = clean_text(am)
    return [v] if v else []

def fmt_hotels(props, location_query, check_in, check_out, city_name: str = ""):
    """
    city_name: human-readable destination city (e.g. "London") used for
               the area fallback instead of the raw IATA/query string.
    """
    out = []
    fallback_link = google_hotels_search_url(location_query, check_in, check_out)
    # Use city_name for fallback if provided, else location_query
    fallback_area = f"Near {city_name}" if city_name else f"Near {location_query}"

    for h in props[:18]:
        rate = (h.get("rate_per_night", {}) or {}).get("lowest", h.get("price", "N/A"))
        link = h.get("link") or fallback_link
        area = _pick_hotel_area(h)
        if not area:
            area = fallback_area

        # hotel_class = official star classification (1–5), distinct from review score
        hotel_class = None
        raw_class = h.get("hotel_class") or h.get("class") or h.get("star_rating")
        if raw_class is not None:
            try:
                hotel_class = int(float(str(raw_class).replace("★","").strip()))
                hotel_class = max(1, min(5, hotel_class))  # clamp to 1-5
            except Exception:
                hotel_class = None

        out.append({
            "name": clean_text(h.get("name", "Unknown")),
            "price": clean_text(rate),
            "review_score": float(h.get("overall_rating", 0.0) or 0.0),  # 0-10 guest score
            "hotel_class": hotel_class,                                    # 1-5 luxury stars
            "rating": float(h.get("overall_rating", 0.0) or 0.0),        # kept for compat
            "reviews_count": clean_text(h.get("reviews") or h.get("reviews_count") or ""),
            "area": area,
            "type": clean_text(h.get("property_type") or h.get("type") or ""),
            "amenities": _pick_hotel_amenities(h),
            "distance": clean_text(h.get("distance_to_center") or h.get("distance") or ""),
            "link": clean_text(link),
            "raw": h,
        })
    return out

def _parse_price(price_val) -> float:
    if price_val is None:
        return float("inf")
    if isinstance(price_val, (int, float)):
        return float(price_val)
    digits = "".join(ch for ch in clean_text(price_val) if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else float("inf")
    except Exception:
        return float("inf")

def sort_flights(flights, sort_key):
    if sort_key == "Cheapest":
        return sorted(flights, key=lambda x: _parse_price(x.get("price")))
    if sort_key == "Fastest":
        return sorted(flights, key=lambda x: (x.get("duration_min") is None, x.get("duration_min") or 10**9))
    if sort_key == "Fewest stops":
        def stops_num(s):
            s = (s or "").lower()
            if "nonstop" in s:
                return 0
            digits = "".join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else 99
        return sorted(flights, key=lambda x: stops_num(x.get("stops", "")))
    return flights

def sort_hotels(hotels, sort_key):
    if sort_key == "Top rated":
        return sorted(hotels, key=lambda x: x.get("rating", 0), reverse=True)
    if sort_key == "Cheapest":
        return sorted(hotels, key=lambda x: _parse_price(x.get("price")))
    return hotels

def clamp_multiselect(selection, limit):
    if len(selection) <= limit:
        return selection, False
    return selection[:limit], True

def fmt_duration(mins):
    if mins is None:
        return "—"
    h, m = divmod(int(mins), 60)
    return f"{h}h {m}m" if h else f"{m}m"

def fmt_flights_for_ai(flights_list: list) -> str:
    """Format flight dicts into clean structured text for Gemini."""
    if not flights_list:
        return "No flights available."
    lines = []
    for i, f in enumerate(flights_list, 1):
        dur = fmt_duration(f.get("duration_min"))
        lines.append(
            f"Flight {i}: {f.get('airline','?')} | {f.get('class','Economy')} | "
            f"{f.get('price','N/A')} | {f.get('stops','?')} | "
            f"{f.get('depart_code','?')} {f.get('depart_time','')} → "
            f"{f.get('arrive_code','?')} {f.get('arrive_time','')} | "
            f"Duration: {dur}"
        )
        if f.get("link"):
            lines.append(f"  Link: {f['link']}")
    return "\n".join(lines)

def fmt_hotels_for_ai(hotels_list: list) -> str:
    """Format hotel dicts into clean structured text for Gemini."""
    if not hotels_list:
        return "No hotels available."
    lines = []
    for i, h in enumerate(hotels_list, 1):
        amenities = ", ".join(h.get("amenities") or []) or "N/A"
        lines.append(
            f"Hotel {i}: {h.get('name','?')} | {h.get('type','')} | "
            f"{h.get('price','N/A')}/night | Rating: {h.get('rating',0)/2:.1f}/5 | "
            f"Area: {h.get('area','N/A')} | Highlights: {amenities}"
        )
        if h.get("link"):
            lines.append(f"  Link: {h['link']}")
    return "\n".join(lines)

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

# -------------------------
# CSS — Sleek Minimal Dark
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:       #1a1008;
  --surface:  #231508;
  --surface2: #2c1c0e;
  --border:   rgba(255,200,150,0.10);
  --border2:  rgba(255,200,150,0.22);
  --text:     #fdf4ec;
  --muted:    #a8896e;
  --accent:   #fa7c4f;
  --accent2:  #ffb347;
  --warn:     #f5c842;
  --radius:   14px;
  --radius-sm:9px;
}

/* ---------- base ---------- */
html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main, .main .block-container,
.stApp {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}
.block-container {
  padding-top: 2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px !important;
}

/* ---------- headings ---------- */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; font-weight: 600 !important; letter-spacing: -0.02em; }
p, li, span, label, div { color: var(--text) !important; }
.stMarkdown p { line-height: 1.7; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: #1e1108 !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* sidebar section label */
.sidebar-label {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted) !important;
  margin-bottom: 0.5rem;
  margin-top: 1.2rem;
}

/* ---------- top chrome bar ---------- */
header[data-testid="stHeader"] {
  background: var(--bg) !important;
  border-bottom: 1px solid var(--border) !important;
}

/* ---------- date input field ---------- */
div[data-testid="stDateInput"] input,
div[data-testid="stDateInput"] input:hover,
div[data-testid="stDateInput"] input:focus {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
}
div[data-testid="stDateInput"] > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
}

/* ---------- calendar popup ---------- */
/* Outer popover shell */
[data-baseweb="calendar"],
[data-baseweb="datepicker"],
[data-baseweb="calendar"] *,
div[class*="calendarContainer"],
div[class*="CalendarContainer"] {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border-color: var(--border2) !important;
}

/* Calendar wrapper / month container */
[data-baseweb="calendar"] [role="application"],
[data-baseweb="calendar"] table,
[data-baseweb="calendar"] thead,
[data-baseweb="calendar"] tbody,
[data-baseweb="calendar"] tr,
[data-baseweb="calendar"] th,
[data-baseweb="calendar"] td {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: none !important;
}

/* Day cells */
[data-baseweb="calendar"] [role="gridcell"] button,
[data-baseweb="calendar"] [role="gridcell"] div {
  background: transparent !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}
[data-baseweb="calendar"] [role="gridcell"] button:hover {
  background: rgba(250,124,79,0.2) !important;
  color: var(--text) !important;
}

/* Selected / today highlight */
[data-baseweb="calendar"] [aria-selected="true"] button,
[data-baseweb="calendar"] [aria-selected="true"] div {
  background: var(--accent) !important;
  color: #1a1008 !important;
  border-radius: 8px !important;
}

/* Month/year header nav buttons */
[data-baseweb="calendar"] button[aria-label*="previous"],
[data-baseweb="calendar"] button[aria-label*="next"],
[data-baseweb="calendar"] button[aria-label*="Previous"],
[data-baseweb="calendar"] button[aria-label*="Next"] {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
}
[data-baseweb="calendar"] button[aria-label*="previous"]:hover,
[data-baseweb="calendar"] button[aria-label*="next"]:hover,
[data-baseweb="calendar"] button[aria-label*="Previous"]:hover,
[data-baseweb="calendar"] button[aria-label*="Next"]:hover {
  background: rgba(250,124,79,0.15) !important;
}

/* Month/year dropdowns inside calendar */
[data-baseweb="calendar"] select,
[data-baseweb="calendar"] [data-baseweb="select"] > div {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
}

/* Weekday header labels (Su Mo Tu ...) */
[data-baseweb="calendar"] [role="columnheader"] {
  background: var(--surface2) !important;
  color: var(--muted) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
}

/* Disabled (out of range) days */
[data-baseweb="calendar"] [aria-disabled="true"] button {
  color: rgba(168,137,110,0.35) !important;
  cursor: not-allowed !important;
}

/* The floating popover container itself */
[data-baseweb="popover"][data-placement],
[data-baseweb="popover"] > div:first-child {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius) !important;
  box-shadow: 0 8px 40px rgba(0,0,0,0.6) !important;
}

/* ---------- text inputs ---------- */
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] input:hover {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.95rem !important;
  transition: border-color 0.2s;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(250,124,79,0.15) !important;
  outline: none !important;
}

/* ---------- selectbox / multiselect ---------- */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] > div > div > div,
div[data-testid="stMultiSelect"] > div > div,
div[data-testid="stMultiSelect"] > div > div > div,
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
ul[data-baseweb="menu"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
}
[data-baseweb="option"],
[role="option"] {
  background: var(--surface2) !important;
  color: var(--text) !important;
}
[data-baseweb="option"]:hover,
[role="option"]:hover {
  background: rgba(250,124,79,0.18) !important;
  color: var(--text) !important;
}
[aria-selected="true"][data-baseweb="option"],
[aria-selected="true"][role="option"] {
  background: rgba(250,124,79,0.25) !important;
  color: var(--text) !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] input { color: var(--text) !important; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }

/* ---------- number input ---------- */
div[data-testid="stNumberInput"] input,
div[data-testid="stNumberInput"] input:hover,
div[data-testid="stNumberInput"] input:focus {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
}
div[data-testid="stNumberInput"] > div,
div[data-testid="stNumberInput"] > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
}
/* +/- stepper buttons */
div[data-testid="stNumberInput"] button {
  background: var(--surface2) !important;
  border: 0 !important;
  color: var(--muted) !important;
}
div[data-testid="stNumberInput"] button:hover {
  background: rgba(250,124,79,0.15) !important;
  color: var(--text) !important;
}

/* ---------- slider ---------- */
div[data-testid="stSlider"] * { color: var(--text) !important; }
div[data-testid="stSlider"] [role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
div[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:nth-child(2) {
  background: var(--accent) !important;
}
/* ---------- buttons ---------- */
.stButton > button {
  background: var(--accent) !important;
  color: #1a1008 !important;
  border: 0 !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.6rem 1.4rem !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.01em !important;
  transition: opacity 0.18s, transform 0.12s !important;
}
.stButton > button:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.55rem 1.2rem !important;
  font-weight: 500 !important;
  transition: background 0.18s !important;
}
.stDownloadButton > button:hover {
  background: rgba(250,124,79,0.12) !important;
}

/* ---------- tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
  background: transparent !important;
  border: 0 !important;
  color: var(--muted) !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  padding: 0.6rem 1.2rem !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.18s, border-color 0.18s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--text) !important;
}

/* ---------- cards (border containers) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--border2) !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
}

/* ---------- caption / muted ---------- */
.stCaption, [data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.03em;
}

/* ---------- info / warning / error ---------- */
[data-testid="stAlert"] {
  border-radius: var(--radius-sm) !important;
  border-left-width: 3px !important;
}

/* ---------- multiselect tags ---------- */
[data-baseweb="tag"] {
  background: rgba(250,124,79,0.15) !important;
  border: 1px solid rgba(250,124,79,0.35) !important;
  border-radius: 6px !important;
}

/* ---------- custom card classes ---------- */
.flight-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.3rem;
  margin-bottom: 0.75rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.flight-card:hover {
  border-color: var(--border2);
  box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.pill {
  display: inline-block;
  padding: 0.18rem 0.65rem;
  border-radius: 99px;
  font-size: 0.75rem;
  font-weight: 500;
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--border);
  margin-right: 0.35rem;
}
.pill-green { background: rgba(255,179,71,0.14); border-color: rgba(255,179,71,0.35); color: var(--accent2) !important; }
.pill-blue  { background: rgba(250,124,79,0.12); border-color: rgba(250,124,79,0.30); color: var(--accent)  !important; }
.pill-warn  { background: rgba(245,200,66,0.12);  border-color: rgba(245,200,66,0.30);  color: var(--warn)   !important; }

.route-line {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-family: 'DM Mono', monospace;
}
.iata { font-size: 1.5rem; font-weight: 500; letter-spacing: -0.02em; }
.time-small { font-size: 0.8rem; color: var(--muted) !important; }
.dot-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border2), transparent 40%, var(--border2));
  position: relative;
}
.dot-line::before, .dot-line::after {
  content:'';
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--muted);
}
.dot-line::before { left: 0; }
.dot-line::after  { right: 0; }

.price-tag {
  font-family: 'DM Mono', monospace;
  font-size: 1.25rem;
  font-weight: 500;
  color: var(--accent2) !important;
}

.hotel-name { font-size: 1.05rem; font-weight: 600; }
.stars { color: #f5c518 !important; letter-spacing: 0.05em; }
.amenity-chip {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 6px;
  font-size: 0.72rem;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  margin: 0.15rem 0.15rem 0 0;
  color: var(--muted) !important;
}

.section-title {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted) !important;
  margin-bottom: 0.25rem;
}
.link-btn {
  font-size: 0.8rem;
  color: var(--accent) !important;
  text-decoration: none;
}
.link-btn:hover { text-decoration: underline; }

/* ---------- empty state ---------- */
.empty-state {
  text-align: center;
  padding: 3.5rem 1rem;
  color: var(--muted) !important;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-title { font-size: 1rem; font-weight: 500; color: var(--muted) !important; margin-bottom: 0.3rem; }
.empty-sub { font-size: 0.83rem; color: var(--muted) !important; }

/* ---------- hero header ---------- */
.hero {
  padding: 1.8rem 0 2rem;
  position: relative;
  overflow: hidden;
}

/* animated gradient orbs behind the title */
.hero::before {
  content: '';
  position: absolute;
  top: -60px; left: -80px;
  width: 340px; height: 340px;
  background: radial-gradient(circle, rgba(250,124,79,0.18) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
  animation: orb-drift 8s ease-in-out infinite alternate;
}
.hero::after {
  content: '';
  position: absolute;
  top: -20px; right: -60px;
  width: 260px; height: 260px;
  background: radial-gradient(circle, rgba(255,179,71,0.13) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
  animation: orb-drift 11s ease-in-out infinite alternate-reverse;
}
@keyframes orb-drift {
  from { transform: translate(0, 0) scale(1); }
  to   { transform: translate(30px, 20px) scale(1.08); }
}

.hero-inner { position: relative; z-index: 1; }

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: rgba(250,124,79,0.12);
  border: 1px solid rgba(250,124,79,0.30);
  border-radius: 99px;
  padding: 0.25rem 0.85rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: #fa7c4f !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.9rem;
}

.hero h1 {
  font-size: 2.8rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.04em !important;
  line-height: 1.1 !important;
  margin-bottom: 0.6rem !important;
  background: linear-gradient(135deg, #fdf4ec 0%, #ffb347 55%, #fa7c4f 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
}

.hero-sub {
  font-size: 1rem !important;
  color: #a8896e !important;
  line-height: 1.6;
  max-width: 520px;
  margin-bottom: 1.2rem;
}

.hero-features {
  display: flex;
  gap: 1.2rem;
  flex-wrap: wrap;
  margin-top: 0.2rem;
}
.hero-feat {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.8rem;
  color: #a8896e !important;
  font-weight: 500;
}
.hero-feat-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
}

/* ---------- export ---------- */
.export-meta {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
  font-size: 0.88rem;
}
.export-meta span { color: var(--muted) !important; }
.export-meta strong { color: var(--text) !important; }


</style>

""", unsafe_allow_html=True)

# -------------------------
# Hero header
# -------------------------
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-badge">✦ Powered by Gemini AI</div>
    <h1>AI Travel Planner</h1>
    <p class="hero-sub">
      Find flights & hotels, then let Gemini craft your complete
      day-by-day itinerary, packing list and local tips — in one click.
    </p>
    <div class="hero-features">
      <div class="hero-feat"><div class="hero-feat-dot"></div>Real-time flight prices</div>
      <div class="hero-feat"><div class="hero-feat-dot"></div>Live hotel availability</div>
      <div class="hero-feat"><div class="hero-feat-dot"></div>AI itinerary builder</div>
      <div class="hero-feat"><div class="hero-feat-dot"></div>Weather & visa info</div>
      <div class="hero-feat"><div class="hero-feat-dot"></div>Free to use</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Secrets / Clients
# -------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY"))
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

if not GEMINI_API_KEY or not SERPAPI_API_KEY:
    st.error("Missing API keys — set GEMINI_API_KEY and SERPAPI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("### ⚙ Preferences")

    st.markdown('<div class="sidebar-label">Trip style</div>', unsafe_allow_html=True)
    style = st.selectbox("Trip style", ["Balanced", "Foodie", "Culture", "Nature", "Shopping", "Luxury", "Budget"], index=0, label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">Budget</div>', unsafe_allow_html=True)
    max_flight_budget = st.number_input("Max flight price", min_value=0, max_value=10000, value=0, step=50,
        help="0 = no limit")
    max_hotel_budget  = st.number_input("Max hotel price/night", min_value=0, max_value=2000, value=0, step=10,
        help="0 = no limit")

    st.markdown('<div class="sidebar-label">Filters</div>', unsafe_allow_html=True)
    max_stops = st.slider("Max stops", 0, 2, 1)
    min_hotel_rating = st.slider("Min hotel rating ★", 1, 5, 3, 1)

    st.markdown('<div class="sidebar-label">Sort</div>', unsafe_allow_html=True)
    flight_sort = st.selectbox("Sort flights", ["Cheapest", "Fastest", "Fewest stops"], index=0)
    hotel_sort = st.selectbox("Sort hotels", ["Top rated", "Cheapest"], index=0)

    st.markdown('<div class="sidebar-label">Currency</div>', unsafe_allow_html=True)
    currency = st.selectbox("Currency", ["USD", "SGD", "MYR", "JPY", "EUR", "GBP"], index=0, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">How it works</div>', unsafe_allow_html=True)
    st.caption("1. Enter cities & dates below\n2. Hit **Search** or **Plan My Entire Trip**\n3. Select flights & hotels\n4. Generate AI picks & itinerary")

# -------------------------
# Route inputs
# -------------------------
from datetime import date, timedelta

_today      = date.today()
_default_out = _today + timedelta(days=30)
_default_ret = _today + timedelta(days=37)

# Restore saved dates from session state
_saved_out = st.session_state.get("outbound_date", "")
_saved_ret = st.session_state.get("return_date", "")
try:
    _default_out = date.fromisoformat(_saved_out) if _saved_out else _default_out
except Exception:
    pass
try:
    _default_ret = date.fromisoformat(_saved_ret) if _saved_ret else _default_ret
except Exception:
    pass

# Row 1: origin + destination | outbound + return dates
# On desktop: 4 columns in one row. CSS stacks to 2×2 on mobile.
r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])

_pill_style = ("font-size:0.73rem;padding:0.22rem 0.55rem;"
               "background:rgba(250,124,79,0.10);border:1px solid rgba(250,124,79,0.25);"
               "border-radius:6px;color:#fa7c4f;display:inline-block;margin-top:2px;")

def _render_airport_input(col, label, state_key, state_iata_key, default_city, default_iata):
    """
    Renders a city text input. If the city has multiple airports, shows a
    compact selectbox to pick one. Returns the final resolved IATA code.
    """
    raw = col.text_input(label, value=st.session_state.get(state_key, default_city),
                         placeholder="City, e.g. Bangkok").strip()
    st.session_state[state_key] = raw

    options = get_airport_options(raw)

    if options:
        # Multiple airports — show selectbox
        opt_labels = [lbl for _, lbl in options]
        opt_iatas  = [iata for iata, _ in options]
        # Restore previous selection if same city
        prev_iata = st.session_state.get(state_iata_key, opt_iatas[0])
        prev_idx  = opt_iatas.index(prev_iata) if prev_iata in opt_iatas else 0
        chosen_label = col.selectbox(
            f"Airport for {raw}",
            options=opt_labels,
            index=prev_idx,
            key=f"airport_sel_{state_key}",
            label_visibility="collapsed",
        )
        chosen_iata = opt_iatas[opt_labels.index(chosen_label)]
        st.session_state[state_iata_key] = chosen_iata
        col.markdown(
            f'<div style="{_pill_style}">✈ {chosen_iata}  ·  {raw}</div>',
            unsafe_allow_html=True,
        )
        return chosen_iata
    else:
        # Single airport — auto-resolve and show pill
        iata = resolve_to_iata(raw) if raw else default_iata
        st.session_state[state_iata_key] = iata
        if raw:
            resolved = iata != raw.upper()
            pill_text = f"✈ {iata}  ·  {raw}" if resolved else f"✈ {iata}"
            col.markdown(
                f'<div style="{_pill_style}">{pill_text}</div>',
                unsafe_allow_html=True,
            )
        return iata

origin      = _render_airport_input(r1c1, "From", "origin_city",      "origin_iata",      "Singapore", "SIN")
destination = _render_airport_input(r1c2, "To",   "destination_city", "destination_iata", "Tokyo",     "NRT")
outbound_date_obj = r1c3.date_input("Outbound date", value=_default_out, min_value=_today, format="YYYY-MM-DD")
return_date_obj   = r1c4.date_input("Return date",   value=_default_ret, min_value=_default_out, format="YYYY-MM-DD")

outbound_date = outbound_date_obj.strftime("%Y-%m-%d")
return_date   = return_date_obj.strftime("%Y-%m-%d")

# Row 2: hotel location
location = st.text_input(
    "Hotel location (optional — defaults to destination)",
    value=st.session_state.get("location", ""),
    placeholder="e.g. Tokyo, Shinjuku"
).strip()

col_s1, col_s2, col_s3 = st.columns([1.1, 1.1, 0.8])
search_clicked   = col_s1.button("🔍  Search Flights & Hotels", type="primary", use_container_width=True)
plan_clicked     = col_s2.button("✨  Plan My Entire Trip", type="secondary", use_container_width=True)
surprise_clicked = col_s3.button("🎲  Surprise Me!", use_container_width=True)

# -------------------------
# Search
# -------------------------
def _do_search(origin, destination, out_d, ret_d, location, currency,
               min_hotel_rating, max_flight_budget, max_hotel_budget,
               flight_sort, hotel_sort):
    """Shared search logic used by both Search and Plan buttons."""
    hotel_loc = location or destination

    async def fetch_all():
        flights_raw, hotels_raw = await asyncio.gather(
            get_flights(SERPAPI_API_KEY, origin, destination, out_d, ret_d, currency, client, GEMINI_MODEL),
            get_hotels(SERPAPI_API_KEY, hotel_loc, out_d, ret_d, currency, min_hotel_rating),
        )
        return flights_raw, hotels_raw

    flights_raw, hotels_raw = run_async(fetch_all())
    flights    = sort_flights(fmt_flights(flights_raw, origin, destination, out_d, ret_d), flight_sort)
    _city_name = st.session_state.get("destination_city", "")
    hotels_fmt = sort_hotels(fmt_hotels(hotels_raw, hotel_loc, out_d, ret_d, _city_name), hotel_sort)

    if max_flight_budget > 0:
        flights = [f for f in flights if _parse_price(f.get("price")) <= max_flight_budget]
    if max_hotel_budget > 0:
        hotels_fmt = [h for h in hotels_fmt if _parse_price(h.get("price")) <= max_hotel_budget]
    # expose raw for filter-stability re-sort
    _do_search.last_raw = (flights_raw, hotels_raw)

    st.session_state.update({
        "origin": origin, "destination": destination,
        "outbound_date": out_d, "return_date": ret_d,
        "location": location, "hotel_loc": hotel_loc,
        "flights": flights, "hotels": hotels_fmt,
        "flights_raw_cache": flights_raw,
        "hotels_raw_cache":  hotels_raw,
        "max_flight_budget": max_flight_budget,
        "max_hotel_budget":  max_hotel_budget,
        "selected_flights": [], "selected_hotels": [],
        "ai_flights_md": "", "ai_hotels_md": "",
        "itinerary_md": "", "tips_md": "", "insights_md": "",
        "packing_md": "", "visa_md": "", "weather": {},
        "price_calendar": {},
    })
    return flights, hotels_fmt, hotel_loc


# ── FILTER STABILITY: re-apply sort/budget to cached raw results ────────────
# This prevents sidebar changes from wiping results (no new API call needed)
if not (search_clicked or plan_clicked) and st.session_state.get("flights_raw_cache"):
    _cached = st.session_state["flights_raw_cache"]
    _cached_h = st.session_state.get("hotels_raw_cache", [])
    _co = st.session_state.get("origin", origin)
    _cd = st.session_state.get("destination", destination)
    _cod = st.session_state.get("outbound_date", outbound_date)
    _crd = st.session_state.get("return_date", return_date)
    _cloc = st.session_state.get("hotel_loc", destination)
    _flights_re = sort_flights(fmt_flights(_cached, _co, _cd, _cod, _crd), flight_sort)
    _city_nm = st.session_state.get("destination_city", "")
    _hotels_re  = sort_hotels(fmt_hotels(_cached_h, _cloc, _cod, _crd, _city_nm), hotel_sort)
    if max_flight_budget > 0:
        _flights_re = [f for f in _flights_re if _parse_price(f.get("price")) <= max_flight_budget]
    if max_hotel_budget > 0:
        _hotels_re = [h for h in _hotels_re if _parse_price(h.get("price")) <= max_hotel_budget]
    st.session_state["flights"]    = _flights_re
    st.session_state["hotels"]     = _hotels_re

# ── SURPRISE ME ─────────────────────────────────────────────────────────────
if surprise_clicked:
    surprise_prompt = f"""You are a travel inspiration AI.
The user is departing from {origin or "Singapore (SIN)"} on {outbound_date} returning {return_date}.
Their trip style is: {style}. Budget level: {'tight' if max_flight_budget and max_flight_budget < 500 else 'moderate' if max_flight_budget and max_flight_budget < 1500 else 'open'}.

Pick ONE surprising, specific destination that perfectly fits this traveller.
Respond ONLY in this exact JSON format (no markdown, no explanation outside the JSON):
{{"iata": "NRT", "city": "Tokyo", "country": "Japan", "tagline": "one evocative sentence about why this destination is perfect for them", "highlights": ["thing1", "thing2", "thing3"]}}"""

    with st.spinner("🎲 Finding your perfect surprise destination..."):
        raw_surprise = run_async(gemini_call(client, GEMINI_MODEL, surprise_prompt))

    import json as _json
    try:
        # strip any accidental markdown fences
        _clean = raw_surprise.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        _s = _json.loads(_clean)
        st.session_state["surprise"] = _s
    except Exception:
        st.session_state["surprise"] = None
        st.warning("Couldn't parse surprise destination. Try again!")

if st.session_state.get("surprise") and not (search_clicked or plan_clicked):
    _s = st.session_state["surprise"]
    st.markdown(f"""
    <div style="background:rgba(250,124,79,0.10);border:1px solid rgba(250,124,79,0.30);
    border-radius:14px;padding:1rem 1.3rem;margin-bottom:1rem;">
      <div style="font-size:1.1rem;font-weight:700;color:#fa7c4f !important;margin-bottom:0.3rem;">
        🎲 Your surprise destination: {_s.get('city','?')}, {_s.get('country','?')} ({_s.get('iata','?')})
      </div>
      <div style="font-size:0.9rem;color:#fdf4ec !important;margin-bottom:0.5rem;">{_s.get('tagline','')}</div>
      <div style="font-size:0.82rem;color:#a8896e !important;">✨ {"&nbsp;&nbsp;·&nbsp;&nbsp;".join(_s.get("highlights",[]))}</div>
      <div style="font-size:0.78rem;color:#a8896e !important;margin-top:0.4rem;">
        👆 Update Destination to <strong>{_s.get('iata','?')}</strong> then hit Search or Plan My Entire Trip
      </div>
    </div>
    """, unsafe_allow_html=True)

if search_clicked or plan_clicked:
    try:
        out_d = outbound_date
        ret_d = return_date

        if return_date_obj <= outbound_date_obj:
            st.error("Return date must be after outbound date.")
            st.stop()
        if origin and destination and origin == destination:
            st.error("Origin and destination cannot be the same.")
            st.stop()

        with st.spinner("Searching flights & hotels..."):
            flights, hotels_fmt, hotel_loc = _do_search(
                origin, destination, out_d, ret_d, location, currency,
                min_hotel_rating, max_flight_budget, max_hotel_budget,
                flight_sort, hotel_sort,
            )

        if plan_clicked and (flights or hotels_fmt):
            # ── ONE-CLICK: run all AI in parallel ──────────────────
            _ft = fmt_flights_for_ai(flights[:6])
            _ht = fmt_hotels_for_ai(hotels_fmt[:6])
            _dest = destination
            _nights = (return_date_obj - outbound_date_obj).days

            fp = f"""You are an expert travel assistant.
Route: {origin} → {_dest} | Dates: {out_d} to {ret_d}
Trip style: {style} | Max stops: {max_stops} | Currency: {currency}
Budget: flights ≤ {'any' if max_flight_budget==0 else f'{currency} {max_flight_budget}'}

Available flights:
{_ft}

Recommend the best 1-2 flights with 2-3 bullet reasons each. Include booking links."""

            hp = f"""You are an expert travel assistant.
Location: {hotel_loc} | Dates: {out_d} to {ret_d}
Trip style: {style} | Min rating: {min_hotel_rating}★ | Currency: {currency}
Budget: hotels ≤ {'any' if max_hotel_budget==0 else f'{currency} {max_hotel_budget}'}/night

Available hotels:
{_ht}

Recommend the best 2-3 hotels with 2-3 bullet reasons each. Include listing links."""

            ip = f"""Create a practical {_nights}-day Markdown itinerary for {_dest} ({out_d} to {ret_d}).
Trip style: {style}. For each day: Morning/Afternoon/Evening. Include 2-3 food ideas per day."""

            tp = f"""Concise Markdown travel tips for {_dest}:
- Best areas to stay (2-4)
- Getting around
- Budget tips
- 5 tourist mistakes to avoid
- 5 local etiquette tips"""

            ins_p = f"""Travel review summary for {_dest} in Markdown:
### 🌟 Overall Vibe
### ❤️ What Travellers Love
### 😬 Common Complaints
### 🗓 Best Time to Visit
### 💡 Insider Tips
Be specific and honest."""

            pack_p = f"""Generate a smart packing list in Markdown for:
- Destination: {_dest}
- Trip duration: {_nights} nights
- Trip style: {style}
- Dates: {out_d} to {ret_d}

Sections: 👕 Clothing, 🧴 Toiletries, 💊 Health & Safety, 📱 Tech & Documents, 🎒 Day-trip Essentials.
Keep it practical, not exhaustive. Flag destination-specific items (e.g. temple dress code, mosquito repellent)."""

            visa_p = f"""Provide a concise Markdown visa & entry requirements summary for travellers going to {_dest}.
Include:
### 🛂 Visa Overview
- Common nationalities that get visa-free / visa-on-arrival
- Who needs to apply in advance
### 📋 Entry Requirements
- Passport validity rules
- Common documents needed
- Health / vaccination requirements if any
### ⚠️ Important Notes
- Any recent policy changes worth flagging

Keep it factual. Note that requirements vary by nationality and travellers should verify with official embassy sources."""

            async def run_plan_all():
                results = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, fp),
                    gemini_call(client, GEMINI_MODEL, hp),
                    gemini_call(client, GEMINI_MODEL, ip),
                    gemini_call(client, GEMINI_MODEL, tp),
                    gemini_call(client, GEMINI_MODEL, ins_p),
                    gemini_call(client, GEMINI_MODEL, pack_p),
                    gemini_call(client, GEMINI_MODEL, visa_p),
                )
                return results

            _progress_bar  = st.progress(0, text="✨ Starting your trip plan...")
            _status        = st.empty()

            _status.markdown("⏳ **Step 1/4** — Analysing flights & hotels...")
            _progress_bar.progress(10, text="Analysing options...")

            async def run_plan_all_phased():
                # Phase 1: flight + hotel picks (fast, small prompts)
                f_md, h_md = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, fp),
                    gemini_call(client, GEMINI_MODEL, hp),
                )
                return f_md, h_md

            f_md, h_md = run_async(run_plan_all_phased())
            _progress_bar.progress(35, text="Building itinerary...")
            _status.markdown("⏳ **Step 2/4** — Crafting your day-by-day itinerary...")

            async def run_plan_phase2():
                i_md, t_md = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, ip),
                    gemini_call(client, GEMINI_MODEL, tp),
                )
                return i_md, t_md

            i_md, t_md = run_async(run_plan_phase2())
            _progress_bar.progress(65, text="Researching destination...")
            _status.markdown("⏳ **Step 3/4** — Researching destination insights...")

            async def run_plan_phase3():
                ins_md, wx_data = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, ins_p),
                    get_weather(destination, out_d, ret_d),
                )
                return ins_md, wx_data

            ins_md, wx_data = run_async(run_plan_phase3())
            _progress_bar.progress(85, text="Generating essentials...")
            _status.markdown("⏳ **Step 4/4** — Packing list & visa info...")

            async def run_plan_phase4():
                p_md, v_md = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, pack_p),
                    gemini_call(client, GEMINI_MODEL, visa_p),
                )
                return p_md, v_md

            pack_md, visa_md = run_async(run_plan_phase4())
            _progress_bar.progress(100, text="Done!")
            _status.empty()
            _progress_bar.empty()

            st.session_state.update({
                "ai_flights_md": f_md, "ai_hotels_md": h_md,
                "itinerary_md": i_md,  "tips_md": t_md,
                "insights_md": ins_md,
                "packing_md": pack_md, "visa_md": visa_md,
                "weather": wx_data,
            })
            st.success("✅ Your full trip plan is ready! Browse the tabs below.")

    except Exception as e:
        st.error(str(e))

# -------------------------
# Results
# -------------------------
flights    = st.session_state.get("flights", [])
hotels_fmt = st.session_state.get("hotels", [])

if flights or hotels_fmt:
    # --- Trip summary bar ---
    _o  = st.session_state.get("origin", "")
    _d  = st.session_state.get("destination", "")
    _od = st.session_state.get("outbound_date", "")
    _rd = st.session_state.get("return_date", "")
    try:
        _nights = (date.fromisoformat(_rd) - date.fromisoformat(_od)).days
        _nights_str = f"{_nights} night{'s' if _nights != 1 else ''}"
    except Exception:
        _nights_str = ""
    _nf = len(flights)
    _nh = len(hotels_fmt)
    st.markdown(f"""
    <div style="
      background: rgba(250,124,79,0.08);
      border: 1px solid rgba(250,124,79,0.22);
      border-radius: 12px;
      padding: 0.7rem 1.2rem;
      display: flex;
      align-items: center;
      gap: 1.5rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    ">
      <span style="font-size:1.05rem;font-weight:600;color:#fa7c4f !important;letter-spacing:-0.01em;">
        {st.session_state.get("origin_city", _o)} → {st.session_state.get("destination_city", _d)}
        <span style="font-size:0.75rem;font-weight:400;color:#a8896e !important;margin-left:4px;">({_o} → {_d})</span>
      </span>
      <span style="color:#a8896e !important;font-size:0.85rem;">📅 {_od} → {_rd} &nbsp;·&nbsp; {_nights_str}</span>
      <span style="color:#a8896e !important;font-size:0.85rem;">✈ {_nf} flight{'s' if _nf != 1 else ''} found</span>
      <span style="color:#a8896e !important;font-size:0.85rem;">🏨 {_nh} hotel{'s' if _nh != 1 else ''} found</span>
      {'<span style="color:#a8896e !important;font-size:0.85rem;">💰 Budget filtered</span>' if (st.session_state.get("max_flight_budget",0) or st.session_state.get("max_hotel_budget",0)) else ''}
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "✈  Flights", "🏨  Hotels", "✦  AI Picks",
        "🗺  Itinerary", "🌍  Destination", "🎒  Essentials"
    ])

    # ==============================
    # TAB 1 — Flights
    # ==============================
    with tab1:
        if not flights:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">✈️</div>
              <div class="empty-title">No flights found</div>
              <div class="empty-sub">Try different dates, route, or check the city/airport names.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # ── PRICE CALENDAR ───────────────────────────────────
            with st.expander("📅 Price Calendar — cheapest days to fly this month", expanded=False):
                _pc_origin = st.session_state.get("origin", origin)
                _pc_dest   = st.session_state.get("destination", destination)
                _pc_out    = st.session_state.get("outbound_date", outbound_date)
                _pc_curr   = currency
                try:
                    _trip_len = (date.fromisoformat(st.session_state.get("return_date", return_date)) -
                                 date.fromisoformat(_pc_out)).days
                except Exception:
                    _trip_len = 7

                if st.button("Load Price Calendar", key="btn_price_cal"):
                    # Sample 14 dates: selected date ± 7 days
                    from datetime import date as _date, timedelta as _td
                    _base = _date.fromisoformat(_pc_out)
                    _sample_dates = [
                        (_base + _td(days=i)).strftime("%Y-%m-%d")
                        for i in range(-6, 8)
                        if (_base + _td(days=i)) >= _date.today()
                    ]

                    async def fetch_calendar():
                        tasks = [
                            get_price_for_date(SERPAPI_API_KEY, _pc_origin, _pc_dest, d, _trip_len, _pc_curr)
                            for d in _sample_dates
                        ]
                        return await asyncio.gather(*tasks)

                    with st.spinner(f"Checking prices for {len(_sample_dates)} dates..."):
                        _cal_results = run_async(fetch_calendar())

                    _cal_dict = {d: p for d, p in _cal_results if p is not None}
                    st.session_state["price_calendar"] = _cal_dict

                _cal = st.session_state.get("price_calendar", {})
                if _cal:
                    _prices = [v for v in _cal.values() if v]
                    _min_p  = min(_prices) if _prices else 0
                    _max_p  = max(_prices) if _prices else 1
                    _range  = max(_max_p - _min_p, 1)

                    _cal_cols = st.columns(min(len(_cal), 4))
                    for _ci, (_dt, _pr) in enumerate(sorted(_cal.items())):
                        _col_i = _ci % 7
                        with _cal_cols[_col_i]:
                            _ratio = (_pr - _min_p) / _range
                            # green = cheap, red = expensive
                            _r = int(80  + _ratio * 175)
                            _g = int(200 - _ratio * 130)
                            _b = 80
                            _is_selected = _dt == _pc_out
                            _border = "2px solid #fa7c4f" if _is_selected else "1px solid rgba(255,200,150,0.15)"
                            st.markdown(f"""
                            <div style="text-align:center;background:rgba({_r},{_g},{_b},0.18);
                            border:{_border};border-radius:10px;padding:0.4rem 0.1rem;margin-bottom:0.3rem;">
                              <div style="font-size:0.65rem;color:#a8896e !important;">{_dt[5:]}</div>
                              <div style="font-size:0.82rem;font-weight:600;color:#fdf4ec !important;">{_pc_curr} {_pr:,.0f}</div>
                              {'<div style="font-size:0.6rem;color:#38d96a !important;">★ cheapest</div>' if _pr == _min_p else ''}
                            </div>""", unsafe_allow_html=True)
                elif not st.session_state.get("price_calendar") is None:
                    st.caption("Click 'Load Price Calendar' to see prices across nearby dates.")

            labels = [f"{i+1}. {f['airline']}  ·  {f['price']}  ·  {f['stops']}  ·  {f['depart_code']} {f['depart_time']} → {f['arrive_code']} {f['arrive_time']}" for i, f in enumerate(flights)]
            selected = st.multiselect("Select up to 3 flights to personalise AI picks", options=labels, default=st.session_state.get("selected_flights", []))
            selected, clipped = clamp_multiselect(selected, 3)
            if clipped:
                st.warning("Maximum 3 flights — keeping the first 3 selected.")
            st.session_state["selected_flights"] = selected

            st.markdown("")
            for i, f in enumerate(flights[:10]):
                logo_url = f.get("airline_logo", "")
                dur_str  = fmt_duration(f.get("duration_min"))
                nonstop  = "nonstop" in f.get("stops", "").lower()
                dep_code = f.get("depart_code", "—")
                dep_time = f.get("depart_time", "")
                arr_code = f.get("arrive_code", "—")
                arr_time = f.get("arrive_time", "")
                link = f.get("link") or google_flights_search_url(
                    st.session_state.get("origin", origin),
                    st.session_state.get("destination", destination),
                    st.session_state.get("outbound_date", ""),
                    st.session_state.get("return_date", ""),
                )
                logo_html = f'<img class="airline-logo" src="{logo_url}">' if logo_url else "✈"
                stop_pill = '<span class="pill pill-green">Nonstop</span>' if nonstop else f'<span class="pill pill-warn">{f["stops"]}</span>'
                class_pill = f'<span class="pill pill-blue">{f["class"]}</span>'

                st.markdown(f"""
                <div class="flight-card">
                  <div class="flight-header">
                    <div class="flight-airline">
                      {logo_html}
                      <div>
                        <div class="airline-name">{f['airline']}</div>
                        <div class="pill-row" style="margin-top:4px;">{class_pill}{stop_pill}</div>
                      </div>
                    </div>
                    <div class="flight-price-block">
                      <div class="flight-price">{f['price']}</div>
                      <div class="flight-price-sub">per person · one way</div>
                    </div>
                  </div>
                  <div class="route-line">
                    <div>
                      <div class="iata">{dep_code}</div>
                      <div class="time-small">{dep_time}</div>
                    </div>
                    <div class="route-mid">
                      <div class="route-dur">⏱ {dur_str}</div>
                      <div class="dot-line"></div>
                    </div>
                    <div style="text-align:right;">
                      <div class="iata">{arr_code}</div>
                      <div class="time-small">{arr_time}</div>
                    </div>
                  </div>
                  <div class="flight-footer">
                    <a class="link-btn" href="{link}" target="_blank">Open in Google Flights →</a>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ==============================
    # TAB 2 — Hotels
    # ==============================
    with tab2:
        if not hotels_fmt:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🏨</div>
              <div class="empty-title">No hotels found</div>
              <div class="empty-sub">Try lowering the minimum rating, or change the hotel location.</div>
            </div>""", unsafe_allow_html=True)
        else:
            def _hotel_label(i, h):
                hc = h.get("hotel_class")
                stars = ("★" * hc) if hc else "?"
                rs = h.get("review_score", h.get("rating", 0))
                return f"{i+1}. {h['name']}  ·  {h['price']}  ·  {stars} ({rs:.1f}/10)"
            labels = [_hotel_label(i, h) for i, h in enumerate(hotels_fmt)]
            selected = st.multiselect("Select up to 3 hotels to personalise AI picks", options=labels, default=st.session_state.get("selected_hotels", []))
            selected, clipped = clamp_multiselect(selected, 3)
            if clipped:
                st.warning("Maximum 3 hotels — keeping the first 3 selected.")
            st.session_state["selected_hotels"] = selected

            st.markdown("")
            for i, h in enumerate(hotels_fmt[:12]):
                hc = h.get("hotel_class")
                rs = h.get("review_score", h.get("rating", 0.0))
                reviews_txt = f"· {h['reviews_count']} reviews" if h.get("reviews_count") else ""
                prop_type = h.get("type", "")
                link = h.get("link") or google_hotels_search_url(
                    st.session_state.get("location", "") or destination,
                    st.session_state.get("outbound_date", ""),
                    st.session_state.get("return_date", ""),
                )
                if hc:
                    class_block = f'<span class="hotel-class-stars">{"★" * hc}{"☆" * (5 - hc)}</span><span class="hotel-class-label">{hc}-star hotel</span>'
                else:
                    class_block = '<span class="hotel-class-label">Unclassified</span>'

                if rs > 0:
                    score_cls = "green" if rs >= 8 else "amber" if rs >= 6 else "salmon"
                    score_block = f'<span class="hotel-score-badge {score_cls}">{rs:.1f}/10 <span style="font-weight:400;font-size:0.65rem;">{reviews_txt}</span></span>'
                else:
                    score_block = '<span style="font-size:0.75rem;color:var(--muted);">No reviews</span>'

                type_pill = f'<span class="pill" style="margin-left:6px;">{prop_type}</span>' if prop_type else ""
                dist_html = f'<span style="margin-left:auto;font-size:0.78rem;color:var(--muted);">📍 {h["distance"]}</span>' if h.get("distance") else ""
                chips_html = ""
                if h.get("amenities"):
                    chips_html = '<div style="margin-top:0.6rem;">' + "".join(f'<span class="amenity-chip">{a}</span>' for a in h["amenities"]) + "</div>"

                st.markdown(f"""
                <div class="hotel-card">
                  <div class="hotel-header">
                    <div><div class="hotel-name">{i+1}. {h['name']}{type_pill}</div></div>
                    <div class="hotel-price-block">
                      <div class="hotel-price">{h['price']}</div>
                      <div class="hotel-price-sub">per night</div>
                    </div>
                  </div>
                  <div class="hotel-ratings">
                    {class_block}
                    <span style="color:rgba(255,200,150,0.2);">|</span>
                    {score_block}
                  </div>
                  <div class="hotel-location">📍 {h['area']}{dist_html}</div>
                  {chips_html}
                  <div style="margin-top:0.75rem;">
                    <a class="link-btn" href="{link}" target="_blank">View hotel listing →</a>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ==============================
    # TAB 3 — AI Picks
    # ==============================
    with tab3:
        origin2    = st.session_state.get("origin", origin)
        dest2      = st.session_state.get("destination", destination)
        out_d2     = st.session_state.get("outbound_date", outbound_date)
        ret_d2     = st.session_state.get("return_date", return_date)
        hotel_loc2 = (st.session_state.get("location", "") or dest2).strip()

        flights_for_ai = flights[:6]
        hotels_for_ai  = hotels_fmt[:6]

        if st.session_state.get("selected_flights"):
            idxs = []
            for s in st.session_state["selected_flights"]:
                try: idxs.append(int(s.split(".")[0]) - 1)
                except: pass
            flights_for_ai = [flights[i] for i in idxs if 0 <= i < len(flights)] or flights_for_ai

        if st.session_state.get("selected_hotels"):
            idxs = []
            for s in st.session_state["selected_hotels"]:
                try: idxs.append(int(s.split(".")[0]) - 1)
                except: pass
            hotels_for_ai = [hotels_fmt[i] for i in idxs if 0 <= i < len(hotels_fmt)] or hotels_for_ai

        has_ai = st.session_state.get("ai_flights_md") or st.session_state.get("ai_hotels_md")

        if not has_ai:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">✦</div>
              <div class="empty-title">AI picks not generated yet</div>
              <div class="empty-sub">Select flights & hotels in the tabs above, then hit the button below.</div>
            </div>""", unsafe_allow_html=True)

        if st.button("Generate AI Picks", type="primary"):
            _flights_text = fmt_flights_for_ai(flights_for_ai)
            _hotels_text  = fmt_hotels_for_ai(hotels_for_ai)

            flight_prompt = f"""You are an expert travel assistant helping a traveller plan a trip.

Trip details:
- Route: {origin2} → {dest2}
- Dates: {out_d2} to {ret_d2}
- Trip style: {style}
- Max stops preferred: {max_stops}
- Currency: {currency}

Available flights:
{_flights_text}

Task: Recommend the best 1-2 flights. For each pick explain WHY in 2-3 bullet points covering price, duration, stops and convenience. Include the booking link.""".strip()

            hotel_prompt = f"""You are an expert travel assistant helping a traveller plan a trip.

Trip details:
- Location: {hotel_loc2}
- Dates: {out_d2} to {ret_d2}
- Trip style: {style}
- Min hotel quality: {min_hotel_rating}★ out of 5
- Currency: {currency}

Available hotels:
{_hotels_text}

Task: Recommend the best 2-3 hotels. For each pick explain WHY in 2-3 bullet points covering price, rating, location and highlights. Include the listing link.""".strip()

            async def run_ai_picks():
                f_md, h_md = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, flight_prompt),
                    gemini_call(client, GEMINI_MODEL, hotel_prompt),
                )
                return f_md, h_md

            with st.spinner("Asking Gemini..."):
                f_md, h_md = run_async(run_ai_picks())
                st.session_state["ai_flights_md"] = f_md
                st.session_state["ai_hotels_md"]  = h_md

        if st.session_state.get("ai_flights_md"):
            st.markdown('''
            <div class="ai-result-header">
              <div class="ai-result-icon flight">✈</div>
              <div><div class="ai-result-title">Recommended Flights</div>
              <div class="ai-result-sub">Gemini\'s top picks based on price, duration &amp; stops</div></div>
            </div>''', unsafe_allow_html=True)
            st.markdown(st.session_state["ai_flights_md"])

        if st.session_state.get("ai_hotels_md"):
            st.markdown('''
            <div class="ai-result-header" style="margin-top:1.5rem;">
              <div class="ai-result-icon hotel">🏨</div>
              <div><div class="ai-result-title">Recommended Hotels</div>
              <div class="ai-result-sub">Gemini\'s top picks based on rating, location &amp; value</div></div>
            </div>''', unsafe_allow_html=True)
            st.markdown(st.session_state["ai_hotels_md"])

    # ==============================
    # TAB 4 — Itinerary & Export
    # ==============================
    with tab4:
        dest2  = st.session_state.get("destination", destination)
        out_d2 = st.session_state.get("outbound_date", outbound_date)
        ret_d2 = st.session_state.get("return_date", return_date)

        # ── WEATHER WIDGET ────────────────────────────────────────
        wx = st.session_state.get("weather", {})
        if not wx:
            with st.spinner("Fetching weather forecast..."):
                wx = run_async(get_weather(dest2, out_d2, ret_d2))
                st.session_state["weather"] = wx

        if wx and wx.get("daily"):
            st.markdown(f"#### 🌤 Weather in {wx.get('city', dest2)} during your trip")
            st.markdown('<div class="wx-scroll-wrap">', unsafe_allow_html=True)
            wx_cols = st.columns(min(len(wx["daily"]), 7))
            for ci, day in enumerate(wx["daily"][:7]):
                with wx_cols[ci]:
                    d_label = day["date"][5:]  # MM-DD
                    icon    = day.get("icon", "🌡")
                    tmax    = day.get("tmax")
                    tmin    = day.get("tmin")
                    precip  = day.get("precip")
                    temp_str  = f"{tmax:.0f}°" if tmax is not None else "—"
                    tmin_str  = f"{tmin:.0f}°" if tmin is not None else ""
                    precip_str = f"💧{precip}%" if precip is not None else ""
                    st.markdown(f"""
                    <div style="text-align:center;background:rgba(255,255,255,0.04);
                    border:1px solid rgba(255,200,150,0.12);border-radius:10px;padding:0.5rem 0.2rem;">
                      <div style="font-size:0.7rem;color:#a8896e !important;">{d_label}</div>
                      <div style="font-size:1.4rem;">{icon}</div>
                      <div style="font-size:0.9rem;font-weight:600;">{temp_str}</div>
                      <div style="font-size:0.7rem;color:#a8896e !important;">{tmin_str}</div>
                      <div style="font-size:0.68rem;color:#a8896e !important;">{precip_str}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("")

        has_itinerary = st.session_state.get("itinerary_md") or st.session_state.get("tips_md")

        if not has_itinerary:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🗺</div>
              <div class="empty-title">No itinerary yet</div>
              <div class="empty-sub">Generate your AI picks first, then create the full itinerary here.</div>
            </div>""", unsafe_allow_html=True)

        if st.button("Generate Itinerary & Tips", type="primary"):
            itinerary_prompt = f"""Create a practical day-by-day itinerary in Markdown for {dest2}
from {out_d2} to {ret_d2}.

Trip style: {style}

For each day: Morning / Afternoon / Evening.
Include 2-3 food ideas per day.
Keep it realistic and not overly packed.""".strip()

            tips_prompt = f"""Give concise travel tips in Markdown for {dest2}:
- Best areas to stay (2-4 options)
- Getting around (local transit + passes)
- Budget tips (specific)
- 5 tourist mistakes to avoid
- 5 local etiquette tips""".strip()

            async def run_itinerary():
                i_md, t_md = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, itinerary_prompt),
                    gemini_call(client, GEMINI_MODEL, tips_prompt),
                )
                return i_md, t_md

            with st.spinner("Generating itinerary & tips..."):
                i_md, t_md = run_async(run_itinerary())
                st.session_state["itinerary_md"] = i_md
                st.session_state["tips_md"]       = t_md

        if st.session_state.get("itinerary_md"):
            st.markdown('''
            <div class="ai-result-header">
              <div class="ai-result-icon map">🗓</div>
              <div><div class="ai-result-title">Day-by-Day Itinerary</div>
              <div class="ai-result-sub">Your personalised travel plan</div></div>
            </div>''', unsafe_allow_html=True)
            st.markdown(st.session_state["itinerary_md"])

        if st.session_state.get("tips_md"):
            st.markdown('''
            <div class="ai-result-header" style="margin-top:1.5rem;">
              <div class="ai-result-icon flight">💡</div>
              <div><div class="ai-result-title">Insider Tips</div>
              <div class="ai-result-sub">Local knowledge &amp; practical advice</div></div>
            </div>''', unsafe_allow_html=True)
            st.markdown(st.session_state["tips_md"])

        # ---------- Export ----------
        st.markdown("---")
        st.markdown("#### Export your trip")

        o  = st.session_state.get("origin", "")
        d  = st.session_state.get("destination", "")
        od = st.session_state.get("outbound_date", "")
        rd = st.session_state.get("return_date", "")

        st.markdown(f"""
        <div class="export-meta">
          <span>Route</span> &nbsp;<strong>{o} → {d}</strong>
          &nbsp;&nbsp;·&nbsp;&nbsp;
          <span>Dates</span> &nbsp;<strong>{od} → {rd}</strong>
          &nbsp;&nbsp;·&nbsp;&nbsp;
          <span>Style</span> &nbsp;<strong>{style}</strong>
          &nbsp;&nbsp;·&nbsp;&nbsp;
          <span>Currency</span> &nbsp;<strong>{currency}</strong>
        </div>
        """, unsafe_allow_html=True)

        export_md = f"""# AI Travel Plan — {o} → {d}

**Dates:** {od} → {rd}
**Trip style:** {style} · **Max stops:** {max_stops} · **Min hotel stars:** {min_hotel_rating}★ · **Currency:** {currency}

---

## ✈ AI Flight Picks
{st.session_state.get("ai_flights_md", "_(not generated)_")}

---

## 🏨 AI Hotel Picks
{st.session_state.get("ai_hotels_md", "_(not generated)_")}

---

## 🗓 Itinerary
{st.session_state.get("itinerary_md", "_(not generated)_")}

---

## 💡 Travel Tips
{st.session_state.get("tips_md", "_(not generated)_")}

---

## 🌍 Destination Insights
{st.session_state.get("insights_md", "_(not generated)_")}

---

## 🎒 Packing List
{st.session_state.get("packing_md", "_(not generated)_")}

---

## 🛂 Visa & Entry
{st.session_state.get("visa_md", "_(not generated)_")}
"""
        st.download_button(
            "⬇  Download as Markdown",
            data=export_md,
            file_name=f"travel_plan_{o}_{d}_{od}.md",
            mime="text/markdown",
        )

    # ==============================
    # TAB 5 — Destination Insights
    # ==============================
    with tab5:
        dest2 = st.session_state.get("destination", "")

        has_insights = st.session_state.get("insights_md")

        if not has_insights:
            st.markdown(f"""
            <div class="empty-state">
              <div class="empty-icon">🌍</div>
              <div class="empty-title">Discover {dest2}</div>
              <div class="empty-sub">Get AI-powered local insights — sentiment, highlights, tips and common pitfalls.</div>
            </div>""", unsafe_allow_html=True)

        if st.button("Generate Destination Insights", type="primary", key="btn_insights"):
            insights_prompt = f"""You are a travel review analyst with deep knowledge of destinations worldwide.

Summarise traveller experience for **{dest2}** in Markdown with these sections:

### 🌟 Overall Vibe
2-3 bullets: general atmosphere, pace, who it suits best.

### ❤️ What Travellers Love
5-6 bullets: food scene, landmarks, nature, nightlife, people, unique experiences.

### 😬 Common Complaints
4-5 bullets: crowds, costs, weather, safety issues, scams, transport pain points.

### 🗓 Best Time to Visit
2-3 bullets: peak vs off-peak, weather patterns, festival seasons.

### 💡 Insider Tips
5 practical tips: transport hacks, etiquette, money saving, safety reminders, hidden gems.

Keep it honest, specific and genuinely useful. Avoid generic travel blog fluff.""".strip()

            with st.spinner(f"Researching {dest2}..."):
                insights_md = run_async(gemini_call(client, GEMINI_MODEL, insights_prompt))
                st.session_state["insights_md"] = insights_md

        if st.session_state.get("insights_md"):
            st.markdown(st.session_state["insights_md"])

    # ==============================
    # TAB 6 — Essentials (Packing + Visa + Share)
    # ==============================
    with tab6:
        dest2  = st.session_state.get("destination", "")
        out_d2 = st.session_state.get("outbound_date", "")
        ret_d2 = st.session_state.get("return_date", "")
        try:
            _n2 = (date.fromisoformat(ret_d2) - date.fromisoformat(out_d2)).days
        except Exception:
            _n2 = 7

        has_essentials = st.session_state.get("packing_md") or st.session_state.get("visa_md")
        if not has_essentials:
            st.markdown(f"""
            <div class="empty-state">
              <div class="empty-icon">🎒</div>
              <div class="empty-title">Packing list & visa info</div>
              <div class="empty-sub">Generate a smart packing list and entry requirements for {dest2}.</div>
            </div>""", unsafe_allow_html=True)

        if st.button("Generate Packing List & Visa Info", type="primary", key="btn_essentials"):
            pack_p2 = f"""Generate a smart packing list in Markdown for:
- Destination: {dest2}
- Trip duration: {_n2} nights
- Trip style: {style}
- Dates: {out_d2} to {ret_d2}

Sections: 👕 Clothing, 🧴 Toiletries, 💊 Health & Safety, 📱 Tech & Documents, 🎒 Day-trip Essentials.
Keep it practical. Flag destination-specific items (e.g. temple dress code, rain gear, adapters)."""

            visa_p2 = f"""Concise Markdown visa & entry requirements for travellers visiting {dest2}.

### 🛂 Visa Overview
- Nationalities that get visa-free / visa-on-arrival
- Who needs advance application

### 📋 Entry Requirements
- Passport validity
- Common documents
- Health / vaccination requirements

### ⚠️ Important Notes
- Recent policy changes
- Always verify with official embassy sources."""

            async def run_essentials():
                p_md, v_md = await asyncio.gather(
                    gemini_call(client, GEMINI_MODEL, pack_p2),
                    gemini_call(client, GEMINI_MODEL, visa_p2),
                )
                return p_md, v_md

            with st.spinner("Generating packing list & visa info..."):
                p_md, v_md = run_async(run_essentials())
                st.session_state["packing_md"] = p_md
                st.session_state["visa_md"]    = v_md

        if st.session_state.get("packing_md"):
            st.markdown("### 🎒 Packing List")
            st.markdown(st.session_state["packing_md"])

        if st.session_state.get("visa_md"):
            st.markdown("---")
            st.markdown("### 🛂 Visa & Entry Requirements")
            st.markdown(st.session_state["visa_md"])

        # ── WHATSAPP / EMAIL SHARE ──────────────────────────────
        st.markdown("---")
        st.markdown("### 📤 Share this trip")

        _o2  = st.session_state.get("origin", "")
        _d2  = st.session_state.get("destination", "")
        _od2 = st.session_state.get("outbound_date", "")
        _rd2 = st.session_state.get("return_date", "")

        share_text = (
            f"✈ AI Travel Plan: {_o2} → {_d2}\n"
            f"📅 {_od2} to {_rd2} ({_n2} nights)\n"
            f"🌐 Plan yours free at: https://travel-ai-planner-tcghmxw5yc5z5dst8xarek.streamlit.app"
        )

        from urllib.parse import quote as url_quote
        wa_url    = f"https://wa.me/?text={url_quote(share_text)}"
        email_url = f"mailto:?subject={url_quote(f'Trip Plan: {_o2} to {_d2}')}&body={url_quote(share_text)}"

        sh1, sh2 = st.columns(2)
        sh1.markdown(
            f'<a href="{wa_url}" target="_blank" style="'
            f'display:block;text-align:center;padding:0.6rem 1rem;'
            f'background:rgba(37,211,102,0.15);border:1px solid rgba(37,211,102,0.35);'
            f'border-radius:10px;color:#25d366 !important;font-weight:600;font-size:0.9rem;'
            f'text-decoration:none;">📱 Share on WhatsApp</a>',
            unsafe_allow_html=True,
        )
        sh2.markdown(
            f'<a href="{email_url}" style="'
            f'display:block;text-align:center;padding:0.6rem 1rem;'
            f'background:rgba(250,124,79,0.12);border:1px solid rgba(250,124,79,0.3);'
            f'border-radius:10px;color:#fa7c4f !important;font-weight:600;font-size:0.9rem;'
            f'text-decoration:none;">✉️ Share via Email</a>',
            unsafe_allow_html=True,
        )

else:
    # ---- Landing empty state ----
    st.markdown("""
    <style>
    .destinations-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.65rem;
      margin: 1.5rem 0 2rem;
    }
    .dest-card {
      position: relative;
      border-radius: 14px;
      overflow: hidden;
      aspect-ratio: 3/2;
      cursor: default;
      border: 1px solid rgba(255,200,150,0.10);
      transition: transform 0.2s, border-color 0.2s;
    }
    .dest-card:hover {
      transform: translateY(-3px);
      border-color: rgba(250,124,79,0.35);
    }
    .dest-bg {
      width: 100%; height: 100%;
      background-size: cover;
      background-position: center;
      filter: brightness(0.55) saturate(1.1);
      transition: filter 0.3s;
    }
    .dest-card:hover .dest-bg { filter: brightness(0.70) saturate(1.2); }
    .dest-label {
      position: absolute;
      bottom: 0; left: 0; right: 0;
      padding: 0.55rem 0.7rem 0.5rem;
      background: linear-gradient(transparent, rgba(10,5,2,0.85));
    }
    .dest-city {
      font-size: 0.88rem;
      font-weight: 700;
      color: #fdf4ec !important;
      line-height: 1.2;
    }
    .dest-country {
      font-size: 0.68rem;
      color: rgba(253,244,236,0.65) !important;
    }
    .dest-emoji {
      position: absolute;
      top: 0.5rem; right: 0.6rem;
      font-size: 1.1rem;
      filter: drop-shadow(0 1px 3px rgba(0,0,0,0.6));
    }
    .landing-cta {
      text-align: center;
      padding: 0.5rem 0 1.5rem;
    }
    .landing-cta-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #fdf4ec !important;
      margin-bottom: 0.4rem;
    }
    .landing-cta-sub {
      font-size: 0.85rem;
      color: #a8896e !important;
    }
    .landing-cta-sub strong { color: #fa7c4f !important; }
    .landing-divider {
      display: flex;
      align-items: center;
      gap: 0.8rem;
      margin: 1.2rem 0 1rem;
      color: #a8896e !important;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .landing-divider::before, .landing-divider::after {
      content: '';
      flex: 1;
      height: 1px;
      background: rgba(255,200,150,0.10);
    }
    </style>

    <div class="landing-cta">
      <div class="landing-cta-title">✈ Where are you going?</div>
      <div class="landing-cta-sub">
        Enter your route &amp; dates above — hit <strong>Search</strong> or <strong>Plan My Entire Trip</strong> for a full AI itinerary
      </div>
    </div>

    <div class="landing-divider">Popular destinations</div>

    <div class="destinations-grid">
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400&q=70');"></div>
        <div class="dest-emoji">🗼</div>
        <div class="dest-label"><div class="dest-city">Tokyo</div><div class="dest-country">Japan</div></div>
      </div>
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=400&q=70');"></div>
        <div class="dest-emoji">🥐</div>
        <div class="dest-label"><div class="dest-city">Paris</div><div class="dest-country">France</div></div>
      </div>
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1538970272646-f61fabb3a8a2?w=400&q=70');"></div>
        <div class="dest-emoji">🌴</div>
        <div class="dest-label"><div class="dest-city">Bali</div><div class="dest-country">Indonesia</div></div>
      </div>
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1533929736458-ca588d08c8be?w=400&q=70');"></div>
        <div class="dest-emoji">🎡</div>
        <div class="dest-label"><div class="dest-city">London</div><div class="dest-country">United Kingdom</div></div>
      </div>
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=400&q=70');"></div>
        <div class="dest-emoji">🌆</div>
        <div class="dest-label"><div class="dest-city">Dubai</div><div class="dest-country">UAE</div></div>
      </div>
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1534430480872-3498386e7856?w=400&q=70');"></div>
        <div class="dest-emoji">🦁</div>
        <div class="dest-label"><div class="dest-city">Phuket</div><div class="dest-country">Thailand</div></div>
      </div>
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1555993539-1732b0258235?w=400&q=70');"></div>
        <div class="dest-emoji">🎰</div>
        <div class="dest-label"><div class="dest-city">New York</div><div class="dest-country">USA</div></div>
      </div>
      <div class="dest-card">
        <div class="dest-bg" style="background-image:url('https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?w=400&q=70');"></div>
        <div class="dest-emoji">🦘</div>
        <div class="dest-label"><div class="dest-city">Sydney</div><div class="dest-country">Australia</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
