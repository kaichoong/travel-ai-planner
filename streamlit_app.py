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
async def get_flights(serp_key, origin, destination, out_date, ret_date, currency):
    params = {
        "engine": "google_flights",
        "api_key": serp_key,
        "departure_id": origin.upper(),
        "arrival_id": destination.upper(),
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

def _pick_hotel_area(raw):
    candidates = [
        raw.get("neighborhood"), raw.get("area"),
        raw.get("address"), raw.get("location"),
        raw.get("formatted_address"), raw.get("city"),
    ]
    for v in candidates:
        v = clean_text(v)
        if v:
            return v
    addr = raw.get("address_info") or raw.get("address_details") or {}
    if isinstance(addr, dict):
        for k in ["neighborhood", "area", "address", "formatted_address", "city", "region"]:
            v = clean_text(addr.get(k))
            if v:
                return v
    return ""

def _pick_hotel_amenities(raw):
    am = raw.get("amenities") or raw.get("top_features") or raw.get("features") or []
    if isinstance(am, list):
        am = [clean_text(x) for x in am if clean_text(x)][:4]
        return am
    v = clean_text(am)
    return [v] if v else []

def fmt_hotels(props, location_query, check_in, check_out):
    out = []
    fallback_link = google_hotels_search_url(location_query, check_in, check_out)
    for h in props[:18]:
        rate = (h.get("rate_per_night", {}) or {}).get("lowest", h.get("price", "N/A"))
        link = h.get("link") or fallback_link
        area = _pick_hotel_area(h)
        if not area:
            area = f"Near {location_query}"
        out.append({
            "name": clean_text(h.get("name", "Unknown")),
            "price": clean_text(rate),
            "rating": float(h.get("overall_rating", 0.0) or 0.0),
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
.hero { padding: 0.5rem 0 1.8rem; }
.hero h1 { font-size: 2rem !important; font-weight: 600 !important; letter-spacing: -0.03em; margin-bottom: 0.2rem; }
.hero-sub { color: var(--muted) !important; font-size: 0.9rem; }

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
  <h1><span style="color:#fa7c4f;">✈</span> AI Travel Planner</h1>
  <p class="hero-sub">Search flights & hotels · select your picks · let Gemini build your trip</p>
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
    st.caption("1. Enter your route & dates below\n2. Hit **Search**\n3. Select up to 3 flights & 3 hotels\n4. Generate AI picks & itinerary")

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

# Row 1: origin + destination (2 cols) | outbound + return dates (2 cols)
r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1.2, 1.2])
origin      = r1c1.text_input("Origin (IATA)", value=st.session_state.get("origin", "SIN"), placeholder="e.g. SIN").strip().upper()
destination = r1c2.text_input("Destination (IATA)", value=st.session_state.get("destination", "NRT"), placeholder="e.g. NRT").strip().upper()
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

col_s1, col_s2 = st.columns([1, 1])
search_clicked = col_s1.button("🔍  Search Flights & Hotels", type="primary", use_container_width=True)
plan_clicked   = col_s2.button("✨  Plan My Entire Trip", type="secondary", use_container_width=True)

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
            get_flights(SERPAPI_API_KEY, origin, destination, out_d, ret_d, currency),
            get_hotels(SERPAPI_API_KEY, hotel_loc, out_d, ret_d, currency, min_hotel_rating),
        )
        return flights_raw, hotels_raw

    flights_raw, hotels_raw = run_async(fetch_all())
    flights    = sort_flights(fmt_flights(flights_raw, origin, destination, out_d, ret_d), flight_sort)
    hotels_fmt = sort_hotels(fmt_hotels(hotels_raw, hotel_loc, out_d, ret_d), hotel_sort)

    if max_flight_budget > 0:
        flights = [f for f in flights if _parse_price(f.get("price")) <= max_flight_budget]
    if max_hotel_budget > 0:
        hotels_fmt = [h for h in hotels_fmt if _parse_price(h.get("price")) <= max_hotel_budget]

    st.session_state.update({
        "origin": origin, "destination": destination,
        "outbound_date": out_d, "return_date": ret_d,
        "location": location, "hotel_loc": hotel_loc,
        "flights": flights, "hotels": hotels_fmt,
        "max_flight_budget": max_flight_budget,
        "max_hotel_budget":  max_hotel_budget,
        "selected_flights": [], "selected_hotels": [],
        "ai_flights_md": "", "ai_hotels_md": "",
        "itinerary_md": "", "tips_md": "", "insights_md": "",
        "packing_md": "", "visa_md": "",
    })
    return flights, hotels_fmt, hotel_loc


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

            with st.spinner("✨ Gemini is planning your entire trip — hang tight..."):
                f_md, h_md, i_md, t_md, ins_md, pack_md, visa_md = run_async(run_plan_all())

            st.session_state.update({
                "ai_flights_md": f_md, "ai_hotels_md": h_md,
                "itinerary_md": i_md,  "tips_md": t_md,
                "insights_md": ins_md,
                "packing_md": pack_md, "visa_md": visa_md,
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
      <span style="font-size:1.05rem;font-weight:600;color:#fa7c4f !important;letter-spacing:-0.01em;">{_o} → {_d}</span>
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
              <div class="empty-sub">Try different dates, route, or check the IATA codes.</div>
            </div>""", unsafe_allow_html=True)
        else:
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

                with st.container(border=True):
                    # Row 1: logo + airline + class pill | price
                    col_info, col_price = st.columns([4, 1])
                    with col_info:
                        if logo_url:
                            st.markdown(
                                f'<img src="{logo_url}" style="height:22px;vertical-align:middle;margin-right:8px;filter:brightness(1.1);">'
                                f'<strong>{f["airline"]}</strong>'
                                f'<span class="pill pill-blue" style="margin-left:8px;">{f["class"]}</span>'
                                + (f'<span class="pill pill-green">Nonstop</span>' if nonstop else f'<span class="pill pill-warn">{f["stops"]}</span>'),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<strong>{f["airline"]}</strong>'
                                f'<span class="pill pill-blue" style="margin-left:8px;">{f["class"]}</span>'
                                + (f'<span class="pill pill-green">Nonstop</span>' if nonstop else f'<span class="pill pill-warn">{f["stops"]}</span>'),
                                unsafe_allow_html=True,
                            )
                    with col_price:
                        st.markdown(f'<div class="price-tag" style="text-align:right;">{f["price"]}</div>', unsafe_allow_html=True)

                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                    # Row 2: route visual
                    dep_code = f.get("depart_code", "—")
                    dep_time = f.get("depart_time", "")
                    arr_code = f.get("arrive_code", "—")
                    arr_time = f.get("arrive_time", "")

                    st.markdown(f"""
                    <div class="route-line">
                      <div>
                        <div class="iata">{dep_code}</div>
                        <div class="time-small">{dep_time}</div>
                      </div>
                      <div class="dot-line"></div>
                      <div style="text-align:center;">
                        <div class="time-small">⏱ {dur_str}</div>
                      </div>
                      <div class="dot-line"></div>
                      <div style="text-align:right;">
                        <div class="iata">{arr_code}</div>
                        <div class="time-small">{arr_time}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    link = f.get("link") or google_flights_search_url(
                        st.session_state.get("origin", origin),
                        st.session_state.get("destination", destination),
                        st.session_state.get("outbound_date", ""),
                        st.session_state.get("return_date", ""),
                    )
                    st.markdown(f'<a class="link-btn" href="{link}" target="_blank">Open in Google Flights →</a>', unsafe_allow_html=True)

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
            labels = [f"{i+1}. {h['name']}  ·  {h['price']}  ·  {render_star_rating(h['rating'])}" for i, h in enumerate(hotels_fmt)]
            selected = st.multiselect("Select up to 3 hotels to personalise AI picks", options=labels, default=st.session_state.get("selected_hotels", []))
            selected, clipped = clamp_multiselect(selected, 3)
            if clipped:
                st.warning("Maximum 3 hotels — keeping the first 3 selected.")
            st.session_state["selected_hotels"] = selected

            st.markdown("")
            for i, h in enumerate(hotels_fmt[:12]):
                with st.container(border=True):
                    top_l, top_r = st.columns([3, 1])
                    with top_l:
                        prop_type = f' <span class="pill">{h["type"]}</span>' if h.get("type") else ""
                        st.markdown(f'<span class="hotel-name">{i+1}. {h["name"]}</span>{prop_type}', unsafe_allow_html=True)
                    with top_r:
                        st.markdown(f'<div class="price-tag" style="text-align:right;">{h["price"]}</div>', unsafe_allow_html=True)

                    # Rating + reviews
                    full_stars = int(h["rating"] // 2)
                    half_star  = 1 if (h["rating"] % 2) >= 1 else 0
                    empty_stars = 5 - full_stars - half_star
                    stars_html = "★" * full_stars + ("½" if half_star else "") + "☆" * empty_stars
                    reviews_txt = f" · {h['reviews_count']} reviews" if h.get("reviews_count") else ""
                    st.markdown(
                        f'<span class="stars">{stars_html}</span>'
                        f'<span style="font-size:0.82rem;color:#6b7a94;margin-left:4px;">{h["rating"]/2:.1f}/5{reviews_txt}</span>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

                    # Area + distance
                    c_area, c_dist = st.columns([3, 1])
                    with c_area:
                        st.markdown(f'<div class="section-title">Location</div><div style="font-size:0.88rem;">{h["area"]}</div>', unsafe_allow_html=True)
                    with c_dist:
                        if h.get("distance"):
                            st.markdown(f'<div class="section-title">Distance</div><div style="font-size:0.88rem;">{h["distance"]}</div>', unsafe_allow_html=True)

                    # Amenities chips
                    if h.get("amenities"):
                        chips = "".join(f'<span class="amenity-chip">{a}</span>' for a in h["amenities"])
                        st.markdown(f'<div class="section-title" style="margin-top:0.6rem;">Highlights</div>{chips}', unsafe_allow_html=True)

                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    link = h.get("link") or google_hotels_search_url(
                        st.session_state.get("location", "") or destination,
                        st.session_state.get("outbound_date", ""),
                        st.session_state.get("return_date", ""),
                    )
                    st.markdown(f'<a class="link-btn" href="{link}" target="_blank">View hotel listing →</a>', unsafe_allow_html=True)

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
            st.markdown("### ✈ Recommended Flights")
            st.markdown(st.session_state["ai_flights_md"])

        if st.session_state.get("ai_hotels_md"):
            st.markdown("### 🏨 Recommended Hotels")
            st.markdown(st.session_state["ai_hotels_md"])

    # ==============================
    # TAB 4 — Itinerary & Export
    # ==============================
    with tab4:
        dest2  = st.session_state.get("destination", destination)
        out_d2 = st.session_state.get("outbound_date", outbound_date)
        ret_d2 = st.session_state.get("return_date", return_date)

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
            st.markdown("### 🗓 Itinerary")
            st.markdown(st.session_state["itinerary_md"])

        if st.session_state.get("tips_md"):
            st.markdown("### 💡 Tips")
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
    <div class="empty-state" style="padding: 5rem 1rem;">
      <div class="empty-icon">🌍</div>
      <div class="empty-title" style="font-size:1.1rem;">Where are you going?</div>
      <div class="empty-sub">Enter your route & dates above — then hit <strong>Search</strong> or <strong>Plan My Entire Trip</strong> for a full AI itinerary in one click.</div>
    </div>
    """, unsafe_allow_html=True)
