# streamlit_app.py
import os
import json
import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from serpapi import GoogleSearch
from google import genai

# =================================================
# THEME / CSS
# =================================================
def inject_css():
    st.markdown(
        """
        <style>
        :root{
          --bg: #0B0F17;
          --panel: rgba(255,255,255,0.06);
          --panel2: rgba(255,255,255,0.04);
          --border: rgba(255,255,255,0.12);
          --muted: rgba(232,234,240,0.72);
          --text: #E8EAF0;
          --accent: #FF4848;
          --good: #43d17a;
          --warn: #f4b740;
        }
        .stApp {
          background:
            radial-gradient(1000px 560px at 18% 0%, rgba(255,255,255,0.09), transparent 62%),
            radial-gradient(1000px 560px at 82% 6%, rgba(255,255,255,0.07), transparent 62%),
            radial-gradient(900px 500px at 60% 95%, rgba(255,72,72,0.10), transparent 62%),
            var(--bg);
          color: var(--text);
        }
        header[data-testid="stHeader"]{ background: transparent; }
        .block-container{ padding-top: 1.8rem; max-width: 1250px; }

        /* Inputs */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
          background: rgba(255,255,255,0.06) !important;
          border: 1px solid rgba(255,255,255,0.12) !important;
          color: var(--text) !important;
          border-radius: 14px !important;
        }
        .stSlider [data-baseweb="slider"]{ margin-top: 10px; }

        /* Buttons */
        div.stButton>button {
          border-radius: 14px !important;
          padding: 0.68rem 1.0rem !important;
          font-weight: 900 !important;
          border: 1px solid rgba(255,255,255,0.14) !important;
          background: linear-gradient(180deg, rgba(255,72,72,0.95), rgba(255,72,72,0.74)) !important;
          color: white !important;
          transition: transform 0.08s ease, filter 0.08s ease;
        }
        div.stButton>button:hover { transform: translateY(-1px); filter: brightness(1.06); }

        /* Secondary button style via markdown container */
        .secondaryBtn button {
          background: rgba(255,255,255,0.06) !important;
          border: 1px solid rgba(255,255,255,0.14) !important;
          color: var(--text) !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
          background: rgba(255,255,255,0.03);
          border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* Cards */
        .card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 16px 16px;
          box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        }
        .card2 {
          background: var(--panel2);
          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 16px;
          padding: 14px 14px;
        }
        .title { font-size: 2.15rem; font-weight: 950; letter-spacing: -0.02em; line-height: 1.1; }
        .subtitle { color: var(--muted); margin-top: 6px; }
        .card-title { font-size: 1.05rem; font-weight: 950; margin-bottom: 6px; letter-spacing: -0.01em; }
        .muted { color: var(--muted); }
        .pill {
          display: inline-block;
          padding: 6px 10px;
          border-radius: 999px;
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.11);
          margin-right: 8px;
          font-size: 0.86rem;
        }
        .pill.good{ border-color: rgba(67,209,122,0.35); background: rgba(67,209,122,0.10); }
        .pill.warn{ border-color: rgba(244,183,64,0.35); background: rgba(244,183,64,0.10); }
        .pill.accent{ border-color: rgba(255,72,72,0.40); background: rgba(255,72,72,0.10); }
        .divider { height: 1px; background: rgba(255,255,255,0.10); margin: 12px 0; }

        /* Tabs */
        button[data-baseweb="tab"] { font-weight: 900 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def card(html: str):
    st.markdown(f"<div class='card'>{html}</div>", unsafe_allow_html=True)

def card2(html: str):
    st.markdown(f"<div class='card2'>{html}</div>", unsafe_allow_html=True)

# =================================================
# HELPERS
# =================================================
def normalize_date(date_str: str) -> str:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError("Invalid date. Use YYYY-MM-DD (or YYYY/MM/DD).")

def _extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""

@st.cache_data(ttl=3600, show_spinner=False)
def serpapi_search_sync(params: dict) -> dict:
    return GoogleSearch(params).get_dict()

def get_flights_sync(serp_key: str, origin: str, destination: str, out_date: str, ret_date: str) -> List[Dict]:
    params = {
        "engine": "google_flights",
        "api_key": serp_key,
        "departure_id": origin.upper(),
        "arrival_id": destination.upper(),
        "outbound_date": out_date,
        "return_date": ret_date,
        "currency": "USD",
        "hl": "en",
        "gl": "us",
    }
    results = serpapi_search_sync(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    return results.get("best_flights", []) or []

def get_hotels_sync(serp_key: str, location: str, check_in: str, check_out: str) -> List[Dict]:
    params = {
        "engine": "google_hotels",
        "api_key": serp_key,
        "q": location,
        "check_in_date": check_in,
        "check_out_date": check_out,
        "currency": "USD",
        "hl": "en",
        "gl": "us",
        "rating": 8,
    }
    results = serpapi_search_sync(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    return results.get("properties", []) or []

def fmt_flights(best_flights: List[Dict]) -> List[Dict]:
    out = []
    for f in best_flights[:18]:
        legs = f.get("flights") or []
        if not legs:
            continue
        leg = legs[0]
        dep = leg.get("departure_airport", {}) or {}
        arr = leg.get("arrival_airport", {}) or {}
        out.append({
            "airline": leg.get("airline", "Unknown"),
            "price": f.get("price", None),  # may be int
            "price_str": str(f.get("price", "N/A")),
            "duration_min": f.get("total_duration", None),
            "stops_count": max(0, len(legs) - 1),
            "stops": "Nonstop" if len(legs) == 1 else f"{len(legs)-1} stop(s)",
            "depart": f"{dep.get('id','')} {dep.get('time','')}",
            "arrive": f"{arr.get('id','')} {arr.get('time','')}",
            "class": leg.get("travel_class", "Economy"),
        })
    return out

def fmt_hotels(props: List[Dict]) -> List[Dict]:
    out = []
    for h in props[:18]:
        out.append({
            "name": h.get("name", "Unknown"),
            "price": (h.get("rate_per_night", {}) or {}).get("lowest", "N/A"),
            "rating": float(h.get("overall_rating", 0.0) or 0.0),
            "location": h.get("location", ""),
            "link": h.get("link", ""),
        })
    return out

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def sort_flights(flights: List[Dict], mode: str) -> List[Dict]:
    if mode == "Best value":
        # heuristic: low price, low duration, fewer stops
        def score(f):
            p = safe_int(f.get("price"), 999999) if f.get("price") is not None else 999999
            d = safe_int(f.get("duration_min"), 999999) if f.get("duration_min") is not None else 999999
            s = safe_int(f.get("stops_count"), 99)
            return p * 1.0 + d * 0.6 + s * 200
        return sorted(flights, key=score)
    if mode == "Cheapest":
        return sorted(flights, key=lambda f: safe_int(f.get("price"), 999999) if f.get("price") is not None else 999999)
    if mode == "Fastest":
        return sorted(flights, key=lambda f: safe_int(f.get("duration_min"), 999999) if f.get("duration_min") is not None else 999999)
    return flights

def sort_hotels(hotels: List[Dict], mode: str) -> List[Dict]:
    if mode == "Top rated":
        return sorted(hotels, key=lambda h: float(h.get("rating", 0.0)), reverse=True)
    if mode == "Cheapest (rough)":
        # many hotel prices are strings like "$120"; keep simple
        def pval(h):
            s = str(h.get("price", "")).replace("$", "").replace(",", "").strip()
            return float(s) if s.replace(".", "", 1).isdigit() else 1e9
        return sorted(hotels, key=pval)
    return hotels

async def gemini_call(client, model: str, prompt: str, retries: int = 5) -> str:
    def _call() -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return _extract_text(resp)

    for i in range(retries):
        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            msg = str(e).lower()
            if ("503" in msg) or ("overloaded" in msg) or ("unavailable" in msg) or ("429" in msg) or ("quota" in msg):
                wait = (2 ** i) + random.uniform(0, 0.8)
                await asyncio.sleep(wait)
                continue
            return f"⚠️ AI error: {e}"
    return "⚠️ AI is busy / quota-limited. Please try again shortly."

# =================================================
# UI RENDERERS
# =================================================
def badge_for_flight(f: Dict, cheapest_price: Optional[int], fastest_dur: Optional[int]) -> str:
    p = safe_int(f.get("price"), None)
    d = safe_int(f.get("duration_min"), None)
    s = safe_int(f.get("stops_count"), 99)
    tags = []
    if cheapest_price is not None and p is not None and p == cheapest_price:
        tags.append("<span class='pill good'>🏷️ Cheapest</span>")
    if fastest_dur is not None and d is not None and d == fastest_dur:
        tags.append("<span class='pill warn'>⚡ Fastest</span>")
    if s == 0:
        tags.append("<span class='pill accent'>🟢 Nonstop</span>")
    return " ".join(tags)

def render_flights(flights: List[Dict]):
    if not flights:
        st.info("No flights found (try loosening filters).")
        return

    prices = [safe_int(f.get("price"), None) for f in flights if safe_int(f.get("price"), None) is not None]
    durs = [safe_int(f.get("duration_min"), None) for f in flights if safe_int(f.get("duration_min"), None) is not None]
    cheapest = min(prices) if prices else None
    fastest = min(durs) if durs else None

    for idx, f in enumerate(flights[:10]):
        chosen = (st.session_state.get("chosen_flight_idx") == idx)
        tags = badge_for_flight(f, cheapest, fastest)
        title = f"{f.get('airline','Unknown')} — {f.get('class','')}"
        html = f"""
        <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
          <div>
            <div class="card-title">{title}</div>
            <div style="margin-bottom:10px;">
              <span class="pill">💰 {f.get('price_str','N/A')}</span>
              <span class="pill">⏱️ {f.get('duration_min','N/A')} min</span>
              <span class="pill">🧭 {f.get('stops','')}</span>
              {tags}
            </div>
            <div class="muted">🛫 {f.get('depart','')} &nbsp;→&nbsp; 🛬 {f.get('arrive','')}</div>
          </div>
          <div style="min-width:160px; text-align:right;">
            <div class="muted" style="font-size:0.86rem; margin-bottom:6px;">Select for AI</div>
          </div>
        </div>
        """
        card(html)
        c1, c2 = st.columns([1, 6])
        with c1:
            if st.button("✅ Selected" if chosen else "Select", key=f"pick_f_{idx}"):
                st.session_state.chosen_flight_idx = idx
        with c2:
            st.caption("Tip: choose one flight so the AI itinerary & picks can be tailored.")
        st.write("")

def render_hotels(hotels: List[Dict]):
    if not hotels:
        st.info("No hotels found (try lowering min rating).")
        return

    for idx, h in enumerate(hotels[:10]):
        chosen_set = st.session_state.get("chosen_hotel_idxs", set())
        chosen = idx in chosen_set
        link = h.get("link","").strip()
        name = h.get("name","Unknown")
        name_html = f'<a href="{link}" target="_blank" style="color:#E8EAF0; text-decoration:none;">{name}</a>' if link else name
        badge = "<span class='pill good'>⭐ Top rated</span>" if h.get("rating", 0) >= 9 else ""
        html = f"""
        <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start;">
          <div>
            <div class="card-title">{name_html}</div>
            <div style="margin-bottom:10px;">
              <span class="pill">💵 {h.get('price','N/A')}</span>
              <span class="pill">⭐ {h.get('rating',0.0):.1f}</span>
              {badge}
            </div>
            <div class="muted">📍 {h.get('location','')}</div>
          </div>
          <div style="min-width:180px; text-align:right;">
            <div class="muted" style="font-size:0.86rem; margin-bottom:6px;">Select up to 3</div>
          </div>
        </div>
        """
        card(html)

        c1, c2 = st.columns([1, 6])
        with c1:
            label = "✅ Selected" if chosen else "Select"
            if st.button(label, key=f"pick_h_{idx}"):
                chosen_set = set(st.session_state.get("chosen_hotel_idxs", set()))
                if chosen:
                    chosen_set.remove(idx)
                else:
                    if len(chosen_set) >= 3:
                        st.warning("You can select up to 3 hotels.")
                    else:
                        chosen_set.add(idx)
                st.session_state.chosen_hotel_idxs = chosen_set
        with c2:
            st.caption("Tip: select 1–3 hotels so the AI can recommend based on your shortlist.")
        st.write("")

def progress_rail(step: int):
    # 1..4
    labels = ["Trip", "Results", "AI Picks", "Itinerary"]
    pills = []
    for i, l in enumerate(labels, start=1):
        cls = "pill good" if i < step else ("pill accent" if i == step else "pill")
        pills.append(f"<span class='{cls}'> {i}. {l}</span>")
    card2("".join(pills))

# =================================================
# APP
# =================================================
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")
inject_css()

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom: 6px;">
      <div style="font-size:2.2rem;">✈️</div>
      <div>
        <div class="title">AI Travel Planner</div>
        <div class="subtitle">Pick flights + hotels → get an itinerary crafted around your choices.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Keys
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY"))
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

if not GEMINI_API_KEY or not SERPAPI_API_KEY:
    st.error("Missing API keys. Set GEMINI_API_KEY and SERPAPI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# Session defaults
st.session_state.setdefault("loading", False)
st.session_state.setdefault("step", 1)
st.session_state.setdefault("results", None)
st.session_state.setdefault("chosen_flight_idx", None)
st.session_state.setdefault("chosen_hotel_idxs", set())

# Sidebar = Trip Builder
with st.sidebar:
    st.markdown("## 🧰 Trip Builder")
    preset = st.selectbox("Preset", ["Custom", "Weekend getaway", "Budget trip", "Luxury chill"], index=0)

    if preset == "Weekend getaway":
        budget = "Mid-range"
        max_stops = "Up to 1 stop"
        min_hotel_rating = 8.0
    elif preset == "Budget trip":
        budget = "Budget"
        max_stops = "Any"
        min_hotel_rating = 7.5
    elif preset == "Luxury chill":
        budget = "Luxury"
        max_stops = "Nonstop only"
        min_hotel_rating = 9.0
    else:
        budget = st.selectbox("Travel style", ["Any", "Budget", "Mid-range", "Luxury"], index=1)
        max_stops = st.selectbox("Max stops", ["Any", "Nonstop only", "Up to 1 stop"], index=0)
        min_hotel_rating = st.slider("Min hotel rating", 0.0, 10.0, 8.0, 0.5)

    sort_f = st.selectbox("Flight sort", ["Best value", "Cheapest", "Fastest"], index=0)
    sort_h = st.selectbox("Hotel sort", ["Top rated", "Cheapest (rough)", "None"], index=0)
    st.divider()
    st.caption(f"Model: `{GEMINI_MODEL}`")

    if st.button("🧹 Reset selections"):
        st.session_state.chosen_flight_idx = None
        st.session_state.chosen_hotel_idxs = set()
        st.session_state.step = 1

# Inputs row
col1, col2, col3, col4 = st.columns(4)
origin = col1.text_input("Origin (IATA)", value="SIN")
destination = col2.text_input("Destination (IATA)", value="NRT")
outbound_date = col3.text_input("Outbound date (YYYY-MM-DD)", value="2026-02-10")
return_date = col4.text_input("Return date (YYYY-MM-DD)", value="2026-02-15")
location = st.text_input("Hotel location (optional, defaults to destination city/code)", value="")

# Progress rail
progress_rail(st.session_state.step)

# Validation
def validate_inputs() -> Tuple[bool, str]:
    if not origin.strip() or not destination.strip():
        return False, "Origin and destination cannot be empty."
    try:
        _ = normalize_date(outbound_date)
        _ = normalize_date(return_date)
    except Exception as e:
        return False, str(e)
    return True, ""

ok, err = validate_inputs()
if not ok:
    st.warning(err)

# Primary action
search_clicked = st.button("Search", type="primary", disabled=st.session_state.loading or (not ok))

if search_clicked:
    st.session_state.loading = True
    try:
        out_d = normalize_date(outbound_date)
        ret_d = normalize_date(return_date)
        hotel_loc = location.strip() or destination.strip()

        st.session_state.step = 2

        with st.spinner("Searching flights & hotels..."):
            best_flights_raw = get_flights_sync(SERPAPI_API_KEY, origin, destination, out_d, ret_d)
            hotels_raw = get_hotels_sync(SERPAPI_API_KEY, hotel_loc, out_d, ret_d)

        flights = fmt_flights(best_flights_raw)
        hotels = fmt_hotels(hotels_raw)

        # Filters: stops
        if max_stops == "Nonstop only":
            flights = [f for f in flights if f.get("stops_count", 99) == 0]
        elif max_stops == "Up to 1 stop":
            flights = [f for f in flights if f.get("stops_count", 99) <= 1]

        # Rating
        hotels = [h for h in hotels if float(h.get("rating", 0.0) or 0.0) >= float(min_hotel_rating)]

        # Sorting
        flights = sort_flights(flights, sort_f)
        hotels = sort_hotels(hotels, sort_h)

        # Store results in session
        st.session_state.results = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "out_d": out_d,
            "ret_d": ret_d,
            "hotel_loc": hotel_loc,
            "budget": budget,
            "flights": flights,
            "hotels": hotels,
        }

    except Exception as e:
        st.error(str(e))
    finally:
        st.session_state.loading = False

# If we have results, show main content
res = st.session_state.get("results")

if res:
    flights = res["flights"]
    hotels = res["hotels"]

    # Summary hero
    top_price = flights[0]["price_str"] if flights else "N/A"
    top_hotel = hotels[0]["name"] if hotels else "N/A"

    card(
        f"""
        <div class="card-title">Trip Summary</div>
        <div style="margin-bottom:10px;">
          <span class="pill accent">🛫 {res['origin']} → {res['destination']}</span>
          <span class="pill">📅 {res['out_d']} → {res['ret_d']}</span>
          <span class="pill">✨ {res['budget']}</span>
        </div>
        <div class="muted">From results: best flight price <b>{top_price}</b> • top hotel <b>{top_hotel}</b></div>
        """
    )
    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Compare", "AI Picks", "Itinerary & Export"])

    with tab1:
        st.session_state.step = max(st.session_state.step, 2)
        st.subheader("Flights")
        render_flights(flights)

        st.subheader("Hotels")
        render_hotels(hotels)

        card2(
            f"""
            <div class="card-title">Selection status</div>
            <div class="muted">
              Selected flight: <b>{'Yes' if st.session_state.chosen_flight_idx is not None else 'No'}</b><br/>
              Selected hotels: <b>{len(st.session_state.chosen_hotel_idxs)}</b> / 3
            </div>
            """
        )

    with tab2:
        st.subheader("Top 3 comparison (quick decision)")
        f3 = flights[:3]
        h3 = hotels[:3]
        c1, c2, c3 = st.columns(3)

        for i, col in enumerate([c1, c2, c3]):
            with col:
                if i < len(f3):
                    f = f3[i]
                    card2(
                        f"""
                        <div class="card-title">✈️ #{i+1} {f.get('airline','')}</div>
                        <div class="muted">{f.get('depart','')} → {f.get('arrive','')}</div>
                        <div class="divider"></div>
                        <span class="pill">💰 {f.get('price_str','')}</span>
                        <span class="pill">⏱️ {f.get('duration_min','')} min</span>
                        <span class="pill">🧭 {f.get('stops','')}</span>
                        """
                    )
                st.write("")
                if i < len(h3):
                    h = h3[i]
                    card2(
                        f"""
                        <div class="card-title">🏨 #{i+1} {h.get('name','')}</div>
                        <div class="muted">{h.get('location','')}</div>
                        <div class="divider"></div>
                        <span class="pill">💵 {h.get('price','')}</span>
                        <span class="pill">⭐ {h.get('rating',0.0):.1f}</span>
                        """
                    )

    with tab3:
        st.session_state.step = max(st.session_state.step, 3)
        st.subheader("AI Picks (uses your selected flight/hotels if chosen)")

        chosen_f = None
        if st.session_state.chosen_flight_idx is not None and st.session_state.chosen_flight_idx < len(flights):
            chosen_f = flights[st.session_state.chosen_flight_idx]

        chosen_h = []
        for idx in sorted(list(st.session_state.chosen_hotel_idxs)):
            if idx < len(hotels):
                chosen_h.append(hotels[idx])

        flight_context = chosen_f if chosen_f else flights[:6]
        hotel_context = chosen_h if chosen_h else hotels[:6]

        flight_prompt = f"""You are a travel concierge. Recommend the best 1–2 flight options from the provided JSON.
Route: {res['origin']} -> {res['destination']}
Dates: {res['out_d']} to {res['ret_d']}
Travel style: {res['budget']}
If a flight is selected, prioritize that but still suggest a backup.
Flights JSON: {flight_context}
Return Markdown with:
- Final pick + why
- Backup pick + why
- Any risks (tight layovers, long travel time, etc.)"""

        hotel_prompt = f"""You are a travel concierge. Recommend the best 2–3 hotels from the provided JSON.
Location: {res['hotel_loc']}
Dates: {res['out_d']} to {res['ret_d']}
Travel style: {res['budget']}
If hotels are selected, rank those first, but still suggest alternatives.
Hotels JSON: {hotel_context}
Return Markdown with:
- Ranked list (2–3)
- Why it fits the style
- Tips for choosing rooms/areas
Include links if present in JSON."""

        with st.spinner("Asking Gemini..."):
            ai_flights = asyncio.run(gemini_call(client, GEMINI_MODEL, flight_prompt))
            ai_hotels = asyncio.run(gemini_call(client, GEMINI_MODEL, hotel_prompt))

        card(f"<div class='card-title'>✈️ Flight picks</div><div class='divider'></div>{ai_flights}")
        st.write("")
        card(f"<div class='card-title'>🏨 Hotel picks</div><div class='divider'></div>{ai_hotels}")

    with tab4:
        st.session_state.step = max(st.session_state.step, 4)
        st.subheader("Itinerary & Export")

        chosen_f = None
        if st.session_state.chosen_flight_idx is not None and st.session_state.chosen_flight_idx < len(flights):
            chosen_f = flights[st.session_state.chosen_flight_idx]

        chosen_h = []
        for idx in sorted(list(st.session_state.chosen_hotel_idxs)):
            if idx < len(hotels):
                chosen_h.append(hotels[idx])

        context_md = ""
        if chosen_f:
            context_md += f"\nSelected flight: {json.dumps(chosen_f, ensure_ascii=False)}\n"
        if chosen_h:
            context_md += f"\nSelected hotels: {json.dumps(chosen_h, ensure_ascii=False)}\n"

        itinerary_prompt = f"""Create a practical day-by-day itinerary in Markdown for {res['destination']}
from {res['out_d']} to {res['ret_d']}.
Travel style: {res['budget']}.
Must include:
- Morning / Afternoon / Evening per day
- 2–3 food ideas per day
- Transit hints (walk/train/taxi)
- One "must-do" highlight per day
Use the selected flight/hotel context if provided.
Context: {context_md}
"""

        tips_prompt = f"""Give concise travel tips for {res['destination']} in Markdown:
- Best areas to stay (match travel style: {res['budget']})
- Getting around (IC cards, passes, typical costs)
- Budget tips
- 5 tourist mistakes to avoid
- A simple packing list for the season/date window
"""

        with st.spinner("Generating itinerary & tips..."):
            itinerary = asyncio.run(gemini_call(client, GEMINI_MODEL, itinerary_prompt))
            tips = asyncio.run(gemini_call(client, GEMINI_MODEL, tips_prompt))

        card(f"<div class='card-title'>🗺️ Itinerary</div><div class='divider'></div>{itinerary}")
        st.write("")
        card(f"<div class='card-title'>✅ Tips</div><div class='divider'></div>{tips}")

        export = {
            "trip": {
                "origin": res["origin"],
                "destination": res["destination"],
                "outbound_date": res["out_d"],
                "return_date": res["ret_d"],
                "hotel_location": res["hotel_loc"],
                "budget": res["budget"],
            },
            "selected": {
                "flight": chosen_f,
                "hotels": chosen_h,
            },
            "results": {
                "flights": flights[:10],
                "hotels": hotels[:10],
            }
        }

        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                "Download itinerary (Markdown)",
                data=itinerary,
                file_name=f"itinerary_{res['destination']}_{res['out_d']}_to_{res['ret_d']}.md",
                mime="text/markdown",
            )
        with colB:
            st.download_button(
                "Download trip pack (JSON)",
                data=json.dumps(export, indent=2, ensure_ascii=False),
                file_name=f"trip_pack_{res['destination']}_{res['out_d']}_to_{res['ret_d']}.json",
                mime="application/json",
            )
