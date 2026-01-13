# streamlit_app.py
import os
import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from serpapi import GoogleSearch
from google import genai

# =================================================
# WORLD-CLASS UI (CSS)
# =================================================
def inject_css():
    st.markdown(
        """
        <style>
        /* Background */
        .stApp {
          background:
            radial-gradient(900px 500px at 20% 0%, rgba(255,255,255,0.08), transparent 60%),
            radial-gradient(900px 500px at 80% 10%, rgba(255,255,255,0.06), transparent 60%),
            #0B0F17;
          color: #E8EAF0;
        }
        .block-container { padding-top: 2.0rem; max-width: 1200px; }

        /* Remove extra header spacing */
        header[data-testid="stHeader"] { background: transparent; }

        /* Inputs */
        .stTextInput input, .stTextArea textarea {
          background: rgba(255,255,255,0.06) !important;
          border: 1px solid rgba(255,255,255,0.12) !important;
          color: #E8EAF0 !important;
          border-radius: 14px !important;
        }

        /* Buttons */
        div.stButton>button {
          border-radius: 14px !important;
          padding: 0.65rem 1.0rem !important;
          font-weight: 800 !important;
          border: 1px solid rgba(255,255,255,0.14) !important;
          background: linear-gradient(180deg, rgba(255,72,72,0.95), rgba(255,72,72,0.75)) !important;
          color: white !important;
          transition: transform 0.08s ease, filter 0.08s ease;
        }
        div.stButton>button:hover { transform: translateY(-1px); filter: brightness(1.06); }

        /* Sidebar */
        section[data-testid="stSidebar"] {
          background: rgba(255,255,255,0.03);
          border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* Card */
        .card {
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 18px;
          padding: 16px 16px;
          box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        }
        .card-title { font-size: 1.05rem; font-weight: 900; margin-bottom: 6px; letter-spacing: -0.01em; }
        .muted { color: rgba(232,234,240,0.72); }
        .pill {
          display: inline-block;
          padding: 6px 10px;
          border-radius: 999px;
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.11);
          margin-right: 8px;
          font-size: 0.86rem;
        }
        .divider { height: 1px; background: rgba(255,255,255,0.10); margin: 12px 0; }

        /* Tabs */
        button[data-baseweb="tab"] { font-weight: 800 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def card(html: str):
    st.markdown(f"<div class='card'>{html}</div>", unsafe_allow_html=True)

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

@st.cache_data(ttl=3600)
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
    for f in best_flights[:12]:
        legs = f.get("flights") or []
        if not legs:
            continue
        leg = legs[0]
        dep = leg.get("departure_airport", {}) or {}
        arr = leg.get("arrival_airport", {}) or {}
        out.append({
            "airline": leg.get("airline", "Unknown"),
            "price": f.get("price", "N/A"),
            "duration_min": f.get("total_duration", "N/A"),
            "stops": "Nonstop" if len(legs) == 1 else f"{len(legs)-1} stop(s)",
            "depart": f"{dep.get('id','')} {dep.get('time','')}",
            "arrive": f"{arr.get('id','')} {arr.get('time','')}",
            "class": leg.get("travel_class", "Economy"),
        })
    return out

def fmt_hotels(props: List[Dict]) -> List[Dict]:
    out = []
    for h in props[:12]:
        out.append({
            "name": h.get("name", "Unknown"),
            "price": (h.get("rate_per_night", {}) or {}).get("lowest", "N/A"),
            "rating": float(h.get("overall_rating", 0.0) or 0.0),
            "location": h.get("location", ""),
            "link": h.get("link", ""),
        })
    return out

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

def render_flight_cards(flights: List[Dict]):
    if not flights:
        st.info("No flights found (try loosening filters).")
        return
    for f in flights:
        html = f"""
        <div class="card-title">{f.get('airline','Unknown')} <span class="muted">— {f.get('class','')}</span></div>
        <div style="margin-bottom:10px;">
          <span class="pill">💰 {f.get('price','N/A')}</span>
          <span class="pill">⏱️ {f.get('duration_min','N/A')} min</span>
          <span class="pill">🧭 {f.get('stops','')}</span>
        </div>
        <div class="muted">🛫 {f.get('depart','')} &nbsp;→&nbsp; 🛬 {f.get('arrive','')}</div>
        """
        card(html)
        st.write("")

def render_hotel_cards(hotels: List[Dict]):
    if not hotels:
        st.info("No hotels found (try lowering min rating).")
        return
    for h in hotels:
        link = h.get("link","").strip()
        name = h.get("name","Unknown")
        name_html = f'<a href="{link}" target="_blank" style="color:#E8EAF0; text-decoration:none;">{name}</a>' if link else name
        html = f"""
        <div class="card-title">{name_html}</div>
        <div style="margin-bottom:10px;">
          <span class="pill">💵 {h.get('price','N/A')}</span>
          <span class="pill">⭐ {h.get('rating',0.0):.1f}</span>
        </div>
        <div class="muted">📍 {h.get('location','')}</div>
        """
        card(html)
        st.write("")

# =================================================
# APP
# =================================================
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")
inject_css()

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom: 8px;">
      <div style="font-size:2.2rem;">✈️</div>
      <div>
        <div style="font-size:2.15rem; font-weight:950; letter-spacing:-0.02em; line-height:1.1;">AI Travel Planner</div>
        <div class="muted" style="margin-top:6px;">Flights + hotels + AI itinerary — in one clean flow.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Secrets (Cloud) or env (local)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY"))
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

if not GEMINI_API_KEY or not SERPAPI_API_KEY:
    st.error("Missing API keys. Set GEMINI_API_KEY and SERPAPI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# Sidebar controls (premium UX)
with st.sidebar:
    st.markdown("### ⚙️ Preferences")
    max_stops = st.selectbox("Max stops", ["Any", "Nonstop only", "Up to 1 stop"], index=0)
    budget = st.selectbox("Travel style", ["Any", "Budget", "Mid-range", "Luxury"], index=1)
    min_hotel_rating = st.slider("Min hotel rating", 0.0, 10.0, 8.0, 0.5)

    st.markdown("### 🧠 Model")
    st.caption(f"Using: `{GEMINI_MODEL}`")

# Inputs row
col1, col2, col3, col4 = st.columns(4)
origin = col1.text_input("Origin (IATA)", value="SIN")
destination = col2.text_input("Destination (IATA)", value="NRT")
outbound_date = col3.text_input("Outbound date (YYYY-MM-DD)", value="2026-02-10")
return_date = col4.text_input("Return date (YYYY-MM-DD)", value="2026-02-15")

location = st.text_input("Hotel location (optional, defaults to destination city/code)", value="")

# Loading lock (prevents double clicks / rerun confusion)
if "loading" not in st.session_state:
    st.session_state.loading = False

search_clicked = st.button("Search", type="primary", disabled=st.session_state.loading)

if search_clicked:
    st.session_state.loading = True
    try:
        out_d = normalize_date(outbound_date)
        ret_d = normalize_date(return_date)
        hotel_loc = location.strip() or destination.strip()

        with st.spinner("Searching flights & hotels..."):
            best_flights = get_flights_sync(SERPAPI_API_KEY, origin, destination, out_d, ret_d)
            hotels_raw = get_hotels_sync(SERPAPI_API_KEY, hotel_loc, out_d, ret_d)

        flights = fmt_flights(best_flights)
        hotels_fmt = fmt_hotels(hotels_raw)

        # Filters
        if max_stops == "Nonstop only":
            flights = [f for f in flights if "Nonstop" in f.get("stops", "")]
        elif max_stops == "Up to 1 stop":
            flights = [f for f in flights if ("Nonstop" in f.get("stops","")) or ("1 stop" in f.get("stops",""))]

        hotels_fmt = [h for h in hotels_fmt if float(h.get("rating", 0.0) or 0.0) >= float(min_hotel_rating)]

        # Summary card
        top_price = flights[0]["price"] if flights else "N/A"
        top_hotel = hotels_fmt[0]["name"] if hotels_fmt else "N/A"

        card(f"""
        <div class="card-title">Trip Summary</div>
        <div style="margin-bottom:10px;">
          <span class="pill">🛫 {origin.upper()} → {destination.upper()}</span>
          <span class="pill">📅 {out_d} → {ret_d}</span>
          <span class="pill">✨ {budget}</span>
        </div>
        <div class="muted">Best flight price (from results): <b>{top_price}</b> • Top hotel (from results): <b>{top_hotel}</b></div>
        """)
        st.write("")

        tab1, tab2, tab3, tab4 = st.tabs(["Flights", "Hotels", "AI Picks", "Itinerary & Tips"])

        with tab1:
            st.subheader("Flights")
            render_flight_cards(flights[:10])

        with tab2:
            st.subheader("Hotels")
            render_hotel_cards(hotels_fmt[:10])

        with tab3:
            st.subheader("AI Picks")
            flight_prompt = f"""Recommend the best 1-2 flights from these options (JSON). Explain briefly in Markdown.
Route: {origin.upper()} -> {destination.upper()}
Dates: {out_d} to {ret_d}
Travel style: {budget}
Flights JSON: {flights[:6]}"""

            hotel_prompt = f"""Recommend the best 2-3 hotels from these options (JSON). Explain briefly in Markdown. Include links if present.
Location: {hotel_loc}
Dates: {out_d} to {ret_d}
Travel style: {budget}
Hotels JSON: {hotels_fmt[:6]}"""

            with st.spinner("Asking Gemini..."):
                ai_flights = asyncio.run(gemini_call(client, GEMINI_MODEL, flight_prompt))
                ai_hotels = asyncio.run(gemini_call(client, GEMINI_MODEL, hotel_prompt))

            card(f"<div class='card-title'>✈️ Flight picks</div><div class='divider'></div>{ai_flights}")
            st.write("")
            card(f"<div class='card-title'>🏨 Hotel picks</div><div class='divider'></div>{ai_hotels}")

        with tab4:
            st.subheader("Itinerary & Tips")
            itinerary_prompt = f"""Create a practical day-by-day itinerary in Markdown for {destination.upper()}
from {out_d} to {ret_d}. Include morning/afternoon/evening and 2-3 food ideas per day.
Travel style: {budget}"""

            tips_prompt = f"""Give concise travel tips for {destination.upper()} in Markdown:
- Best areas to stay
- Getting around
- Budget tips
- 5 tourist mistakes to avoid"""

            with st.spinner("Generating itinerary & tips..."):
                itinerary = asyncio.run(gemini_call(client, GEMINI_MODEL, itinerary_prompt))
                tips = asyncio.run(gemini_call(client, GEMINI_MODEL, tips_prompt))

            card(f"<div class='card-title'>🗺️ Itinerary</div><div class='divider'></div>{itinerary}")
            st.write("")
            card(f"<div class='card-title'>✅ Tips</div><div class='divider'></div>{tips}")

            st.download_button(
                "Download itinerary (Markdown)",
                data=itinerary,
                file_name=f"itinerary_{destination.upper()}_{out_d}_to_{ret_d}.md",
                mime="text/markdown",
            )

    except Exception as e:
        st.error(str(e))
    finally:
        st.session_state.loading = False
