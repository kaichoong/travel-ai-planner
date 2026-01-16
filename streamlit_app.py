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
    """Defensive cleanup so UI never shows HTML fragments."""
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
    raise ValueError("Invalid date. Use YYYY-MM-DD (or YYYY/MM/DD).")

def _extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""

def run_async(coro):
    """
    Streamlit runs in ScriptRunner thread where 'get current loop' can fail.
    Create a fresh event loop per call.
    """
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
            if ("503" in msg) or ("overloaded" in msg) or ("unavailable" in msg) or ("429" in msg) or ("quota" in msg):
                wait = (2 ** i) + random.uniform(0, 0.8)
                await asyncio.sleep(wait)
                continue
            return f"⚠️ AI error: {e}"
    return "⚠️ AI is busy / quota-limited. Please try again shortly."

def google_flights_search_url(origin: str, destination: str, out_date: str, ret_date: str) -> str:
    # Works reliably as a Google Flights search query link
    q = f"Flights from {origin} to {destination} on {out_date} returning {ret_date}"
    return "https://www.google.com/travel/flights?q=" + quote_plus(q)

# -------------------------
# SerpAPI fetchers
# -------------------------
async def get_flights(serp_key: str, origin: str, destination: str, out_date: str, ret_date: str, currency: str) -> List[Dict]:
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

async def get_hotels(serp_key: str, location: str, check_in: str, check_out: str, currency: str, min_rating: float) -> List[Dict]:
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
        props = [p for p in props if float(p.get("overall_rating", 0) or 0) >= float(min_rating)]
    return props

# -------------------------
# Formatting
# -------------------------
def fmt_flights(best_flights: List[Dict], origin: str, destination: str, out_date: str, ret_date: str) -> List[Dict]:
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

        # Best-effort: sometimes SerpAPI includes a link-ish field; otherwise fallback
        link = f.get("link") or f.get("booking_link") or f.get("booking_url") or fallback_link

        out.append({
            "airline": clean_text(leg0.get("airline", "Unknown")),
            "price": clean_text(f.get("price", "N/A")),
            "duration_min": f.get("total_duration", None),
            "stops": clean_text(stops),
            "depart": clean_text(f"{dep.get('id','')} {dep.get('time','')}".strip()),
            "arrive": clean_text(f"{arr.get('id','')} {arr.get('time','')}".strip()),
            "class": clean_text(leg0.get("travel_class", "Economy")),
            "link": clean_text(link),
            "raw": f,
        })
    return out

def fmt_hotels(props: List[Dict]) -> List[Dict]:
    out = []
    for h in props[:18]:
        rate = (h.get("rate_per_night", {}) or {}).get("lowest", "N/A")
        out.append({
            "name": clean_text(h.get("name", "Unknown")),
            "price": clean_text(rate),
            "rating": float(h.get("overall_rating", 0.0) or 0.0),
            "location": clean_text(h.get("location", "")),
            "reviews": clean_text(h.get("reviews", "")),
            "link": clean_text(h.get("link", "")),
            "raw": h,
        })
    return out

def _parse_price_to_number(price_val) -> float:
    if price_val is None:
        return float("inf")
    if isinstance(price_val, (int, float)):
        return float(price_val)
    s = clean_text(price_val)
    digits = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    try:
        return float(digits) if digits else float("inf")
    except Exception:
        return float("inf")

def sort_flights(flights: List[Dict], sort_key: str) -> List[Dict]:
    if sort_key == "Cheapest":
        return sorted(flights, key=lambda x: _parse_price_to_number(x.get("price")))
    if sort_key == "Fastest":
        return sorted(flights, key=lambda x: (x.get("duration_min") is None, x.get("duration_min") or 10**9))
    if sort_key == "Fewest stops":
        def stops_num(s: str) -> int:
            s = (s or "").lower()
            if "nonstop" in s:
                return 0
            digits = "".join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else 99
        return sorted(flights, key=lambda x: stops_num(x.get("stops", "")))
    return flights

def sort_hotels(hotels: List[Dict], sort_key: str) -> List[Dict]:
    if sort_key == "Top rated":
        return sorted(hotels, key=lambda x: x.get("rating", 0), reverse=True)
    if sort_key == "Cheapest":
        return sorted(hotels, key=lambda x: _parse_price_to_number(x.get("price")))
    return hotels

def clamp_multiselect(selection: List[str], limit: int) -> Tuple[List[str], bool]:
    if len(selection) <= limit:
        return selection, False
    return selection[:limit], True

# -------------------------
# UI (Premium dark, native components)
# -------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

st.markdown(
    """
<style>
:root{
  --bg:#0b0f17;
  --panel:#0f1626;
  --text:#e9eef7;
  --muted:#9aa7bd;
  --line:rgba(255,255,255,0.08);
  --radius:18px;
}

html, body, [class*="css"] { background: var(--bg) !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1180px; }
h1, h2, h3, h4, h5, h6, p, li, span, label { color: var(--text) !important; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(16,26,45,1), rgba(11,15,23,1)) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div,
div[data-testid="stMultiSelect"] div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}

.stButton>button {
  background: linear-gradient(90deg, #7cf5c5, #6aa6ff);
  color: #071018 !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.7rem 1rem !important;
  font-weight: 800 !important;
}

[data-testid="stVerticalBlockBorderWrapper"]{
  border-color: var(--line) !important;
  border-radius: var(--radius) !important;
  background: rgba(255,255,255,0.02) !important;
}

.small-note { color: var(--muted) !important; font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("✈️ AI Travel Planner")
st.caption("Search flights & hotels, select what you like, then let Gemini craft your itinerary.")

# -------------------------
# Secrets / Clients
# -------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY"))
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

if not GEMINI_API_KEY or not SERPAPI_API_KEY:
    st.error("Missing API keys. Set GEMINI_API_KEY and SERPAPI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# Sidebar: Trip Builder
# -------------------------
with st.sidebar:
    st.markdown("### Trip Builder")
    style = st.selectbox("Trip style", ["Balanced", "Foodie", "Culture", "Nature", "Shopping", "Luxury", "Budget"], index=0)
    max_stops = st.slider("Max stops (preference)", 0, 2, 1)
    min_hotel_rating = st.slider("Min hotel rating", 0.0, 9.5, 8.0, 0.5)
    flight_sort = st.selectbox("Sort flights", ["Cheapest", "Fastest", "Fewest stops"], index=0)
    hotel_sort = st.selectbox("Sort hotels", ["Top rated", "Cheapest"], index=0)
    currency = st.selectbox("Currency", ["USD", "SGD", "MYR", "JPY", "EUR", "GBP"], index=0)
    st.markdown("---")
    st.markdown('<div class="small-note">Tip: Hit <b>Search</b>, then select up to 3 flights & 3 hotels for the AI.</div>', unsafe_allow_html=True)

# -------------------------
# Inputs
# -------------------------
c1, c2, c3, c4 = st.columns(4)
origin = c1.text_input("Origin (IATA)", value=st.session_state.get("origin", "SIN")).strip().upper()
destination = c2.text_input("Destination (IATA)", value=st.session_state.get("destination", "NRT")).strip().upper()
outbound_date = c3.text_input("Outbound date (YYYY-MM-DD)", value=st.session_state.get("outbound_date", "2026-02-10")).strip()
return_date = c4.text_input("Return date (YYYY-MM-DD)", value=st.session_state.get("return_date", "2026-02-15")).strip()

location = st.text_input("Hotel location (optional — defaults to destination)", value=st.session_state.get("location", "")).strip()

search_clicked = st.button("Search", type="primary")

# -------------------------
# Search
# -------------------------
if search_clicked:
    try:
        out_d = normalize_date(outbound_date)
        ret_d = normalize_date(return_date)

        if origin and destination and origin == destination:
            st.error("Origin and destination cannot be the same.")
            st.stop()

        hotel_loc = location or destination

        async def fetch_all():
            flights_raw, hotels_raw = await asyncio.gather(
                get_flights(SERPAPI_API_KEY, origin, destination, out_d, ret_d, currency),
                get_hotels(SERPAPI_API_KEY, hotel_loc, out_d, ret_d, currency, min_hotel_rating),
            )
            return flights_raw, hotels_raw

        with st.spinner("Searching flights & hotels..."):
            flights_raw, hotels_raw = run_async(fetch_all())

        flights = sort_flights(fmt_flights(flights_raw, origin, destination, out_d, ret_d), flight_sort)
        hotels_fmt = sort_hotels(fmt_hotels(hotels_raw), hotel_sort)

        st.session_state["origin"] = origin
        st.session_state["destination"] = destination
        st.session_state["outbound_date"] = out_d
        st.session_state["return_date"] = ret_d
        st.session_state["location"] = location

        st.session_state["flights"] = flights
        st.session_state["hotels"] = hotels_fmt

        st.session_state["selected_flights"] = []
        st.session_state["selected_hotels"] = []
        st.session_state["ai_flights_md"] = ""
        st.session_state["ai_hotels_md"] = ""
        st.session_state["itinerary_md"] = ""
        st.session_state["tips_md"] = ""

    except Exception as e:
        st.error(str(e))

# -------------------------
# Results / Tabs
# -------------------------
flights = st.session_state.get("flights", [])
hotels_fmt = st.session_state.get("hotels", [])

if flights or hotels_fmt:
    tab1, tab2, tab3, tab4 = st.tabs(["✈️ Flights", "🏨 Hotels", "✨ AI Picks", "🗺️ Itinerary & Export"])

    # ---- Flights ----
    with tab1:
        st.subheader("Flight options")

        if not flights:
            st.info("No flight results found. Try different dates or route.")
        else:
            labels = []
            for i, f in enumerate(flights):
                labels.append(f"{i+1}. {f['airline']} • {f['price']} • {f['stops']} • {f['depart']} → {f['arrive']}")

            selected = st.multiselect("Select up to 3 flights (personalizes AI picks)", options=labels, default=st.session_state.get("selected_flights", []))
            selected, clipped = clamp_multiselect(selected, 3)
            if clipped:
                st.warning("You can select up to 3 flights. Keeping the first 3.")
            st.session_state["selected_flights"] = selected

            for i, f in enumerate(flights[:10]):
                dur = f.get("duration_min")
                dur_str = f"{dur} min" if isinstance(dur, (int, float)) else "N/A"

                with st.container(border=True):
                    top_l, top_r = st.columns([3, 1])
                    top_l.markdown(f"**{i+1}. {f['airline']} ({f['class']})**")
                    top_r.caption(f"{f['price']} • {f['stops']}")

                    r1, r2, r3 = st.columns(3)
                    r1.caption("Depart")
                    r1.write(f["depart"])
                    r2.caption("Arrive")
                    r2.write(f["arrive"])
                    r3.caption("Duration")
                    r3.write(dur_str)

                    # link (best-effort)
                    st.markdown(f"[Open in Google Flights]({f.get('link') or google_flights_search_url(origin, destination, st.session_state.get('outbound_date',''), st.session_state.get('return_date',''))})")

    # ---- Hotels ----
    with tab2:
        st.subheader("Hotel options")

        if not hotels_fmt:
            st.info("No hotel results found (or filtered out by rating). Try lowering min rating or changing location.")
        else:
            labels = []
            for i, h in enumerate(hotels_fmt):
                labels.append(f"{i+1}. {h['name']} • {h['price']} • ⭐ {h['rating']:.1f}")

            selected = st.multiselect("Select up to 3 hotels (personalizes AI picks)", options=labels, default=st.session_state.get("selected_hotels", []))
            selected, clipped = clamp_multiselect(selected, 3)
            if clipped:
                st.warning("You can select up to 3 hotels. Keeping the first 3.")
            st.session_state["selected_hotels"] = selected

            for i, h in enumerate(hotels_fmt[:12]):
                with st.container(border=True):
                    top_l, top_r = st.columns([3, 1])
                    top_l.markdown(f"**{i+1}. {h['name']}**")
                    top_r.caption(f"{h['price']} • ⭐ {h['rating']:.1f}")

                    r1, r2 = st.columns(2)
                    r1.caption("Area")
                    r1.write(h.get("location", "—") or "—")
                    r2.caption("Reviews")
                    r2.write(h.get("reviews", "—") or "—")

                    if h.get("link"):
                        st.markdown(f"[Open hotel listing]({h['link']})")

    # ---- AI Picks ----
    with tab3:
        st.subheader("AI recommendations (Gemini)")

        origin2 = st.session_state.get("origin", origin)
        dest2 = st.session_state.get("destination", destination)
        out_d2 = st.session_state.get("outbound_date", outbound_date)
        ret_d2 = st.session_state.get("return_date", return_date)
        hotel_loc2 = (st.session_state.get("location", "") or dest2).strip()

        flights_for_ai = flights[:6]
        hotels_for_ai = hotels_fmt[:6]

        if st.session_state.get("selected_flights"):
            idxs = []
            for s in st.session_state["selected_flights"]:
                try:
                    idxs.append(int(s.split(".")[0]) - 1)
                except Exception:
                    pass
            flights_for_ai = [flights[i] for i in idxs if 0 <= i < len(flights)] or flights_for_ai

        if st.session_state.get("selected_hotels"):
            idxs = []
            for s in st.session_state["selected_hotels"]:
                try:
                    idxs.append(int(s.split(".")[0]) - 1)
                except Exception:
                    pass
            hotels_for_ai = [hotels_fmt[i] for i in idxs if 0 <= i < len(hotels_fmt)] or hotels_for_ai

        if st.button("Generate AI Picks", type="primary"):
            flight_prompt = f"""
You are an expert travel assistant.

User preferences:
- Trip style: {style}
- Max stops preference: {max_stops} (prefer <= this, best-effort)
- Currency: {currency}

Task:
1) Pick the best 1-2 flights from the list.
2) Explain clearly in Markdown with bullet points.
3) Include the flight link if available.

Route: {origin2} -> {dest2}
Dates: {out_d2} to {ret_d2}

Flights data:
{flights_for_ai}
""".strip()

            hotel_prompt = f"""
You are an expert travel assistant.

User preferences:
- Trip style: {style}
- Minimum hotel rating: {min_hotel_rating}
- Currency: {currency}

Task:
1) Pick the best 2-3 hotels from the list.
2) Explain clearly in Markdown with bullet points.
3) Include links as plain text.

Location query: {hotel_loc2}
Dates: {out_d2} to {ret_d2}

Hotels data:
{hotels_for_ai}
""".strip()

            with st.spinner("Asking Gemini..."):
                st.session_state["ai_flights_md"] = run_async(gemini_call(client, GEMINI_MODEL, flight_prompt))
                st.session_state["ai_hotels_md"] = run_async(gemini_call(client, GEMINI_MODEL, hotel_prompt))

        if st.session_state.get("ai_flights_md"):
            st.markdown("### ✈️ Flights")
            st.markdown(st.session_state["ai_flights_md"])

        if st.session_state.get("ai_hotels_md"):
            st.markdown("### 🏨 Hotels")
            st.markdown(st.session_state["ai_hotels_md"])

    # ---- Itinerary + Export ----
    with tab4:
        st.subheader("Itinerary & Export")

        origin2 = st.session_state.get("origin", origin)
        dest2 = st.session_state.get("destination", destination)
        out_d2 = st.session_state.get("outbound_date", outbound_date)
        ret_d2 = st.session_state.get("return_date", return_date)

        if st.button("Generate Itinerary & Tips", type="primary"):
            itinerary_prompt = f"""
Create a practical day-by-day itinerary in Markdown for {dest2}
from {out_d2} to {ret_d2}.

User preferences:
- Trip style: {style}

Constraints:
- For each day: Morning / Afternoon / Evening
- Include 2-3 food ideas per day
- Keep it realistic and not overly packed
""".strip()

            tips_prompt = f"""
Give concise travel tips in Markdown for {dest2}:
- Best areas to stay (2-4 options)
- Getting around (local transit + passes)
- Budget tips (specific)
- 5 tourist mistakes to avoid
- 5 local etiquette tips
""".strip()

            with st.spinner("Generating itinerary & tips..."):
                st.session_state["itinerary_md"] = run_async(gemini_call(client, GEMINI_MODEL, itinerary_prompt))
                st.session_state["tips_md"] = run_async(gemini_call(client, GEMINI_MODEL, tips_prompt))

        if st.session_state.get("itinerary_md"):
            st.markdown("### 🗓️ Itinerary")
            st.markdown(st.session_state["itinerary_md"])

        if st.session_state.get("tips_md"):
            st.markdown("### 💡 Tips")
            st.markdown(st.session_state["tips_md"])

        st.markdown("---")
        export_md = f"""# AI Travel Planner Export

**Route:** {st.session_state.get("origin","")} → {st.session_state.get("destination","")}
**Dates:** {st.session_state.get("outbound_date","")} to {st.session_state.get("return_date","")}
**Trip style:** {style}
**Max stops preference:** {max_stops}
**Min hotel rating:** {min_hotel_rating}
**Currency:** {currency}

---

## AI Picks — Flights
{st.session_state.get("ai_flights_md","(not generated)")}

---

## AI Picks — Hotels
{st.session_state.get("ai_hotels_md","(not generated)")}

---

## Itinerary
{st.session_state.get("itinerary_md","(not generated)")}

---

## Tips
{st.session_state.get("tips_md","(not generated)")}
"""
        st.download_button(
            "Download Markdown",
            data=export_md,
            file_name="travel_plan.md",
            mime="text/markdown",
            type="primary",
        )
else:
    st.info("Enter your route + dates, then click **Search**.")
