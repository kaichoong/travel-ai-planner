# streamlit_app.py
import os
import asyncio
import random
import html
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Tuple

import streamlit as st
from serpapi import GoogleSearch
from google import genai

# -------------------------
# Helpers
# -------------------------
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
    Streamlit runs scripts in a ScriptRunner thread where 'current loop'
    assumptions can break. Creating a fresh event loop per run is reliable.
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

async def get_flights(
    serp_key: str,
    origin: str,
    destination: str,
    out_date: str,
    ret_date: str,
    currency: str = "USD",
) -> List[Dict]:
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

async def get_hotels(
    serp_key: str,
    location: str,
    check_in: str,
    check_out: str,
    currency: str = "USD",
    min_rating: float = 0.0,
) -> List[Dict]:
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

def fmt_flights(best_flights: List[Dict]) -> List[Dict]:
    out = []
    for f in best_flights[:12]:
        legs = f.get("flights") or []
        if not legs:
            continue

        leg0 = legs[0]
        dep = leg0.get("departure_airport", {}) or {}
        arr = leg0.get("arrival_airport", {}) or {}

        stops = "Nonstop" if len(legs) == 1 else f"{len(legs)-1} stop(s)"

        out.append({
            "airline": leg0.get("airline", "Unknown"),
            "price": f.get("price", "N/A"),
            "duration_min": f.get("total_duration", None),
            "stops": stops,
            "depart": f"{dep.get('id','')} {dep.get('time','')}".strip(),
            "arrive": f"{arr.get('id','')} {arr.get('time','')}".strip(),
            "class": leg0.get("travel_class", "Economy"),
            "raw": f,
        })
    return out

def fmt_hotels(props: List[Dict]) -> List[Dict]:
    out = []
    for h in props[:18]:
        rate = (h.get("rate_per_night", {}) or {}).get("lowest", "N/A")
        out.append({
            "name": h.get("name", "Unknown"),
            "price": rate,
            "rating": float(h.get("overall_rating", 0.0) or 0.0),
            "reviews": h.get("reviews", None),
            "location": h.get("location", ""),
            "link": h.get("link", ""),
            "raw": h,
        })
    return out

def _parse_price_to_number(price_val) -> float:
    if price_val is None:
        return float("inf")
    if isinstance(price_val, (int, float)):
        return float(price_val)
    s = str(price_val)
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

def card_html(title: str, rows: List[Tuple[str, str]], badge: str = "", link: str = "") -> str:
    """
    IMPORTANT: avoid leading indentation in the returned HTML.
    If HTML starts with 4 spaces, Streamlit treats it as a Markdown code block
    and you will see raw <div> tags in the UI.
    """
    title_e = html.escape(title)
    badge_e = html.escape(badge) if badge else ""
    link_e = html.escape(link) if link else ""

    rows_html = ""
    for k, v in rows:
        rows_html += (
            f'<div class="row">'
            f'<div class="k">{html.escape(k)}</div>'
            f'<div class="v">{html.escape(v)}</div>'
            f'</div>'
        )

    btn_html = ""
    if link_e:
        btn_html = f'<a class="btn" href="{link_e}" target="_blank" rel="noopener noreferrer">Open</a>'

    badge_html = f'<span class="badge">{badge_e}</span>' if badge_e else ""

    html_out = f"""
<div class="card">
  <div class="card-top">
    <div class="card-title">{title_e}</div>
    <div class="card-actions">
      {badge_html}
      {btn_html}
    </div>
  </div>
  <div class="card-body">
    {rows_html}
  </div>
</div>
"""
    return textwrap.dedent(html_out).strip()


# -------------------------
# UI (Premium Dark)
# -------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

CSS = """
<style>
:root{
  --bg:#0b0f17;
  --panel:#0f1626;
  --card:#101a2d;
  --muted:#9aa7bd;
  --text:#e9eef7;
  --line:rgba(255,255,255,0.07);
  --brand:#7cf5c5;
  --brand2:#6aa6ff;
  --shadow: 0 10px 28px rgba(0,0,0,0.35);
  --radius: 18px;
}

html, body, [class*="css"]  { background: var(--bg) !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1180px; }
h1, h2, h3, h4, h5, h6, p, li, span, label { color: var(--text) !important; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(16,26,45,1), rgba(11,15,23,1)) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div,
div[data-testid="stMultiSelect"] div,
div[data-testid="stDateInput"] input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}

.stButton>button {
  background: linear-gradient(90deg, var(--brand), var(--brand2));
  color: #071018 !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.7rem 1rem !important;
  font-weight: 800 !important;
  box-shadow: var(--shadow);
}
.stButton>button:hover { filter: brightness(1.05); transform: translateY(-1px); }

hr { border-color: var(--line) !important; }

.hero {
  background: radial-gradient(1000px 600px at 20% 0%, rgba(124,245,197,0.10), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(106,166,255,0.12), transparent 60%),
              linear-gradient(180deg, rgba(16,26,45,0.75), rgba(11,15,23,0.2));
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 18px 18px 12px 18px;
  box-shadow: var(--shadow);
  margin-bottom: 14px;
}
.hero-title { font-size: 1.8rem; font-weight: 900; letter-spacing: -0.02em; margin: 0; }
.hero-sub { color: var(--muted) !important; margin: 6px 0 0 0; }

.card {
  background: linear-gradient(180deg, rgba(16,26,45,0.92), rgba(16,26,45,0.75));
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 14px 14px 10px 14px;
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}
.card-top { display:flex; align-items:flex-start; justify-content:space-between; gap:10px; }
.card-title { font-weight: 850; font-size: 1.02rem; }
.card-actions { display:flex; align-items:center; gap:10px; }
.badge{
  font-size: 0.78rem;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.04);
  color: var(--text);
  white-space: nowrap;
}
.btn{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  font-weight:850;
  font-size:0.82rem;
  padding: 7px 10px;
  border-radius: 12px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.05);
  color: var(--text) !important;
  text-decoration:none !important;
}
.btn:hover{ filter: brightness(1.06); }

.card-body { margin-top: 10px; }
.row { display:flex; justify-content:space-between; gap:12px; padding: 7px 0; border-top: 1px solid var(--line); }
.row:first-child { border-top: 0; padding-top: 0; }
.k { color: var(--muted); font-size: 0.85rem; min-width: 120px; }
.v { color: var(--text); font-size: 0.90rem; text-align:right; }

.small-note { color: var(--muted) !important; font-size: 0.9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <div class="hero-title">✈️ AI Travel Planner</div>
  <div class="hero-sub">Search flights & hotels, pick what you like, then let Gemini craft your itinerary.</div>
</div>
""",
    unsafe_allow_html=True,
)

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
    max_stops = st.slider("Max stops (preference)", 0, 2, 1, help="Used as a preference in AI picks (best-effort).")
    min_hotel_rating = st.slider("Min hotel rating", 0.0, 9.5, 8.0, 0.5)
    flight_sort = st.selectbox("Sort flights", ["Cheapest", "Fastest", "Fewest stops"], index=0)
    hotel_sort = st.selectbox("Sort hotels", ["Top rated", "Cheapest"], index=0)
    currency = st.selectbox("Currency", ["USD", "SGD", "MYR", "JPY", "EUR", "GBP"], index=0)
    st.markdown("---")
    st.markdown(
        '<div class="small-note">Tip: Keep inputs simple (IATA codes + dates). Hit <b>Search</b>, then select up to 3 flights & 3 hotels for the AI.</div>',
        unsafe_allow_html=True,
    )

# -------------------------
# Inputs (Main)
# -------------------------
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
origin = c1.text_input("Origin (IATA)", value=st.session_state.get("origin", "SIN")).strip().upper()
destination = c2.text_input("Destination (IATA)", value=st.session_state.get("destination", "NRT")).strip().upper()
outbound_date = c3.text_input("Outbound date (YYYY-MM-DD)", value=st.session_state.get("outbound_date", "2026-02-10")).strip()
return_date = c4.text_input("Return date (YYYY-MM-DD)", value=st.session_state.get("return_date", "2026-02-15")).strip()

location = st.text_input(
    "Hotel location (optional — defaults to destination)",
    value=st.session_state.get("location", ""),
).strip()

btn_col1, btn_col2 = st.columns([1, 4])
search_clicked = btn_col1.button("Search", type="primary")
btn_col2.markdown(
    '<div class="small-note">You can keep refining inputs and re-search. The app stays single-file and redeploys on push.</div>',
    unsafe_allow_html=True,
)

# -------------------------
# Search + Store results
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
                get_flights(SERPAPI_API_KEY, origin, destination, out_d, ret_d, currency=currency),
                get_hotels(SERPAPI_API_KEY, hotel_loc, out_d, ret_d, currency=currency, min_rating=min_hotel_rating),
            )
            return flights_raw, hotels_raw

        with st.spinner("Searching flights & hotels..."):
            best_flights_raw, hotels_raw = run_async(fetch_all())

        flights = sort_flights(fmt_flights(best_flights_raw), flight_sort)
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
# Display if we have results
# -------------------------
flights = st.session_state.get("flights", [])
hotels_fmt = st.session_state.get("hotels", [])

if flights or hotels_fmt:
    tab1, tab2, tab3, tab4 = st.tabs(["✈️ Flights", "🏨 Hotels", "✨ AI Picks", "🗺️ Itinerary & Export"])

    # ---- Flights tab ----
    with tab1:
        st.markdown("### Flight options")
        if not flights:
            st.info("No flight results found. Try different dates or route.")
        else:
            flight_labels = []
            for i, f in enumerate(flights):
                price = str(f.get("price", "N/A"))
                label = f"{i+1}. {f.get('airline','Unknown')} • {price} • {f.get('stops','')} • {f.get('depart','')} → {f.get('arrive','')}"
                flight_labels.append(label)

            selected_flights = st.multiselect(
                "Select up to 3 flights (used to personalize AI picks)",
                options=flight_labels,
                default=st.session_state.get("selected_flights", []),
            )
            selected_flights, clipped = clamp_multiselect(selected_flights, 3)
            if clipped:
                st.warning("You can select up to 3 flights. Keeping the first 3.")
            st.session_state["selected_flights"] = selected_flights

            for i, f in enumerate(flights[:10]):
                badge = f"{f.get('price','N/A')} • {f.get('stops','')}"
                dur = f.get("duration_min", None)
                dur_str = f"{dur} min" if isinstance(dur, (int, float)) else "N/A"
                st.markdown(
                    card_html(
                        title=f"{i+1}. {f.get('airline','Unknown')} ({f.get('class','Economy')})",
                        badge=badge,
                        rows=[
                            ("Depart", f.get("depart", "")),
                            ("Arrive", f.get("arrive", "")),
                            ("Duration", dur_str),
                        ],
                    ),
                    unsafe_allow_html=True,
                )

    # ---- Hotels tab ----
    with tab2:
        st.markdown("### Hotel options")
        if not hotels_fmt:
            st.info("No hotel results found (or filtered out by rating). Try lowering min rating or changing location.")
        else:
            hotel_labels = []
            for i, h in enumerate(hotels_fmt):
                label = f"{i+1}. {h.get('name','Unknown')} • {h.get('price','N/A')} • ⭐ {h.get('rating',0):.1f}"
                hotel_labels.append(label)

            selected_hotels = st.multiselect(
                "Select up to 3 hotels (used to personalize AI picks)",
                options=hotel_labels,
                default=st.session_state.get("selected_hotels", []),
            )
            selected_hotels, clipped = clamp_multiselect(selected_hotels, 3)
            if clipped:
                st.warning("You can select up to 3 hotels. Keeping the first 3.")
            st.session_state["selected_hotels"] = selected_hotels

            for i, h in enumerate(hotels_fmt[:12]):
                badge = f"{h.get('price','N/A')} • ⭐ {h.get('rating',0):.1f}"
                link = h.get("link", "") or ""
                st.markdown(
                    card_html(
                        title=f"{i+1}. {h.get('name','Unknown')}",
                        badge=badge,
                        link=link,
                        rows=[
                            ("Area", h.get("location", "") or "—"),
                            ("Reviews", str(h.get("reviews", "—"))),
                        ],
                    ),
                    unsafe_allow_html=True,
                )

    # ---- AI Picks tab ----
    with tab3:
        st.markdown("### AI recommendations (Gemini)")

        origin = st.session_state.get("origin", origin)
        destination = st.session_state.get("destination", destination)
        out_d = st.session_state.get("outbound_date", outbound_date)
        ret_d = st.session_state.get("return_date", return_date)
        hotel_loc = (st.session_state.get("location", "") or destination).strip()

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

        gen_col1, gen_col2 = st.columns([1, 3])
        generate_ai = gen_col1.button("Generate AI Picks", type="primary")
        gen_col2.markdown(
            '<div class="small-note">Uses your selected flights/hotels (or top results if none selected) + Trip Builder preferences.</div>',
            unsafe_allow_html=True,
        )

        if generate_ai:
            flight_prompt = f"""
You are an expert travel assistant.

User preferences:
- Trip style: {style}
- Max stops preference: {max_stops} (prefer <= this, best-effort)
- Currency: {currency}

Task:
1) Pick the best 1-2 flights from the list.
2) Explain clearly in Markdown with bullet points.
3) Output a short "Recommended" section plus a short "Why" section.

Route: {origin} -> {destination}
Dates: {out_d} to {ret_d}

Flights data (JSON-ish):
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
3) If a link exists, include it as plain text (no HTML).

Location query: {hotel_loc}
Dates: {out_d} to {ret_d}

Hotels data (JSON-ish):
{hotels_for_ai}
""".strip()

            with st.spinner("Asking Gemini..."):
                ai_flights_md = run_async(gemini_call(client, GEMINI_MODEL, flight_prompt))
                ai_hotels_md = run_async(gemini_call(client, GEMINI_MODEL, hotel_prompt))

            st.session_state["ai_flights_md"] = ai_flights_md
            st.session_state["ai_hotels_md"] = ai_hotels_md

        if st.session_state.get("ai_flights_md"):
            st.markdown("#### ✈️ Flights")
            st.markdown(st.session_state["ai_flights_md"])

        if st.session_state.get("ai_hotels_md"):
            st.markdown("#### 🏨 Hotels")
            st.markdown(st.session_state["ai_hotels_md"])

    # ---- Itinerary tab ----
    with tab4:
        st.markdown("### Itinerary & tips")

        origin = st.session_state.get("origin", origin)
        destination = st.session_state.get("destination", destination)
        out_d = st.session_state.get("outbound_date", outbound_date)
        ret_d = st.session_state.get("return_date", return_date)

        generate_itin = st.button("Generate Itinerary & Tips", type="primary")

        if generate_itin:
            itinerary_prompt = f"""
Create a practical day-by-day itinerary in Markdown for {destination}
from {out_d} to {ret_d}.

User preferences:
- Trip style: {style}

Constraints:
- For each day: Morning / Afternoon / Evening
- Include 2-3 food ideas per day
- Keep it realistic and not overly packed
""".strip()

            tips_prompt = f"""
Give concise travel tips in Markdown for {destination}:
- Best areas to stay (2-4 options)
- Getting around (local transit + passes)
- Budget tips (specific)
- 5 tourist mistakes to avoid
- 5 local etiquette tips
""".strip()

            with st.spinner("Generating itinerary & tips..."):
                itinerary_md = run_async(gemini_call(client, GEMINI_MODEL, itinerary_prompt))
                tips_md = run_async(gemini_call(client, GEMINI_MODEL, tips_prompt))

            st.session_state["itinerary_md"] = itinerary_md
            st.session_state["tips_md"] = tips_md

        if st.session_state.get("itinerary_md"):
            st.markdown("#### 🗓️ Itinerary")
            st.markdown(st.session_state["itinerary_md"])

        if st.session_state.get("tips_md"):
            st.markdown("#### 💡 Tips")
            st.markdown(st.session_state["tips_md"])

        st.markdown("---")
        st.markdown("### Export")

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
