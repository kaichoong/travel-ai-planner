# streamlit_app.py
import os
import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List

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

async def serpapi_search(params: dict) -> dict:
    return await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())

async def gemini_call(client, model: str, prompt: str, retries: int = 5) -> str:
    def _call() -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return _extract_text(resp)

    for i in range(retries):
        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            msg = str(e).lower()
            # retry overload/rate issues
            if ("503" in msg) or ("overloaded" in msg) or ("unavailable" in msg) or ("429" in msg) or ("quota" in msg):
                wait = (2 ** i) + random.uniform(0, 0.8)
                await asyncio.sleep(wait)
                continue
            return f"⚠️ AI error: {e}"
    return "⚠️ AI is busy / quota-limited. Please try again shortly."

async def get_flights(serp_key: str, origin: str, destination: str, out_date: str, ret_date: str) -> List[Dict]:
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
    results = await serpapi_search(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    return results.get("best_flights", []) or []

async def get_hotels(serp_key: str, location: str, check_in: str, check_out: str) -> List[Dict]:
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
    results = await serpapi_search(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    return results.get("properties", []) or []

def fmt_flights(best_flights: List[Dict]) -> List[Dict]:
    out = []
    for f in best_flights[:8]:
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
    for h in props[:10]:
        out.append({
            "name": h.get("name", "Unknown"),
            "price": (h.get("rate_per_night", {}) or {}).get("lowest", "N/A"),
            "rating": h.get("overall_rating", 0.0) or 0.0,
            "location": h.get("location", ""),
            "link": h.get("link", ""),
        })
    return out

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")
st.title("✈️ AI Travel Planner")

# Secrets (Cloud) or env (local)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY"))
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

if not GEMINI_API_KEY or not SERPAPI_API_KEY:
    st.error("Missing API keys. Set GEMINI_API_KEY and SERPAPI_API_KEY in Streamlit Secrets (or .env locally).")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

col1, col2, col3, col4 = st.columns(4)
origin = col1.text_input("Origin (IATA)", value="SIN")
destination = col2.text_input("Destination (IATA)", value="NRT")
outbound_date = col3.text_input("Outbound date (YYYY-MM-DD)", value="2026-02-10")
return_date = col4.text_input("Return date (YYYY-MM-DD)", value="2026-02-15")

location = st.text_input("Hotel location (optional, defaults to destination city/code)", value="")

if st.button("Search", type="primary"):
    try:
        out_d = normalize_date(outbound_date)
        ret_d = normalize_date(return_date)
        hotel_loc = location.strip() or destination.strip()

        with st.spinner("Searching flights & hotels..."):
            best_flights, hotels = asyncio.run(asyncio.gather(
                get_flights(SERPAPI_API_KEY, origin, destination, out_d, ret_d),
                get_hotels(SERPAPI_API_KEY, hotel_loc, out_d, ret_d),
            ))

        flights = fmt_flights(best_flights)
        hotels_fmt = fmt_hotels(hotels)

        tab1, tab2, tab3, tab4 = st.tabs(["Flights", "Hotels", "AI Picks", "Itinerary & Tips"])

        with tab1:
            st.subheader("Top flight options")
            st.dataframe(flights, use_container_width=True)

        with tab2:
            st.subheader("Top hotel options")
            st.dataframe(hotels_fmt, use_container_width=True)

        with tab3:
            st.subheader("AI recommendations")
            flight_prompt = f"""Recommend the best 1-2 flights from these options (JSON). Explain briefly in Markdown.
Route: {origin.upper()} -> {destination.upper()}
Dates: {out_d} to {ret_d}
Flights JSON: {flights[:5]}"""
            hotel_prompt = f"""Recommend the best 2-3 hotels from these options (JSON). Explain briefly in Markdown. Include links if present.
Location: {hotel_loc}
Dates: {out_d} to {ret_d}
Hotels JSON: {hotels_fmt[:5]}"""

            with st.spinner("Asking Gemini..."):
                # sequential to reduce rate/overload
                ai_flights = asyncio.run(gemini_call(client, GEMINI_MODEL, flight_prompt))
                ai_hotels = asyncio.run(gemini_call(client, GEMINI_MODEL, hotel_prompt))

            st.markdown("### Flights")
            st.markdown(ai_flights)
            st.markdown("### Hotels")
            st.markdown(ai_hotels)

        with tab4:
            itinerary_prompt = f"""Create a practical day-by-day itinerary in Markdown for {destination.upper()}
from {out_d} to {ret_d}. Include morning/afternoon/evening and 2-3 food ideas per day."""
            tips_prompt = f"""Give concise travel tips for {destination.upper()} in Markdown:
- Best areas to stay
- Getting around
- Budget tips
- 5 tourist mistakes to avoid"""

            with st.spinner("Generating itinerary & tips..."):
                itinerary = asyncio.run(gemini_call(client, GEMINI_MODEL, itinerary_prompt))
                tips = asyncio.run(gemini_call(client, GEMINI_MODEL, tips_prompt))

            st.markdown("### Itinerary")
            st.markdown(itinerary)
            st.markdown("### Tips")
            st.markdown(tips)

    except Exception as e:
        st.error(str(e))
