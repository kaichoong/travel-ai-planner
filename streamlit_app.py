# streamlit_app.py
import os
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from serpapi import GoogleSearch
from google import genai


# -------------------------
# Helpers
# -------------------------
def normalize_date(date_str: str) -> str:
    if not date_str:
        raise ValueError("Date is required.")
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


def serpapi_search(params: dict) -> dict:
    # SerpAPI Python client is sync, so keep it sync for Streamlit.
    return GoogleSearch(params).get_dict()


def gemini_call(
    client: genai.Client,
    model: str,
    prompt: str,
    retries: int = 5,
) -> str:
    """
    Sync Gemini call with exponential backoff for overload/quota.
    Never raises; always returns a string.
    """
    def _call() -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return _extract_text(resp) or "⚠️ No response returned by model."

    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            return _call()
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Retry transient + quota signals
            retryable = (
                ("503" in msg) or ("overloaded" in msg) or ("unavailable" in msg) or
                ("429" in msg) or ("quota" in msg) or ("resource_exhausted" in msg) or
                ("rate" in msg and "limit" in msg)
            )
            if retryable:
                wait = (2 ** i) + random.uniform(0, 0.8)
                time.sleep(wait)
                continue

            return f"⚠️ AI error: {e}"

    return f"⚠️ AI is busy / quota-limited. Please try again shortly.\n\nDetails: {last_err}"


def get_flights(
    serp_key: str,
    origin: str,
    destination: str,
    out_date: str,
    ret_date: str,
) -> List[Dict]:
    params = {
        "engine": "google_flights",
        "api_key": serp_key,
        "departure_id": origin.upper().strip(),
        "arrival_id": destination.upper().strip(),
        "outbound_date": out_date,
        "return_date": ret_date,
        "currency": "USD",
        "hl": "en",
        "gl": "us",
    }
    results = serpapi_search(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    return results.get("best_flights", []) or []


def get_hotels(
    serp_key: str,
    location: str,
    check_in: str,
    check_out: str,
) -> List[Dict]:
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
    results = serpapi_search(params)
    if "error" in results:
        raise RuntimeError(results["error"])
    return results.get("properties", []) or []


def fmt_flights(best_flights: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
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
            "stops": "Nonstop" if len(legs) == 1 else f"{len(legs) - 1} stop(s)",
            "depart": f"{dep.get('id','')} {dep.get('time','')}",
            "arrive": f"{arr.get('id','')} {arr.get('time','')}",
            "class": leg.get("travel_class", "Economy"),
        })
    return out


def fmt_hotels(props: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for h in props[:10]:
        out.append({
            "name": h.get("name", "Unknown"),
            "price": (h.get("rate_per_night", {}) or {}).get("lowest", "N/A"),
            "rating": h.get("overall_rating", 0.0) or 0.0,
            "location": h.get("location", ""),
            "link": h.get("link", ""),
        })
    return out


def _cache_key(origin: str, destination: str, out_d: str, ret_d: str, hotel_loc: str) -> str:
    return f"{origin.upper().strip()}|{destination.upper().strip()}|{out_d}|{ret_d}|{hotel_loc.strip().lower()}"


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
    st.error("Missing API keys. Set GEMINI_API_KEY and SERPAPI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# Optional: reduce quota hits by caching previous run in session_state
if "results_cache" not in st.session_state:
    st.session_state["results_cache"] = {}

col1, col2, col3, col4 = st.columns(4)
origin = col1.text_input("Origin (IATA)", value="SIN")
destination = col2.text_input("Destination (IATA)", value="NRT")
outbound_date = col3.text_input("Outbound date (YYYY-MM-DD)", value="2026-02-10")
return_date = col4.text_input("Return date (YYYY-MM-DD)", value="2026-02-15")

location = st.text_input("Hotel location (optional, defaults to destination city/code)", value="")

search_clicked = st.button("Search", type="primary")

if search_clicked:
    try:
        out_d = normalize_date(outbound_date)
        ret_d = normalize_date(return_date)
        hotel_loc = (location.strip() or destination.strip())

        key = _cache_key(origin, destination, out_d, ret_d, hotel_loc)

        # --- Step 1: SerpAPI (cacheable) ---
        with st.spinner("Searching flights & hotels..."):
            if key in st.session_state["results_cache"]:
                best_flights, hotels = st.session_state["results_cache"][key]
            else:
                best_flights = get_flights(SERPAPI_API_KEY, origin, destination, out_d, ret_d)
                hotels = get_hotels(SERPAPI_API_KEY, hotel_loc, out_d, ret_d)
                st.session_state["results_cache"][key] = (best_flights, hotels)

        flights = fmt_flights(best_flights)
        hotels_fmt = fmt_hotels(hotels)

        tab1, tab2, tab3, tab4 = st.tabs(["Flights", "Hotels", "AI Picks", "Itinerary & Tips"])

        with tab1:
            st.subheader("Top flight options")
            if flights:
                st.dataframe(flights, use_container_width=True)
            else:
                st.warning("No flights returned. Try different dates/route.")

        with tab2:
            st.subheader("Top hotel options")
            if hotels_fmt:
                st.dataframe(hotels_fmt, use_container_width=True)
            else:
                st.warning("No hotels returned. Try a different location.")

        with tab3:
            st.subheader("AI recommendations")

            flight_prompt = f"""Recommend the best 1-2 flights from these options (JSON). Explain briefly in Markdown.
Route: {origin.upper()} -> {destination.upper()}
Dates: {out_d} to {ret_d}
Flights JSON: {flights[:5]}
"""

            hotel_prompt = f"""Recommend the best 2-3 hotels from these options (JSON). Explain briefly in Markdown. Include links if present.
Location: {hotel_loc}
Dates: {out_d} to {ret_d}
Hotels JSON: {hotels_fmt[:5]}
"""

            with st.spinner("Asking Gemini..."):
                # sequential to reduce rate/overload
                ai_flights = gemini_call(client, GEMINI_MODEL, flight_prompt)
                ai_hotels = gemini_call(client, GEMINI_MODEL, hotel_prompt)

            st.markdown("### Flights")
            st.markdown(ai_flights)

            st.markdown("### Hotels")
            st.markdown(ai_hotels)

        with tab4:
            itinerary_prompt = f"""Create a practical day-by-day itinerary in Markdown for {destination.upper()}
from {out_d} to {ret_d}. Include morning/afternoon/evening and 2-3 food ideas per day.
"""

            tips_prompt = f"""Give concise travel tips for {destination.upper()} in Markdown:
- Best areas to stay
- Getting around
- Budget tips
- 5 tourist mistakes to avoid
"""

            with st.spinner("Generating itinerary & tips..."):
                itinerary = gemini_call(client, GEMINI_MODEL, itinerary_prompt)
                tips = gemini_call(client, GEMINI_MODEL, tips_prompt)

            st.markdown("### Itinerary")
            st.markdown(itinerary)

            st.markdown("### Tips")
            st.markdown(tips)

    except Exception as e:
        st.error(str(e))
