import os
import uvicorn
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from serpapi import GoogleSearch
from datetime import datetime


# -------------------------------------------------
# API KEYS
# -------------------------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "your_gemini_api_key_here")
SERP_API_KEY = os.getenv("SERP_API_KEY", "your_serpapi_key_here")

# -------------------------------------------------
# LOGGER
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------
# GEMINI CLIENT
# -------------------------------------------------
#gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
class FlightRequest(BaseModel):
    origin: str
    destination: str
    outbound_date: str  # YYYY-MM-DD
    return_date: str    # YYYY-MM-DD


class HotelRequest(BaseModel):
    location: str
    check_in_date: str  # YYYY-MM-DD
    check_out_date: str # YYYY-MM-DD


class ItineraryRequest(BaseModel):
    destination: str
    check_in_date: str
    check_out_date: str
    flights: str
    hotels: str


class FlightInfo(BaseModel):
    airline: str
    price: str
    duration: str
    stops: str
    departure: str
    arrival: str
    travel_class: str
    return_date: str
    airline_logo: str


class HotelInfo(BaseModel):
    name: str
    price: str
    rating: float
    location: str
    link: str


class AIResponse(BaseModel):
    flights: List[FlightInfo] = []
    hotels: List[HotelInfo] = []
    ai_flight_recommendation: str = ""
    ai_hotel_recommendation: str = ""
    itinerary: str = ""
    destination_insights: str = ""   # NEW: reviews/feedback summary


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Travel Planning API", version="2.0.0")


# -------------------------------------------------
# SerpAPI helpers
# -------------------------------------------------
async def run_search(params):
    """Run SerpAPI queries in a thread so we don't block the event loop."""
    try:
        return await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())
    except Exception as e:
        logger.exception(f"SerpAPI search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")


async def search_flights(flight_request: FlightRequest):
    """Fetch flights from Google Flights via SerpAPI."""
    logger.info(f"Searching flights: {flight_request.origin} → {flight_request.destination}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": flight_request.origin.strip().upper(),
        "arrival_id": flight_request.destination.strip().upper(),
        "outbound_date": flight_request.outbound_date,
        "return_date": flight_request.return_date,
        "currency": "USD",
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Flight search error: {search_results['error']}")
        return {"error": search_results["error"]}

    best_flights = search_results.get("best_flights", [])
    if not best_flights:
        logger.warning("No flights found in search results")
        return []

    formatted_flights: List[FlightInfo] = []
    for flight in best_flights:
        if not flight.get("flights"):
            continue

        first_leg = flight["flights"][0]
        dep_airport = first_leg.get("departure_airport", {})
        arr_airport = first_leg.get("arrival_airport", {})

        formatted_flights.append(
            FlightInfo(
                airline=first_leg.get("airline", "Unknown Airline"),
                price=str(flight.get("price", "N/A")),
                duration=f"{flight.get('total_duration', 'N/A')} min",
                stops="Nonstop"
                if len(flight["flights"]) == 1
                else f"{len(flight['flights']) - 1} stop(s)",
                departure=f"{dep_airport.get('name', 'Unknown')} "
                          f"({dep_airport.get('id', '???')}) at {dep_airport.get('time', 'N/A')}",
                arrival=f"{arr_airport.get('name', 'Unknown')} "
                        f"({arr_airport.get('id', '???')}) at {arr_airport.get('time', 'N/A')}",
                travel_class=first_leg.get("travel_class", "Economy"),
                return_date=flight_request.return_date,
                airline_logo=first_leg.get("airline_logo", ""),
            )
        )

    logger.info(f"Found {len(formatted_flights)} flights")
    return formatted_flights


async def search_hotels(hotel_request: HotelRequest):
    """Fetch hotels from Google Hotels via SerpAPI."""
    logger.info(f"Searching hotels in: {hotel_request.location}")

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_hotels",
        "q": hotel_request.location,
        "hl": "en",
        "gl": "us",
        "check_in_date": hotel_request.check_in_date,
        "check_out_date": hotel_request.check_out_date,
        "currency": "USD",
        "sort_by": 3,   # price / rating mix
        "rating": 8,    # filter low-rated
    }

    search_results = await run_search(params)

    if "error" in search_results:
        logger.error(f"Hotel search error: {search_results['error']}")
        return {"error": search_results["error"]}

    hotel_properties = search_results.get("properties", [])
    if not hotel_properties:
        logger.warning("No hotels found in search results")
        return []

    formatted_hotels: List[HotelInfo] = []
    for hotel in hotel_properties:
        try:
            formatted_hotels.append(
                HotelInfo(
                    name=hotel.get("name", "Unknown Hotel"),
                    price=hotel.get("rate_per_night", {}).get("lowest", "N/A"),
                    rating=hotel.get("overall_rating", 0.0),
                    location=hotel.get("location", "N/A"),
                    link=hotel.get("link", "N/A"),
                )
            )
        except Exception as e:
            logger.warning(f"Error formatting hotel data: {e}")

    logger.info(f"Found {len(formatted_hotels)} hotels")
    return formatted_hotels


# -------------------------------------------------
# Formatting helpers
# -------------------------------------------------
def format_travel_data(data_type: str, data):
    """Convert flights/hotels into plain-text summaries for Gemini."""
    if not data:
        return f"No {data_type} available."

    if data_type == "flights":
        text = "Available flight options:\n\n"
        for i, f in enumerate(data):
            text += (
                f"Flight {i+1}:\n"
                f"- Airline: {f.airline}\n"
                f"- Price: ${f.price}\n"
                f"- Duration: {f.duration}\n"
                f"- Stops: {f.stops}\n"
                f"- Departure: {f.departure}\n"
                f"- Arrival: {f.arrival}\n"
                f"- Class: {f.travel_class}\n\n"
            )
        return text

    if data_type == "hotels":
        text = "Available hotel options:\n\n"
        for i, h in enumerate(data):
            text += (
                f"Hotel {i+1}:\n"
                f"- Name: {h.name}\n"
                f"- Price: ${h.price}\n"
                f"- Rating: {h.rating}\n"
                f"- Location: {h.location}\n"
                f"- Link: {h.link}\n\n"
            )
        return text

    return "Invalid data type."


# -------------------------------------------------
# Gemini helpers
# -------------------------------------------------
async def get_ai_recommendation(data_type: str, formatted_data: str) -> str:
    """Flight / hotel recommendation via Gemini."""
    logger.info(f"Getting {data_type} recommendation from Gemini")

    if data_type == "flights":
        instructions = (
            "You are an AI flight analyst. From the following flight options, "
            "recommend the single best flight for a typical traveller who cares "
            "about value, reasonable duration, and minimal stops.\n\n"
            "Return your answer as:\n"
            "- A short summary paragraph.\n"
            "- 3–5 bullet points with key reasons (price, duration, stops, convenience).\n"
        )
    else:
        instructions = (
            "You are an AI hotel analyst. From the following hotel options, "
            "recommend one or two of the best hotels balancing price, rating, "
            "and location.\n\n"
            "Return your answer as:\n"
            "- A short summary paragraph.\n"
            "- 3–5 bullet points naming specific hotels and why they are ideal.\n"
        )

    prompt = f"{instructions}\n\nHere are the {data_type} options:\n\n{formatted_data}"

    def _call():
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return getattr(resp, "text", "") or "No response from Gemini."

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.exception(f"Error in Gemini {data_type} analysis: {e}")
        return f"Unable to generate {data_type} recommendation due to an error."


async def generate_itinerary(destination: str,
                             flights_text: str,
                             hotels_text: str,
                             check_in_date: str,
                             check_out_date: str) -> str:
    """Generate a day-by-day itinerary via Gemini."""
    logger.info(f"Generating itinerary for {destination}")

    try:
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")
        days = (check_out - check_in).days
    except Exception:
        days = "several"

    prompt = f"""
You are an expert travel planner.

Create a detailed {days}-day itinerary for a trip to **{destination}**
from **{check_in_date}** to **{check_out_date}**.

Use these flights:
{flights_text}

Use these hotels:
{hotels_text}

Requirements:
- Output in Markdown.
- Start with a short trip overview.
- Then create sections "Day 1", "Day 2", etc.
- For each day include:
  - Morning / afternoon / evening activities.
  - 1–2 attraction suggestions with short descriptions.
  - 1–2 food/restaurant suggestions.
  - Simple transport tips (walk, metro, taxi, etc.).
- Use some emojis: 🏛️ for sights, 🌳 parks, 🍽️ food, 🛍️ shopping, 🚆 transport, etc.

Return ONLY the itinerary in Markdown.
"""

    def _call():
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return getattr(resp, "text", "") or "No itinerary generated."

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.exception(f"Error generating itinerary: {e}")
        return "Unable to generate itinerary due to an error. Please try again later."


async def get_destination_insights(destination: str) -> str:
    """Summarise reviews & feedback for the destination via Gemini."""
    logger.info(f"Generating destination insights for {destination}")

    prompt = f"""
You are a travel review analyst.

Summarise traveller reviews and feedback for **{destination}**.

Return in Markdown with these sections:

### 🌟 Overall Sentiment
- 2–3 bullets describing the general vibe (safe, busy, pricey, friendly, etc.)

### ❤️ Things People Love
- 4–6 bullets about food, attractions, nightlife, people, nature, etc.

### 😬 Common Complaints
- 3–5 bullets (crowds, scams, weather, traffic, noise, etc.)

### 💡 Local Tips
- 3–5 practical tips (best months to visit, transport hacks, etiquette, safety reminders).

Keep it concise, friendly, and genuinely useful for someone planning a trip.
"""

    def _call():
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return getattr(resp, "text", "") or "No insights available."

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.exception(f"Destination insights error: {e}")
        return "Unable to generate destination insights at the moment."


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.post("/search_flights/", response_model=AIResponse)
async def get_flight_recommendations(flight_request: FlightRequest):
    """Search flights and get AI recommendation."""
    try:
        flights = await search_flights(flight_request)

        if isinstance(flights, dict) and "error" in flights:
            raise HTTPException(status_code=400, detail=flights["error"])

        if not flights:
            raise HTTPException(status_code=404, detail="No flights found")

        flights_text = format_travel_data("flights", flights)
        ai_recommendation = await get_ai_recommendation("flights", flights_text)

        return AIResponse(
            flights=flights,
            ai_flight_recommendation=ai_recommendation,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Flight search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Flight search error: {str(e)}")


@app.post("/search_hotels/", response_model=AIResponse)
async def get_hotel_recommendations(hotel_request: HotelRequest):
    """Search hotels and get AI recommendation."""
    try:
        hotels = await search_hotels(hotel_request)

        if isinstance(hotels, dict) and "error" in hotels:
            raise HTTPException(status_code=400, detail=hotels["error"])

        if not hotels:
            raise HTTPException(status_code=404, detail="No hotels found")

        hotels_text = format_travel_data("hotels", hotels)
        ai_recommendation = await get_ai_recommendation("hotels", hotels_text)

        return AIResponse(
            hotels=hotels,
            ai_hotel_recommendation=ai_recommendation,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Hotel search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Hotel search error: {str(e)}")


@app.post("/complete_search/", response_model=AIResponse)
async def complete_travel_search(
    flight_request: FlightRequest,
    hotel_request: Optional[HotelRequest] = None,
):
    """Search flights + hotels concurrently, then AI recommendations + itinerary + destination insights."""
    try:
        # If hotel request isn't provided, infer from flight request
        if hotel_request is None:
            hotel_request = HotelRequest(
                location=flight_request.destination,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date,
            )

        # Run flight + hotel logic in parallel
        flight_task = asyncio.create_task(get_flight_recommendations(flight_request))
        hotel_task = asyncio.create_task(get_hotel_recommendations(hotel_request))

        flight_results, hotel_results = await asyncio.gather(
            flight_task, hotel_task, return_exceptions=True
        )

        if isinstance(flight_results, Exception):
            logger.error(f"Flight search failed: {flight_results}")
            flight_results = AIResponse(
                flights=[], ai_flight_recommendation="Could not retrieve flights."
            )

        if isinstance(hotel_results, Exception):
            logger.error(f"Hotel search failed: {hotel_results}")
            hotel_results = AIResponse(
                hotels=[], ai_hotel_recommendation="Could not retrieve hotels."
            )

        flights_text = format_travel_data("flights", flight_results.flights)
        hotels_text = format_travel_data("hotels", hotel_results.hotels)

        itinerary = ""
        if flight_results.flights and hotel_results.hotels:
            itinerary = await generate_itinerary(
                destination=flight_request.destination,
                flights_text=flights_text,
                hotels_text=hotels_text,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date,
            )

        # NEW: Destination reviews & feedback
        destination_insights = await get_destination_insights(
            flight_request.destination
        )

        return AIResponse(
            flights=flight_results.flights,
            hotels=hotel_results.hotels,
            ai_flight_recommendation=flight_results.ai_flight_recommendation,
            ai_hotel_recommendation=hotel_results.ai_hotel_recommendation,
            itinerary=itinerary,
            destination_insights=destination_insights,
        )
    except Exception as e:
        logger.exception(f"Complete travel search error: {e}")
        raise HTTPException(status_code=500, detail=f"Travel search error: {str(e)}")


@app.post("/generate_itinerary/", response_model=AIResponse)
async def get_itinerary(itinerary_request: ItineraryRequest):
    """Generate an itinerary from given flight + hotel summaries."""
    try:
        itinerary = await generate_itinerary(
            destination=itinerary_request.destination,
            flights_text=itinerary_request.flights,
            hotels_text=itinerary_request.hotels,
            check_in_date=itinerary_request.check_in_date,
            check_out_date=itinerary_request.check_out_date,
        )
        return AIResponse(itinerary=itinerary)
    except Exception as e:
        logger.exception(f"Itinerary generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Itinerary generation error: {str(e)}")


# -------------------------------------------------
# Run server
# -------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Travel Planning API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
