import json

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import _workflow_stage
from app.tools import build_travel_itinerary


def test_build_travel_itinerary_returns_structured_output():
    payload = build_travel_itinerary.invoke(
        {
            "destination": "Barcelona",
            "query_type": "hotels",
            "trip_length_days": 4,
            "trip_style": "luxury",
            "start_date": "2026-06-10",
            "end_date": "2026-06-14",
            "travelers": 2,
            "must_include": ["booking links", "beach access"],
            "options": [
                {
                    "name": "Hotel Arts Barcelona",
                    "category": "hotel",
                    "location": "Barcelona",
                    "summary": "Luxury waterfront hotel near the beach.",
                    "price": "$420 per night",
                    "rating": "4.6/5",
                    "booking_link": "https://example.com/hotel-arts",
                    "notes": ["Great sea views", "Close to nightlife"],
                    "sources": ["https://example.com/hotel-arts"],
                }
            ],
            "itinerary": [
                {
                    "day": 1,
                    "theme": "Arrival and waterfront dinner",
                    "morning": "Check in and settle into the hotel",
                    "afternoon": "Walk along Barceloneta Beach",
                    "evening": "Dinner by the marina",
                    "booking_links": ["https://example.com/day-1-booking"],
                }
            ],
            "recommendation_basis": "Best for travelers prioritizing location and premium amenities.",
        }
    )

    result = json.loads(payload)

    assert result["destination"] == "Barcelona"
    assert result["query_type"] == "hotels"
    assert result["trip_length_days"] == 4
    assert result["trip_style"] == "luxury"
    assert result["dates"]["start_date"] == "2026-06-10"
    assert result["must_include"] == ["booking links", "beach access"]
    assert result["options"][0]["name"] == "Hotel Arts Barcelona"
    assert result["options"][0]["price"] == "$420 per night"
    assert result["itinerary"][0]["day"] == 1
    assert result["itinerary"][0]["booking_links"] == ["https://example.com/day-1-booking"]
    assert result["recommendation_basis"] == (
        "Best for travelers prioritizing location and premium amenities."
    )


def test_build_travel_itinerary_handles_missing_fields():
    payload = build_travel_itinerary.invoke(
        {
            "destination": "Rome",
            "query_type": "attractions",
            "options": [
                {
                    "name": "Colosseum",
                    "category": "attraction",
                    "location": "Rome",
                    "summary": "Historic amphitheater and landmark.",
                    "sources": ["https://example.com/colosseum"],
                }
            ],
        }
    )

    result = json.loads(payload)

    assert result["trip_length_days"] is None
    assert result["dates"]["start_date"] is None
    assert result["must_include"] == []
    assert result["options"][0]["price"] == "Not provided"
    assert result["options"][0]["rating"] == "Not provided"
    assert result["options"][0]["booking_link"] is None
    assert result["itinerary"] == []


def test_workflow_stage_uses_structure_step_for_general_travel_plans():
    stage = _workflow_stage(
        [HumanMessage(content="Plan a 4-day honeymoon in Bali with hotels, beaches, and spas.")]
    )

    assert stage == "travel_structure"


def test_workflow_stage_moves_from_search_to_structure_then_finalize():
    messages = [
        HumanMessage(content="Find current flight options to Singapore for June 2026 and build an itinerary."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "duckduckgo_search",
                    "args": {"query": "Singapore flights June 2026"},
                    "id": "call_search",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    assert _workflow_stage(messages) == "travel_structure"

    messages.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "build_travel_itinerary",
                    "args": {"destination": "Singapore", "query_type": "flights", "options": []},
                    "id": "call_itinerary",
                    "type": "tool_call",
                }
            ],
        )
    )

    assert _workflow_stage(messages) == "travel_finalize"
