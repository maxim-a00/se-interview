"""Tool schemas and implementations.

Design note:
- Tool definitions live in this module so the travel tool contract and its
  structured output are separated from LangGraph orchestration.
- This keeps the agent workflow code focused on routing and execution flow
  instead of mixing graph logic with schema-heavy tool code.
- The split is intentionally lightweight: one module for tools is enough for
  this assignment without introducing extra abstraction layers.
"""

import json

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TravelOption(BaseModel):
    name: str = Field(description="Name of the travel option, such as a hotel, flight, or attraction.")
    category: str = Field(description="Travel category, such as hotel, flight, or attraction.")
    location: str = Field(description="Where the option is located or relevant.")
    summary: str = Field(
        default="",
        description="Short summary of why this option matters.",
    )
    price: str = Field(
        default="Not provided",
        description="Price or pricing guidance, if known.",
    )
    rating: str = Field(
        default="Not provided",
        description="Rating or review summary, if known.",
    )
    booking_link: str = Field(
        default="",
        description="Booking or reference URL for the option.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Additional useful notes, constraints, or highlights.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="URLs or source labels supporting this option's details.",
    )


class ItineraryDay(BaseModel):
    day: int = Field(description="Day number in the itinerary.")
    theme: str = Field(description="Short title for the day's plan.")
    morning: str = Field(default="", description="Morning activity or plan.")
    afternoon: str = Field(default="", description="Afternoon activity or plan.")
    evening: str = Field(default="", description="Evening activity or plan.")
    booking_links: list[str] = Field(
        default_factory=list,
        description="Booking or reservation links relevant to this day.",
    )


class TravelItineraryInput(BaseModel):
    destination: str = Field(description="Destination city, region, or area.")
    query_type: str = Field(
        description="What the user is looking for, such as attractions, hotels, or flights.",
    )
    trip_length_days: int = Field(
        default=0,
        description="Optional trip length in days if the user asked for an itinerary.",
    )
    trip_style: str = Field(
        default="",
        description="Optional trip style such as luxury, family, honeymoon, budget, or food-focused.",
    )
    start_date: str = Field(
        default="",
        description="Optional trip or check-in start date.",
    )
    end_date: str = Field(
        default="",
        description="Optional trip or check-out end date.",
    )
    travelers: int = Field(
        default=1,
        description="Number of travelers relevant to the request.",
    )
    must_include: list[str] = Field(
        default_factory=list,
        description="Specific requested items the answer must include, such as hotels, beaches, or booking links.",
    )
    options: list[TravelOption] = Field(
        description="The researched travel options to structure.",
    )
    itinerary: list[ItineraryDay] = Field(
        default_factory=list,
        description="Optional day-by-day itinerary when the user asks for a plan.",
    )
    recommendation_basis: str = Field(
        default="",
        description="Optional explanation of how to interpret the options.",
    )


@tool(args_schema=TravelItineraryInput)
def build_travel_itinerary(
    destination: str,
    query_type: str,
    trip_length_days: int = 0,
    trip_style: str = "",
    start_date: str = "",
    end_date: str = "",
    travelers: int = 1,
    must_include: list[str] | None = None,
    options: list[TravelOption] | None = None,
    itinerary: list[ItineraryDay] | None = None,
    recommendation_basis: str = "",
) -> str:
    """Build a structured travel itinerary or recommendation set from researched travel options."""
    must_include = must_include or []
    options = options or []
    itinerary = itinerary or []
    normalized_options = []
    for option in options:
        normalized_options.append(
            {
                "name": option.name,
                "category": option.category,
                "location": option.location,
                "summary": option.summary or "No summary provided.",
                "price": option.price or "Not provided",
                "rating": option.rating or "Not provided",
                "booking_link": option.booking_link or None,
                "notes": option.notes,
                "sources": option.sources,
            }
        )
    normalized_itinerary = []
    for day in itinerary:
        normalized_itinerary.append(
            {
                "day": day.day,
                "theme": day.theme,
                "morning": day.morning or None,
                "afternoon": day.afternoon or None,
                "evening": day.evening or None,
                "booking_links": day.booking_links,
            }
        )

    structured_output = {
        "destination": destination,
        "query_type": query_type,
        "trip_length_days": trip_length_days or None,
        "trip_style": trip_style or None,
        "dates": {
            "start_date": start_date or None,
            "end_date": end_date or None,
        },
        "travelers": travelers,
        "must_include": must_include,
        "options": normalized_options,
        "itinerary": normalized_itinerary,
        "recommendation_basis": recommendation_basis or None,
    }
    return json.dumps(structured_output, indent=2)


def get_tools():
    return [DuckDuckGoSearchRun(), build_travel_itinerary]


__all__ = [
    "ItineraryDay",
    "TravelItineraryInput",
    "TravelOption",
    "build_travel_itinerary",
    "get_tools",
]
