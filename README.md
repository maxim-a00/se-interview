# LangGraph Travel Assistant with Web Search

A simple LangGraph travel assistant that can search the web using DuckDuckGo, build structured travel search results, and expose the workflow through a FastAPI server.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- OpenAI API key

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Create a `.env` file from the example:

```bash
cp .env.example .env
```

3. Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_actual_api_key
```

## Running the API

Start the FastAPI server:

```bash
poetry run uvicorn app.api:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /chat

Send a message to the agent and receive a response.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about AI?"}'
```

**Response:**

```json
{
  "response": "Based on the search results, here are the latest developments in AI..."
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

## Travel Tool Extension

The agent now includes an additional tool called `build_travel_itinerary`.

### What It Does

This tool helps the agent turn researched travel information into structured itineraries and recommendation sets. It is useful for prompts such as:

- "Find top attractions in Rome for a weekend trip"
- "Recommend hotels in Barcelona for two travelers"
- "Help me compare flight options from Berlin to Lisbon next month"

### Structured Output

The tool returns JSON with a consistent schema:

```json
{
  "destination": "Rome",
  "query_type": "attractions",
  "trip_length_days": 3,
  "trip_style": "family",
  "dates": {
    "start_date": null,
    "end_date": null
  },
  "travelers": 2,
  "must_include": ["booking links", "itinerary"],
  "options": [
    {
      "name": "Colosseum",
      "category": "attraction",
      "location": "Rome",
      "summary": "Historic amphitheater and one of the city's signature landmarks.",
      "price": "From 18 EUR",
      "rating": "4.7/5",
      "booking_link": "https://example.com/colosseum",
      "notes": ["Book timed entry ahead of time"],
      "sources": ["https://example.com/colosseum"]
    }
  ],
  "itinerary": [
    {
      "day": 1,
      "theme": "Ancient Rome highlights",
      "morning": "Timed-entry visit to the Colosseum",
      "afternoon": "Forum and Palatine Hill",
      "evening": "Dinner in Monti",
      "booking_links": ["https://example.com/colosseum-booking"]
    }
  ],
  "recommendation_basis": "Best for first-time visitors focused on iconic landmarks."
}
```

### Workflow Integration

The LangGraph workflow was intentionally constrained so the agent uses tools more deliberately:

1. The user sends a message to `/chat`
2. The routing policy checks whether the request is travel-related and whether it needs current information
3. If current information is needed, the agent calls web search first
4. For travel-planning requests, the agent then calls `build_travel_itinerary` once to create structured output
5. The structured travel output is added back into the conversation
6. The agent writes the final answer without making additional tool calls

## Design Decisions
- Web search remains responsible for fact gathering.
- The new tool `build_travel_itinerary` is responsible for normalization and structure, not discovery.
- The tool returns structured JSON so the agent can reason over a predictable format instead of raw prose.
- Travel-specific fields such as `category`, `location`, `price`, `rating`, and `booking_link` make the output useful for attractions, hotels, and flights.
- The tool also supports `trip_length_days`, `must_include`, and day-by-day `itinerary` blocks so the model has a natural path to structured travel plans.
- The tool name was changed to `build_travel_itinerary` so the intent is explicit and easier for the model to select.
- Source links are preserved per option so the final answer can stay grounded in the gathered evidence.
- The workflow in `app/agent.py` was constrained after Phoenix evaluation tool-call loops within a single interaction, especially redundant search calls and duplicate itinerary generation. Travel requests now follow a more deliberate sequence: search only when current information is needed, call `build_travel_itinerary` once, and then finalize without further tool use.

### Maintainability Choices

A small set of structural refactors was made to keep the project maintainable without over-engineering it.

- Tool definitions were moved into `app/tools.py` so schema-heavy tool logic is separate from LangGraph orchestration.
- Environment-backed settings were centralized in `app/config.py` so model settings, API metadata, and Phoenix configuration are managed in one place and are easier to test.
- Refactoring stopped after those changes because they addressed real problems in the prototype. For the scope of this assignment, adding more layers would have increased complexity without much practical benefit.


## Project Structure

```
se-interview/
├── app/
│   ├── agent.py     # LangGraph graph and workflow routing
│   ├── api.py       # FastAPI app factory and routes
│   ├── config.py    # Centralized environment-backed app configuration
│   ├── tools.py     # Tool schemas and implementations
│   └── __init__.py
├── scripts/         # Phoenix and evaluation helper utilities
├── tests/           # Unit tests
├── Dockerfile       # Container image definition
├── pyproject.toml   # Poetry dependencies and test config
├── .env.example     # Environment variable template
├── README.md        # Project documentation
├── agent.py         # Compatibility shim
└── api.py           # Compatibility shim
```

## Testing

Run the unit test suite with:

```bash
poetry run pytest
```

The repository includes unit tests for:

- the FastAPI health and chat routes
- environment-backed configuration
- the structured travel tool output

## Docker

Build and run the API in Docker:

```bash
docker build -t se-interview .
docker run --env-file .env -p 8000:8000 se-interview
```

### Docker Environment Notes

The Docker container needs an OpenAI API key to answer `/chat` requests:

- `OPENAI_API_KEY` is required

If you also want Phoenix traces from inside the container, set:

- `PHOENIX_COLLECTOR_ENDPOINT`
- `PHOENIX_PROJECT_NAME`

When Phoenix is running on the host machine, `localhost` inside the container will refer to the container itself rather than the host. In that case, use:

```bash
PHOENIX_COLLECTOR_ENDPOINT=http://host.docker.internal:6006/v1/traces
```

Example:

```bash
docker run \
  --env-file .env \
  -e PHOENIX_COLLECTOR_ENDPOINT=http://host.docker.internal:6006/v1/traces \
  -p 8000:8000 \
  se-interview
```

## Phoenix Observability

This project is instrumented with Arize Phoenix using Phoenix OTEL plus OpenInference instrumentations for LangChain and OpenAI.

### What Phoenix Captures

- LLM calls made through `ChatOpenAI`
- tool usage inside the LangGraph workflow, including web search and `build_travel_itinerary`

### Run Phoenix Locally

Install and start Phoenix in the same Python environment:

```bash
poetry install
poetry run phoenix serve
```

By default, the Phoenix UI will be available at `http://localhost:6006`.

### Phoenix Configuration

The app initializes tracing in [app/observability.py](/Users/maximbelikov/Documents/Arize/se-interview/app/observability.py) using:

- `PHOENIX_COLLECTOR_ENDPOINT`, defaulting to `http://localhost:6006/v1/traces`
- `PHOENIX_PROJECT_NAME`, defaulting to `se-interview`

Tracing is enabled before the agent is built, so LangGraph execution, LLM spans, and tool spans are registered from the start of each request.

## Evaluation Methodology

### Chosen Metric: Agent Tool Selection

`Agent tool selection evals` was chosen as the additional evaluation metric because the project requirement includes extending the agent with a tool, and Phoenix provides a built-in evaluator specifically for measuring whether an agent selected the appropriate tool for a request.

### Why This Metric

- It directly measures agent behavior, not just answer quality.
- It is tightly aligned with the assignment requirement to add a tool.
- Phoenix already captures both root interaction spans and child tool spans, so the metric can be grounded in real execution traces.
- It uses Phoenix's specialized `ToolSelectionEvaluator` rather than a generic custom rubric.

### Iteration and Refinement

This metric choice was informed by earlier Phoenix evaluation runs rather than chosen in the abstract.

- In the first traced runs, the agent often over-relied on `duckduckgo_search` and did not consistently select the custom travel tool for structured travel-planning prompts.
- After reviewing those traces, the travel tool was redesigned to make its intent clearer and to better match the kinds of outputs users were requesting, such as itineraries and booking-oriented recommendations.
- Phoenix evaluation was then rerun, and the custom tool was invoked more often, but tool selection was still judged incorrect in many cases because the agent was overcalling tools or combining search and itinerary generation inefficiently.
- That feedback led to the next workflow change: the LangGraph policy was constrained so travel requests follow a more deliberate sequence, using search only when current information is needed, then calling `build_travel_itinerary` once, and finally producing the answer without additional tool calls.

This iteration cycle made `Agent Tool Selection Evals` the most useful metric for the project, because it directly reflected whether the workflow changes were improving the agent's actual decision-making.

### Method

The evaluation runs on the root `LangGraph` interaction spans in Phoenix. For each interaction, the evaluator collects:

- the input conversation
- the available tool definitions
- the tools actually selected in the trace

Phoenix's built-in `ToolSelectionEvaluator` then labels each interaction as:

- `correct`: the agent selected the right tool or tool combination for the request
- `incorrect`: the agent selected the wrong tool, missed an obvious needed tool, or avoided tool use when tool use was clearly needed

For this travel assistant, the evaluation checks whether requests that need structured or current travel guidance are matched with the appropriate tool strategy, such as using search for current information and the custom travel tool when the user is asking for structured travel planning.

### Phoenix Attachment

The metric is attached back to the root interaction spans in Phoenix as the annotation:

- `tool_selection_correctness`

This makes it possible to filter traces directly in Phoenix and inspect where tool selection behavior is correct versus weak.

### Demo Script

You can send a prompt through the local API and immediately inspect recent Phoenix spans with:

```bash
poetry run python scripts/run_traced_prompt.py "Find hotel options in Barcelona for two travelers."
```

Optional flags:

- `--api-url` to target a different chat endpoint
- `--phoenix-url` to target a different Phoenix instance
- `--project` to inspect a different Phoenix project

## How It Works

1. The agent receives a user message via the `/chat` endpoint
2. Routing logic determines whether the request is a travel-planning request and whether live search is actually needed
3. If live information is needed, the agent calls DuckDuckGo search once and feeds the result back into the graph
4. For structured travel requests, the agent calls `build_travel_itinerary` once to normalize the recommendation set into JSON
5. After the structured tool output is available, the agent finalizes the response without further tool calls
6. The response is returned to the user

## Appendix: Helper Scripts

The `scripts/` directory contains local utilities used for tracing, evaluation, and dataset creation. These scripts are not required to run the FastAPI app itself, but they were used to support the Phoenix analysis and evaluation workflow for the assignment.

- `run_traced_prompt.py`: sends a prompt to the local API and then inspects recent Phoenix spans
- `run_prompt_batch.py`: runs a batch of prompts and saves prompt/response results to a file
- `export_phoenix_spans.py`: exports Phoenix spans to CSV
- `evaluate_user_frustration.py`: runs the user-frustration evaluation in Phoenix
- `create_frustrated_dataset.py`: creates a Phoenix dataset of frustrated interactions from reviewed traces
- `evaluate_tool_selection_correctness.py`: runs Phoenix's `ToolSelectionEvaluator` on traced interactions

These scripts are kept separate from the application package so the production API code stays focused on serving requests, while evaluation and observability workflows remain optional supporting utilities.
