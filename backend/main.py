from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI app instance
app = FastAPI()

# This part is important! It's a security setting that allows
# the frontend website (running on a different address like localhost:3000)
# to make requests to your backend.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# This is your dummy endpoint. When a browser requests this URL,
# this function runs and returns the hard-coded JSON data.
@app.get("/api/risk/glasgow")
async def get_risk_for_glasgow():
    """
    Returns a hard-coded JSON response for a flood risk in Glasgow.
    """
    return {
        "location": "Glasgow",
        "risk_level": "LOW",
        "timestamp": "2025-10-10T23:44:02Z"
    }