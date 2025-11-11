# Imports all necessary tools
import os
import datetime
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Import CORSMiddeware to handle Cross-Origin Resource Sharing (CORS) issues.
from fastapi.middleware.cors import CORSMiddleware

# Imports the local files that define the database connection and the table models.
import database
import models

# This line tells SQLAlchemy to create any missing tables.
# It should be commented out.
#print("--- WARNING: DROPPING AND RECREATING ALL TABLES ---")
#models.database.Base.metadata.drop_all(bind=database.engine)
#models.database.Base.metadata.create_all(bind=database.engine)
#print("--- TABLES RECREATED SUCCESSFULLY ---")


# --- Pydantic Schemas ---

# Defines the shape of data for creating a new location
class LocationCreate(BaseModel):
    name: str
    latitude: float
    longitude: float

# Defines the shape of data for a subscription request
class SubscriptionCreate(BaseModel):
    email: str
    location_id: int

# This schema defines the data to return when reading a location
class Location(BaseModel):
    location_id: int
    name: str
    latitude: float
    longitude: float

    class Config:
        from_attributes = True

# Defines the data to return when reading a prediction
class Prediction(BaseModel):
    prediction_id: int
    
    # --- THIS IS THE CHANGE ---
    # Renamed from predicted_level to match the new model/database
    predicted_discharge: float | None = None 
    
    risk_level: str
    prediction_timestamp: datetime.datetime 
    location_id: int

    class Config:
        from_attributes = True


# --- FastAPI App Instance ---
app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://localhost:3000", # The address of my React frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Dependency ---
def get_db():
    # This function creates and yields a new database session per request.
    db = database.SessionLocal()
    try:
        yield db
    finally:
        # This ensures the database session is always closed.
        db.close()


# --- API Endpoints ---

@app.get("/")
async def read_root():
    # The main 'welcome' endpoint to quickly check if the server is running.
    return {"message": "Flood Prediction API is running!"}


@app.get("/api/test-db")
async def test_database_connection(db: Session = Depends(get_db)):
    # An endpoint to test the database connection.
    try:
        # Wraps the raw SQL 'SELECT 1' query in the text() function for safety.
        db.execute(text('SELECT 1'))
        return {"status": "success", "message": "Database connection is working!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


@app.post("/api/test-email")
async def send_test_email(to_email: str):
    # An endpoint to test the SendGrid email notification.
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    from_email_address = os.getenv("SENDGRID_FROM_EMAIL")

    if not sendgrid_api_key:
        raise HTTPException(status_code=500, detail="SendGrid API key not configured in .env file")
    if not from_email_address:
         raise HTTPException(status_code=500, detail="SENDGRID_FROM_EMAIL not configured in .env file.")

    message = Mail(
        from_email=from_email_address,
        to_emails=to_email,
        subject='Test Email from Flood Prediction System',
        html_content='<strong>This is a test email to confirm SendGrid is working!</strong>'
    )
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        return {"status": "success", "sendgrid_status_code": response.status_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


@app.get("/api/locations", response_model=list[Location])
async def get_locations(db: Session = Depends(get_db)):
    # This queries the database for all records in the 'locations' table.
    locations = db.query(models.Location).all()
    return locations


@app.post("/api/locations")
async def create_location(location: LocationCreate, db: Session = Depends(get_db)):
    # An endpoint to add a new location to the 'locations' table.
    db_location = models.Location(
        name=location.name,
        latitude=location.latitude,
        longitude=location.longitude
    )
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location


@app.post("/api/subscribe")
async def subscribe_user(subscription: SubscriptionCreate, db: Session = Depends(get_db)):
    # Handles user subscription requests.
    location = db.query(models.Location).filter(models.Location.location_id == subscription.location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail=f"Location with id {subscription.location_id} not found.")

    existing_user = db.query(models.User).filter(
        models.User.email == subscription.email,
        models.User.subscribed_location_id == subscription.location_id
    ).first()

    if existing_user:
        return {"status": "success", "message": f"{subscription.email} is already subscribed to {location.name}."}

    db_user = models.User(
        email=subscription.email,
        subscribed_location_id=subscription.location_id
    )

    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return {"status": "success", "message": f"User subscribed to {location.name} successfully.", "user_id": db_user.user_id}
    except Exception as e:
        db.rollback()
        print(f"Database error during subscription: {e}")
        raise HTTPException(status_code=500, detail="Could not save subscription due to a database error.")
    

@app.get("/api/predictions/{location_id}", response_model=list[Prediction])
async def get_predictions_for_location(location_id: int, db: Session = Depends(get_db)):
    """
    Fetches the prediction history for a specific location.
    """
    # Queries the database for predictions matching the location_id,
    # and orders them by the newest first.
    predictions = db.query(models.Prediction).filter(
        models.Prediction.location_id == location_id
    ).order_by(models.Prediction.prediction_timestamp.desc()).all()
    
    if not predictions:
        return []
    
    return predictions