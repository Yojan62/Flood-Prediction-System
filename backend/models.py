import database

# Imports the tools needed to define database columns and their types
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
# Imports the tool to create relationships between tables
from sqlalchemy.orm import relationship
# Imports SQL functions (like 'NOW()' for default timestamps)
from sqlalchemy.sql import func


# Defines the Location table model
class Location(database.Base):
    # This sets the table name in PostgreSQL
    __tablename__ = "locations"

    # Defines each column for the 'locations' table
    location_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    
    danger_threshold = Column(Float, nullable=True) # Storing the m3/s value

    # Sets up the link to the other tables.
    predictions = relationship("Prediction", back_populates="location")
    users = relationship("User", back_populates="location")


# Defines the User table model
class User(database.Base):
    # This sets the table name in PostgreSQL
    __tablename__ = "users"

    # Defines each column for the 'users' table
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True) # Stores my user's email
    
    # This column links a user to a location using a foreign key
    subscribed_location_id = Column(Integer, ForeignKey("locations.location_id"))

    # This creates the reverse link back to the Location model
    location = relationship("Location", back_populates="users")


# Defines the Prediction table model
class Prediction(database.Base):
    # This sets the table name in PostgreSQL
    __tablename__ = "predictions"

    # Defines each column for the 'predictions' table
    prediction_id = Column(Integer, primary_key=True, index=True)
    
    # --- THIS LINE IS UPDATED ---
    # Stores the actual predicted river discharge (e.g., 8500.0 m³/s)
    predicted_discharge = Column(Float) 

    # Stores the risk level string (e.g., "HIGH")
    risk_level = Column(String)
    
    # Sets the default timestamp to be the current time when a prediction is made
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # This column links a prediction to a location using a foreign key
    location_id = Column(Integer, ForeignKey("locations.location_id"))
    
    # This creates the reverse link back to the Location model
    location = relationship("Location", back_populates="predictions")