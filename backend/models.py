# I'm importing the tools I need from SQLAlchemy to define my table structure.
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# I need to get the 'Base' class from my database.py file so my models can inherit from it.
import database

class Location(database.Base):
    # This sets the table name in PostgreSQL.
    __tablename__ = "locations"

    # Here, I define each column for the 'locations' table.
    location_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)

    # I'm setting up the link to the other tables. This says one Location
    # can be linked to many Predictions and many Users.
    predictions = relationship("Prediction", back_populates="location")
    users = relationship("User", back_populates="location")


class User(database.Base):
    # This sets the table name in PostgreSQL.
    __tablename__ = "users"

    # Here, I define each column for the 'users' table.
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    
    # This column links a user to a location using a foreign key.
    subscribed_location_id = Column(Integer, ForeignKey("locations.location_id"))

    # This creates the reverse link back to the Location model.
    location = relationship("Location", back_populates="users")


class Prediction(database.Base):
    # This sets the table name in PostgreSQL.
    __tablename__ = "predictions"

    # Here, I define each column for the 'predictions' table.
    prediction_id = Column(Integer, primary_key=True, index=True)
    risk_level = Column(String)
    
    # I'm setting the default timestamp to be the current time when a prediction is made.
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # This column links a prediction to a location using a foreign key.
    location_id = Column(Integer, ForeignKey("locations.location_id"))
    
    # This creates the reverse link back to the Location model.
    location = relationship("Location", back_populates="predictions")