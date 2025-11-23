import database
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class Location(database.Base):
    __tablename__ = "locations"

    location_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)

    # Used for risk classification (optional)
    danger_threshold = Column(Float, nullable=True)

    predictions = relationship(
        "Prediction",
        back_populates="location",
        cascade="all, delete-orphan"
    )

    users = relationship(
        "User",
        back_populates="location",
        cascade="all, delete-orphan"
    )


class User(database.Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)

    # Allow same email to subscribe to multiple locations
    email = Column(String, index=True)

    subscribed_location_id = Column(Integer, ForeignKey("locations.location_id"))
    location = relationship("Location", back_populates="users")


class Prediction(database.Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True)

    predicted_discharge = Column(Float)
    risk_level = Column(String)

    # DO NOT rely on NOW(), supply prediction date from ML
    prediction_timestamp = Column(DateTime(timezone=True))

    location_id = Column(Integer, ForeignKey("locations.location_id"))
    location = relationship("Location", back_populates="predictions")
