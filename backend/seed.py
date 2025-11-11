import database
import models
from sqlalchemy.orm import Session

# Get a new database session
db: Session = database.SessionLocal()

try:
    print("Seeding database with initial location (Dhaka)...")

    # 1. Create the Dhaka location with the new threshold
    dhaka = models.Location(
        name="Dhaka",
        latitude=23.81,
        longitude=90.41,
        danger_threshold=80.0  # This is the new dynamic value
    )

    # 2. Create the Chittagong location
    chittagong = models.Location(
        name="Chittagong",
        latitude=22.35,
        longitude=91.83,
        danger_threshold=65.0  # The new threshold you found from the plot
    )

    db.add(dhaka)
    db.add(chittagong)
    db.commit()

    print(f"Successfully added {dhaka.name} with ID {dhaka.location_id}.")

except Exception as e:
    print(f"An error occurred: {e}")
    db.rollback()
finally:
    db.close()

print("Database seeding complete.")