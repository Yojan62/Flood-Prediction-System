import os
import sys
import datetime as dt

from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 1. Add paths to sys.path so imports work
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 2. Explicitly load .env from the backend folder
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)
# ----------------

# Now standard imports should work
import database
import models

# Import the new hybrid model prediction helper
from ml.predict import predict_recent_days

# Load environment variables
load_dotenv()

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")


def send_flood_alert(
    user_email: str,
    location_name: str,
    risk_level: str,
    predicted_discharge: float,
    date_str: str,
) -> None:
    """
    Sends a single flood alert email using SendGrid.
    """
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        print("SendGrid credentials not configured. Skipping email.")
        return

    message = Mail(
        from_email=SENDGRID_FROM_EMAIL,
        to_emails=user_email,
        subject=f"FLOOD ALERT: {risk_level} risk detected for {location_name}",
        html_content=f"""
            <strong>Flood alert for {location_name}</strong>
            <p>Date: <strong>{date_str}</strong></p>
            <p>Combined flood risk level: <strong>{risk_level}</strong></p>
            <p>Predicted discharge index: <strong>{predicted_discharge:.2f}</strong></p>
            <p>Please take appropriate precautions.</p>
        """,
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        print(f"Successfully sent alert to {user_email}")
    except Exception as e:
        print(f"Error sending email to {user_email}: {e}")


def run_prediction_cycle(days: int = 3) -> None:
    """
    Main function to:
      - Run the hybrid ML + Global Flood predictions
      - Save them into the database
      - Trigger email alerts for HIGH risk days

    Parameters
    ----------
    days : int
        Number of recent days (including today) to predict and store.
    """
    print(f"--- Starting prediction cycle for last {days} day(s) ---")

    db: Session = database.SessionLocal()

    try:
        locations = db.query(models.Location).all()
        if not locations:
            print("No locations found in database. Add locations first.")
            return

        for location in locations:
            print(f"\nProcessing location: {location.name} (ID: {location.location_id})")

            # 1) Get recent predictions from the ML pipeline
            try:
                df = predict_recent_days(
                    lat=location.latitude,
                    lon=location.longitude,
                    days=days,
                )
            except Exception as e:
                print(f"Prediction failed for {location.name}: {e}")
                continue

            if df.empty:
                print(f"No prediction data returned for {location.name}.")
                continue

            # 2) Loop over each predicted day
            for ts, row in df.iterrows():
                # ts is a pandas.Timestamp
                date_only = ts.date()
                date_str = date_only.isoformat()

                predicted_discharge = float(row["pred_final"])

                # Use the combined risk level from the ML + GloFAS fusion.
                # Fallback to 'low' if the column is missing.
                risk_level = str(row.get("risk_combined_level", "low")).upper()

                # OPTIONAL: override with location-specific danger_threshold if set
                if location.danger_threshold is not None:
                    if predicted_discharge >= location.danger_threshold:
                        risk_level = "HIGH"
                    elif predicted_discharge >= 0.8 * location.danger_threshold:
                        risk_level = max(risk_level, "MEDIUM")  # keep worst case

                # Avoid inserting duplicate predictions for the same day/location
                existing = (
                    db.query(models.Prediction)
                    .filter(
                        models.Prediction.location_id == location.location_id,
                        models.Prediction.prediction_timestamp == date_only,
                    )
                    .first()
                )
                if existing:
                    print(f"Prediction already exists for {location.name} on {date_str}. Skipping insert.")
                    continue

                # 3) Save prediction into DB
                db_prediction = models.Prediction(
                    location_id=location.location_id,
                    predicted_discharge=predicted_discharge,
                    risk_level=risk_level,
                    prediction_timestamp=dt.datetime.combine(
                        date_only, dt.time.min
                    ),  # store as midnight UTC for that day
                )
                db.add(db_prediction)
                db.commit()
                db.refresh(db_prediction)

                print(
                    f"Saved prediction {db_prediction.prediction_id} "
                    f"for {location.name} on {date_str}: "
                    f"{predicted_discharge:.2f} ({risk_level})"
                )

                # 4) Trigger alerts if risk is HIGH
                if risk_level == "HIGH":
                    users_to_alert = (
                        db.query(models.User)
                        .filter(models.User.subscribed_location_id == location.location_id)
                        .all()
                    )

                    if not users_to_alert:
                        print("No users subscribed to this location. No alerts sent.")
                    else:
                        print(f"Sending alerts to {len(users_to_alert)} subscribed user(s).")
                        for user in users_to_alert:
                            send_flood_alert(
                                user_email=user.email,
                                location_name=location.name,
                                risk_level=risk_level,
                                predicted_discharge=predicted_discharge,
                                date_str=date_str,
                            )
                else:
                    print(f"Risk is {risk_level} for {location.name} on {date_str}. No alerts sent.")

    except Exception as e:
        print(f"An error occurred during the prediction cycle: {e}")
        db.rollback()
    finally:
        db.close()
        print("\n--- Prediction cycle finished ---")


if __name__ == "__main__":
    run_prediction_cycle(days=3)