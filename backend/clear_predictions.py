import os
import sys

# This ensures we can find 'database.py' and 'models.py'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# ----------------

from database import SessionLocal
from models import Prediction

def clear_table():
    db = SessionLocal()
    try:
        print("Deleting all prediction records...")
        # Deletes all rows from the 'predictions' table
        num_deleted = db.query(Prediction).delete()
        db.commit()
        print(f"Success! Deleted {num_deleted} predictions.")
        
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clear_table()