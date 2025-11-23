import json
from sqlalchemy.orm import Session
from sqlalchemy import or_
import models
import database


def update_and_clean_locations():
    """
    Reads locations.json and syncs danger thresholds.

    Phase 1:
        - Update matching locations with valid danger thresholds.
        - Set threshold to NULL if invalid or missing.
    
    Phase 2:
        - Delete all locations that still have NULL or 0.0 thresholds.
    """

    db: Session = database.SessionLocal()

    try:
        print("--- Phase 1: Updating thresholds from JSON ---")
        with open('locations.json', 'r', encoding='utf-8') as f:
            locations_data = json.load(f)

        print(f"Loaded {len(locations_data)} stations from JSON.")

        updated_count = 0
        nullified_count = 0
        missing_in_db_count = 0

        # ------ PHASE 1 ------
        for entry in locations_data:
            name = entry.get("name")
            raw_level = entry.get("dangerlevel")

            if not name:
                continue

            loc = db.query(models.Location).filter(
                models.Location.name == name
            ).first()

            if not loc:
                missing_in_db_count += 1
                continue

            try:
                level = float(raw_level)
                if level > 0:
                    loc.danger_threshold = level
                    updated_count += 1
                else:
                    # 0.00 or negative → treat as missing
                    loc.danger_threshold = None
                    nullified_count += 1
            except (TypeError, ValueError):
                # "-", "", or invalid → treat as missing
                loc.danger_threshold = None
                nullified_count += 1

        db.commit()
        print(
            f"Phase 1 Complete: "
            f"{updated_count} updated, "
            f"{nullified_count} nullified, "
            f"{missing_in_db_count} missing in DB."
        )

        # ------ PHASE 2 ------
        print("\n--- Phase 2: Removing invalid stations (NULL or 0.0 thresholds) ---")

        to_delete = db.query(models.Location).filter(
            or_(
                models.Location.danger_threshold == None,
                models.Location.danger_threshold == 0.0,
            )
        ).all()

        delete_count = len(to_delete)

        for loc in to_delete:
            print(f"Deleting '{loc.name}' (no valid threshold).")
            db.delete(loc)

        db.commit()

        print(
            f"\n--- Clean-up Complete ---\n"
            f"🟢 Updated: {updated_count}\n"
            f"🟡 Nullified (set to None): {nullified_count}\n"
            f"🗑️ Deleted in Phase 2: {delete_count}\n"
            f"⏭️ Missing in DB (ignored): {missing_in_db_count}\n"
        )

    except Exception as e:
        print(f"Error during update: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    update_and_clean_locations()
