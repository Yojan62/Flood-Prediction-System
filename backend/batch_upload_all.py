import requests
import json

# This list contains all 145 unique locations from your documents
# (135 from Bangladesh and 10 from Bangkok ).
LOCATIONS_DATA = [
    # Bangladesh Locations
    {"name": "Panchagarh", "latitude": 26.330021, "longitude": 88.544405},
    {"name": "Thakurgaon", "latitude": 25.03796, "longitude": 88.460897},
    {"name": "Dalia", "latitude": 25.125275, "longitude": 89.073729},
    {"name": "Bhusirbandar", "latitude": 25.767523, "longitude": 88.731861},
    {"name": "Dinajpur", "latitude": 25.589237, "longitude": 88.518762},
    {"name": "Badarganj", "latitude": 25.575475, "longitude": 89.062548},
    {"name": "Phulbari", "latitude": 25.5003556, "longitude": 88.9515139},
    {"name": "Pateswari", "latitude": 25.995899, "longitude": 89.730003},
    {"name": "Noonkhawa", "latitude": 25.920668, "longitude": 89.76646},
    {"name": "Kurigram", "latitude": 25.82041, "longitude": 89.667595},
    {"name": "Kaunia", "latitude": 25.787864, "longitude": 89.437306},
    {"name": "Hatia", "latitude": 25.58028, "longitude": 89.69196},
    {"name": "Tarapur", "latitude": 25.642045, "longitude": 89.534597},
    {"name": "Boalmari", "latitude": 25.617417, "longitude": 89.867033},
    {"name": "Chilmari", "latitude": 25.544273, "longitude": 89.685153},
    {"name": "Haripur", "latitude": 25.519167, "longitude": 89.650556},
    {"name": "Char-Rajibpur", "latitude": 25.429977, "longitude": 89.767779},
    {"name": "Kamarjani", "latitude": 25.405211, "longitude": 89.625204},
    {"name": "Goalkanda", "latitude": 25.3664, "longitude": 89.779999},
    {"name": "Gaibandha", "latitude": 25.352536, "longitude": 89.534632},
    {"name": "Nakuagaon", "latitude": 25.189811, "longitude": 90.215824},
    {"name": "Fulchari", "latitude": 25.186701, "longitude": 89.610001},
    {"name": "Chakrahimpur", "latitude": 25.146083, "longitude": 89.377776},
    {"name": "Bahadurabad", "latitude": 25.104949, "longitude": 89.70111},
    {"name": "Saghata", "latitude": 25.10754, "longitude": 89.58656},
    {"name": "Shimulbari", "latitude": 25.0243, "longitude": 89.529999},
    {"name": "Bijoypur", "latitude": 25.158699, "longitude": 90.669998},
    {"name": "Durgapur", "latitude": 25.123256, "longitude": 90.661579},
    {"name": "Kalmakanda", "latitude": 25.073472, "longitude": 90.889765},
    {"name": "Jariajanjail", "latitude": 25.009779, "longitude": 90.652813},
    {"name": "Lorergorh", "latitude": 25.191499, "longitude": 91.250119},
    {"name": "Muslimpur", "latitude": 25.1387, "longitude": 91.389999},
    {"name": "2 sunamganj", "latitude": 25.071634, "longitude": 91.405531},
    {"name": "Chhatak", "latitude": 25.036, "longitude": 91.66},
    {"name": "2 Sylhet", "latitude": 24.888834, "longitude": 91.849452},
    {"name": "2 Sheola", "latitude": 24.890721, "longitude": 92.191013},
    {"name": "Amalshid", "latitude": 24.872495, "longitude": 92.485642},
    {"name": "Kanaighat", "latitude": 25.004479, "longitude": 92.265191},
    {"name": "Lubachara", "latitude": 25.0361995697021, "longitude": 92.3000030517578},
    {"name": "2 Sarighat", "latitude": 25.087706, "longitude": 92.11833},
    {"name": "sariakandi", "latitude": 24.891637, "longitude": 89.580654},
    {"name": "Jamalpur", "latitude": 24.923502, "longitude": 89.967658},
    {"name": "Bogura", "latitude": 24.831299, "longitude": 89.391511},
    {"name": "singra", "latitude": 24.497782, "longitude": 89.138756},
    {"name": "Rajshahi", "latitude": 24.368824, "longitude": 88.553699},
    {"name": "Atrai", "latitude": 24.611, "longitude": 88.980003},
    {"name": "Naogaon", "latitude": 24.844091, "longitude": 88.932033},
    {"name": "Mohadebpur", "latitude": 24.915415, "longitude": 88.744249},
    {"name": "Rohanpur", "latitude": 24.820982, "longitude": 88.318746},
    {"name": "Pankha", "latitude": 24.644253, "longitude": 88.068931},
    {"name": "chapai-Nawabganj", "latitude": 24.599517, "longitude": 88.249526},
    {"name": "Serajganj", "latitude": 24.470952, "longitude": 89.718387},
    {"name": "Kazipur", "latitude": 24.635869, "longitude": 89.689858},
    {"name": "Jagannathganj", "latitude": 24.6544, "longitude": 89.809998},
    {"name": "Mymensingh", "latitude": 24.737208, "longitude": 90.431068},
    {"name": "Khaliajuri", "latitude": 24.70217, "longitude": 91.110489},
    {"name": "Derai", "latitude": 24.785555, "longitude": 91.379531},
    {"name": "Markuli", "latitude": 24.691644, "longitude": 91.389698},
    {"name": "Fenchuganj", "latitude": 24.700799, "longitude": 91.949997},
    {"name": "Sherpur-Sylhet", "latitude": 24.628307, "longitude": 91.681667},
    {"name": "Moulvibazar", "latitude": 24.496424, "longitude": 91.774876},
    {"name": "Manu-RB", "latitude": 24.428089, "longitude": 91.937553},
    {"name": "Kamalganj", "latitude": 24.352029, "longitude": 91.84582},
    {"name": "Habiganj", "latitude": 24.392752, "longitude": 91.410342},
    {"name": "Ballah", "latitude": 24.098525, "longitude": 91.59615},
    {"name": "B. Baria", "latitude": 23.956966, "longitude": 91.119442},
    {"name": "Bhairab Bazar", "latitude": 24.045541, "longitude": 90.991205},
    {"name": "Narsingdi", "latitude": 23.918646, "longitude": 90.72553},
    {"name": "Lakhpur", "latitude": 24.040344, "longitude": 90.550532},
    {"name": "Cumilla", "latitude": 23.471078, "longitude": 91.19839},
    {"name": "Debidwar", "latitude": 23.63091, "longitude": 90.987282},
    {"name": "Meghna Bridge", "latitude": 23.58457, "longitude": 90.638969},
    {"name": "Rekabi-Bazar", "latitude": 23.56269, "longitude": 90.48001},
    {"name": "Bayderbazar", "latitude": 23.649571, "longitude": 90.625332},
    {"name": "Narayanganj", "latitude": 23.630486, "longitude": 90.514175},
    {"name": "Hariharpara", "latitude": 23.63343, "longitude": 90.469183},
    {"name": "Dhaka", "latitude": 23.696849, "longitude": 90.419454},
    {"name": "Demra", "latitude": 23.732784, "longitude": 90.496305},
    {"name": "Mirpur", "latitude": 23.784904, "longitude": 90.336562},
    {"name": "Tongi", "latitude": 23.881792, "longitude": 90.4013},
    {"name": "Nayarhat", "latitude": 23.911094, "longitude": 90.230264},
    {"name": "Kaliakoir", "latitude": 24.082154, "longitude": 90.207596},
    {"name": "Elasin", "latitude": 24.167426, "longitude": 89.835279},
    {"name": "Porabari", "latitude": 24.147259, "longitude": 89.815249},
    {"name": "Baghabari", "latitude": 24.129955, "longitude": 89.582497},
    {"name": "Mathura", "latitude": 23.9496, "longitude": 89.650002},
    {"name": "Jagir", "latitude": 23.878853, "longitude": 90.025397},
    {"name": "Taraghat", "latitude": 23.862485, "longitude": 89.958068},
    {"name": "Aricha", "latitude": 23.881297, "longitude": 89.772772},
    {"name": "Goalondo", "latitude": 23.768642, "longitude": 89.778872},
    {"name": "Hardinge-RB", "latitude": 24.06764, "longitude": 89.03352},
    {"name": "Talbaria", "latitude": 23.961, "longitude": 89.099998},
    {"name": "Goorai-RB", "latitude": 23.883877, "longitude": 89.182899},
    {"name": "Hatboalia", "latitude": 23.787087, "longitude": 88.857085},
    {"name": "Mawa", "latitude": 23.4704, "longitude": 90.260002},
    {"name": "Bhagyakul", "latitude": 23.511165, "longitude": 90.206392},
    {"name": "Sureshswar", "latitude": 23.320123, "longitude": 90.439032},
    {"name": "Faridpur", "latitude": 23.597854, "longitude": 89.831591},
    {"name": "Kamarkhali", "latitude": 23.539828, "longitude": 89.516256},
    {"name": "chuadanga", "latitude": 23.645365, "longitude": 88.842925},
    {"name": "Parshuram", "latitude": 23.221259, "longitude": 91.438231},
    {"name": "Suber Bazar", "latitude": 23.237, "longitude": 91.4144},
    {"name": "Malipur_C", "latitude": 23.0377, "longitude": 91.4389},
    {"name": "Haripur_C", "latitude": 23.0356, "longitude": 91.4856},
    {"name": "Ramgarh", "latitude": 22.968128, "longitude": 91.703954},
    {"name": "Shuvopur", "latitude": 22.9538, "longitude": 91.5442},
    {"name": "Narayanhat", "latitude": 22.807724, "longitude": 91.719765},
    {"name": "Sonapur", "latitude": 22.8369, "longitude": 91.453},
    {"name": "companygani", "latitude": 22.7694, "longitude": 91.3492},
    {"name": "Noakhali", "latitude": 22.845, "longitude": 91.1017},
    {"name": "Lakshmipur", "latitude": 22.9396, "longitude": 90.8411},
    {"name": "Chandpur", "latitude": 23.23159, "longitude": 90.639487},
    {"name": "Madaripur", "latitude": 23.187806, "longitude": 90.209848},
    {"name": "Haridaspur", "latitude": 23.052, "longitude": 89.82},
    {"name": "Daulatkhan", "latitude": 22.511, "longitude": 90.82},
    {"name": "Barishal", "latitude": 22.700199, "longitude": 90.379997},
    {"name": "Mongla", "latitude": 22.464199, "longitude": 89.599998},
    {"name": "Khulna", "latitude": 22.808223, "longitude": 89.580462},
    {"name": "Jhikargacha", "latitude": 23.101868, "longitude": 89.096763},
    {"name": "Kalaroa", "latitude": 22.871623, "longitude": 89.047524},
    {"name": "Sakra", "latitude": 22.617779, "longitude": 88.969965},
    {"name": "Panchpukuria", "latitude": 22.56086, "longitude": 91.846427},
    {"name": "Chattogram", "latitude": 22.322901, "longitude": 91.830002},
    {"name": "Bandarban", "latitude": 22.194878, "longitude": 92.216246},
    {"name": "Dohazari", "latitude": 22.15791, "longitude": 92.063472},
    {"name": "Lama", "latitude": 21.793554, "longitude": 92.209356},
    {"name": "Chiringa", "latitude": 21.773552, "longitude": 92.079743}
]

# This is the URL of your local backend API endpoint 
API_ENDPOINT = "http://127.0.0.1:8000/api/locations"

def upload_locations():
    """
    I'll loop through the LOCATIONS_DATA list and POST each one
    to my FastAPI backend.
    """
    print(f"Starting to upload {len(LOCATIONS_DATA)} locations to {API_ENDPOINT}...")
    
    success_count = 0
    fail_count = 0

    for location in LOCATIONS_DATA:
        try:
            # My data payload must match the LocationCreate schema in main.py 
            payload = {
                "name": location["name"],
                "latitude": location["latitude"],
                "longitude": location["longitude"]
            }
            
            # I'll send the POST request
            response = requests.post(API_ENDPOINT, json=payload)
            
            # I'll check if the request was successful (200 OK)
            if response.status_code == 200:
                print(f"SUCCESS: Added {location['name']}")
                success_count += 1
            else:
                # I'll print an error if the API returned something other than 200
                print(f"FAILED: {location['name']} (Status Code: {response.status_code}, Response: {response.text})")
                fail_count += 1
        
        except requests.exceptions.ConnectionError:
            print("\nFATAL ERROR: Could not connect to the backend server.")
            print(f"I need to make sure my FastAPI server is running at {API_ENDPOINT}")
            return # Stop the script if the server isn't running
        except Exception as e:
            print(f"An unknown error occurred for {location['name']}: {e}")
            fail_count += 1
            
    print("\n--- Batch Upload Complete ---")
    print(f"Successfully added: {success_count}")
    print(f"Failed to add: {fail_count}")

# This part ensures my script only runs when I execute it directly
if __name__ == "__main__":
    # I need to load my environment variables if this script uses them
    # (This one doesn't, but it's good practice)
    # from dotenv import load_dotenv
    # load_dotenv()
    
    upload_locations()