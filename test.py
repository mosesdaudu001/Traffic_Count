import json

# Assume that the total number of vehicles is stored in a variable called `total_vehicles`
total_vehicles = 10

# Create a Python dictionary with the total number of vehicles
vehicle_data = {"total_vehicles": total_vehicles}

# Convert the dictionary to a JSON string
json_data = json.dumps(vehicle_data)

# Write the JSON string to a file
with open("vehicle_data.json", "w") as f:
    f.write(json_data)