# Install Pywr if you haven’t already: pip install pywr

from pywr.core import Model, Node
from pywr.nodes import Input, Output, Storage
import pandas as pd

# Step 1: Create a simple dataset
data = {
    "day": list(range(1, 31)),  # Days 1 to 30
    "river_inflow": [10, 15, 12, 8, 14, 11, 10, 12, 13, 9, 15, 12, 11, 13, 14, 10, 11, 15, 12, 8, 14, 10, 11, 13, 9, 10, 14, 12, 11, 13],
    "city_demand": [8, 9, 8, 7, 8, 9, 10, 9, 10, 8, 9, 8, 9, 10, 8, 9, 8, 9, 10, 8, 9, 8, 9, 10, 8, 9, 8, 9, 10, 8]
}

df = pd.DataFrame(data)

# Step 2: Create the Pywr model
model = Model()

# Define nodes
river = Input(model, "River Inflow")
reservoir = Storage(model, "Reservoir", max_volume=50, initial_volume=30)  # Max capacity 50 units
city = Output(model, "City Demand")

# Connect nodes
river.connect(reservoir)  # River inflow fills the reservoir
reservoir.connect(city)  # Reservoir supplies water to the city

# Step 3: Add data to the model
# River inflow data
river.add_data(df["river_inflow"].tolist())

# City demand data
city.add_data(df["city_demand"].tolist())

# Step 4: Run the simulation
model.run()

# Step 5: Analyze results
print("Reservoir levels over 30 days:")
for day, volume in enumerate(reservoir.volume_record, start=1):
    print(f"Day {day}: {volume} units")

