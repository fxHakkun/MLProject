import pywr
import datetime
import pandas
from pywr.core import Model, Timestepper
from pywr.nodes import Node, Storage, Input, Output
from pywr.recorders import NumpyArrayNodeRecorder

# Create a new model
model = Model()

model.timestepper = Timestepper(
    pandas.to_datetime('2015-01-01'),  # first day
    pandas.to_datetime('2015-12-31'),  # last day
    datetime.timedelta(1)  # interval
)
# Create nodes
reservoir = Storage(model, name="Reservoir", max_volume=100, initial_volume=50)
demand = Output(model, name="Demand", max_flow=20)
river = Input(model, name="River")

# Connect nodes
river.connect(reservoir)
reservoir.connect(demand)

# set cost (+ve) or benefit (-ve)
reservoir.cost = 10.0
river.cost = 3.0
demand.cost = -100.0

# Add a recorder to track output
recorder = NumpyArrayNodeRecorder(model, reservoir)

# Run the model
model.run()

scenario = 0
timestep = 0
print(recorder.data[scenario][timestep])