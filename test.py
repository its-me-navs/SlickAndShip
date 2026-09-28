import os
os.environ["USE_REAL_ENV"] = "1"
from drift.environment_real import get_current, get_wind

# pick a point/time inside your fetched window (t_hours ~82, mid-region)
print(get_wind(15.3, 73.4, 90.0))
print(get_current(15.3, 73.4, 90.0))