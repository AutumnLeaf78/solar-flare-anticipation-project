import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
from sunkit_instruments.goes_xrs import calculate_temperature_em

# Define start and end times for the data retrieval
tstart = "2024-02-25 00:00"
tend = "2024-02-25 23:59"

# Search and fetch GOES data
result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"))
goes_files = Fido.fetch(result)
goes_ts = TimeSeries(goes_files[0])  # Load the TimeSeries object from the fetched files

# Convert the GOES data to a DataFrame for easier manipulation
data = goes_ts.to_dataframe().reset_index()
data.rename(columns={'index': 'time', 'xrsa': 'short_flux', 'xrsb': 'long_flux'}, inplace=True)

# Calculate temperature and emission measure using sunkit_instruments
temp_em_data = calculate_temperature_em(goes_ts, abundance="coronal")
temp_em_df = temp_em_data.to_dataframe()  # Convert to DataFrame

# Merge the calculated temperature and emission measure with the main DataFrame
data = data.merge(temp_em_df[['temperature', 'em']], left_on='time', right_index=True, how='inner')

# Rename the merged columns for clarity
data.rename(columns={'em': 'emission_measure'}, inplace=True)

# Calculate dEM/dt (rate of change in emission measure)
data['dEM_dt'] = data['emission_measure'].diff() / data['time'].diff().dt.total_seconds()

# Define FAI conditions
DELTA_EM_THRESHOLD = 1e-10  # Emission measure rate change threshold
TEMP_MIN = 1e6  # Minimum temperature for FAI condition (in Kelvin)
TEMP_MAX = 1e7  # Maximum temperature for FAI condition (in Kelvin)

# Apply the FAI condition
data['FAI_flag'] = (
    (data['dEM_dt'] >= DELTA_EM_THRESHOLD) &
    (data['temperature'] >= TEMP_MIN) &
    (data['temperature'] <= TEMP_MAX)
).astype(int)

# Display debugging information
print("Adjusted Emission Measure Change Rate (dEM_dt) Statistics:")
print(data['dEM_dt'].describe())

print("\nAdjusted Temperature Statistics:")
print(data['temperature'].describe())

# Save the flagged entries to a CSV file
flagged_entries = data[data['FAI_flag'] == 1][['time', 'FAI_flag']]
flagged_entries.to_csv("flagged_entries.csv", index=False)
print("\nFlagged entries saved to 'flagged_entries.csv'.")

# Plotting GOES Flux with FAI flags
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['long_flux'], color='blue', label='GOES 1-8 Å Flux')
plt.plot(data['time'], data['short_flux'], color='orange', label='GOES 0.5-4 Å Flux')
plt.scatter(data[data['FAI_flag'] == 1]['time'], data[data['FAI_flag'] == 1]['long_flux'], 
            color='red', marker='x', label='FAI Flag')

plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Flux (W/m²)')
plt.title('GOES X-ray Flux with FAI-based Solar Flare Detection')
plt.legend()
plt.grid(True)

# Save the plot to a file
plot_filename = "goes_flux_with_fai_flags.png"
plt.savefig(plot_filename, dpi=300)
print(f"Plot saved to '{plot_filename}'.")

plt.show()
