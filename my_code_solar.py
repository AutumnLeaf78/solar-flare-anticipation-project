import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
from sunkit_instruments.goes_xrs import calculate_temperature_em

# Ensure compatibility
print("Dependencies loaded successfully!")

# Define start and end times for data retrieval
tstart = "2024-02-25 00:00"
tend = "2024-02-25 23:59"

# Search and fetch GOES data
print("Searching for GOES data...")
result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"))
goes_files = Fido.fetch(result)
print("GOES data fetched successfully!")

# Load data into a SunPy TimeSeries object
goes_ts = TimeSeries(goes_files[0])

# Use calculate_temperature_em to get temperature and emission measure
print("Calculating temperature and emission measure...")
temp_em_data = calculate_temperature_em(goes_ts, abundance='coronal')

# Convert TimeSeries to DataFrame
data = goes_ts.to_dataframe().reset_index()
data = data.rename(columns={'index': 'time', 'xrsa': 'short_flux', 'xrsb': 'long_flux'})

# Add temperature and emission measure to the DataFrame
data['temperature'] = temp_em_data["temperature"]
data['emission_measure'] = temp_em_data["em"]

# Calculate the rate of change in emission measure (dEM/dt)
data['dEM_dt'] = data['emission_measure'].diff() / data['time'].diff().dt.total_seconds()

# Set FAI flag conditions
DELTA_EM_THRESHOLD = 1e-9  # Adjusted for solar flare sensitivity
TEMP_MIN = 1e6             # Minimum temperature in Kelvin
TEMP_MAX = 2e7             # Maximum temperature in Kelvin
DURATION_THRESHOLD = 600   # Time window in seconds

# FAI flagging function to mark only peaks
def calculate_fai_flags(data):
    flags = []
    for i, row in data.iterrows():
        # Define the sliding window around each point
        window_start = max(i - DURATION_THRESHOLD, 0)
        window_end = min(i + DURATION_THRESHOLD, len(data) - 1)
        window_data = data.iloc[window_start:window_end]
        
        # Check if any values in the window meet the conditions
        condition_met = (
            (window_data['dEM_dt'] >= DELTA_EM_THRESHOLD) &
            (window_data['temperature'] >= TEMP_MIN) &
            (window_data['temperature'] <= TEMP_MAX)
        ).any()

        if condition_met:
            # Identify the peak flux within the window
            peak_time_idx = window_data['long_flux'].idxmax()
            flags.append(1 if row.name == peak_time_idx else 0)  # Only flag the peak
        else:
            flags.append(0)
    
    return flags

# Apply the FAI flagging function
print("Applying FAI flagging...")
data['FAI_flag'] = calculate_fai_flags(data)

# Output some statistics
print("\nEmission Measure Change Rate (dEM_dt) Statistics:")
print(data['dEM_dt'].describe())
print("\nTemperature Statistics:")
print(data['temperature'].describe())

# Display flagged entries
flagged_entries = data[data['FAI_flag'] == 1][['time', 'FAI_flag']]
print("\nFlagged Entries:")
print(flagged_entries)

# Plot the data
fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# Plot 1: GOES Flux (1-8 Å and 0.5-4 Å)
axs[0].plot(data['time'], data['long_flux'], color='blue', label='GOES 1-8 Å Flux')
axs[0].plot(data['time'], data['short_flux'], color='orange', label='GOES 0.5-4 Å Flux')
axs[0].scatter(data[data['FAI_flag'] == 1]['time'], data[data['FAI_flag'] == 1]['long_flux'], 
                color='red', marker='x', label='FAI Flag (Peak)')
axs[0].set_yscale('log')
axs[0].set_ylabel('Flux (W/m²)')
axs[0].set_title('GOES X-ray Flux with FAI-based Solar Flare Detection')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Temperature
axs[1].plot(data['time'], data['temperature'], color='green', label='Temperature')
axs[1].scatter(data[data['FAI_flag'] == 1]['time'], data[data['FAI_flag'] == 1]['temperature'],
               color='red', marker='x', label='FAI Flag (Peak)')
axs[1].set_ylabel('Temperature (K)')
axs[1].set_title('Temperature during Solar Flare Events')
axs[1].legend()
axs[1].grid(True)

# Plot 3: dEM/dt (Rate of Change in Emission Measure)
axs[2].plot(data['time'], data['dEM_dt'], color='purple', label='dEM/dt')
axs[2].scatter(data[data['FAI_flag'] == 1]['time'], data[data['FAI_flag'] == 1]['dEM_dt'], 
               color='red', marker='x', label='FAI Flag (Peak)')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('dEM/dt (cm⁻³ s⁻¹)')
axs[2].set_title('Rate of Change of Emission Measure (dEM/dt) during Solar Flare Events')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
