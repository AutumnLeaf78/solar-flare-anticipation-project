import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries as ts
from goes_chianti_tem_mod import calculate_temperature_em_mod
from tqdm import tqdm
from datetime import timedelta
from astropy import units as u

# Define start and end times for the data retrieval
tstart = "2024-02-25 00:00"
tend = "2024-02-25 23:59"

# Search and fetch GOES data
result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"))
goes_files = Fido.fetch(result)
goes_ts_orig = ts(goes_files[0])  # Load the TimeSeries object from the fetched files

# Convert the GOES data to a DataFrame for easier manipulation
data = goes_ts_orig.to_dataframe().reset_index()
data.rename(columns={'index': 'time', 'xrsa': 'short_flux', 'xrsb': 'long_flux'}, inplace=True)

# Ensure 'time' is in datetime format
data['time'] = pd.to_datetime(data['time'])

# Create difference timeseries

# Define the interval in seconds for calculating rate of change
interval = 300  # 300 seconds = 5 minutes
delta_t = timedelta(seconds=interval)  # Interval as a timedelta object
time = data['time']

xrsa = data['short_flux']
data['xrsa_diff'] = np.nan
xrsb = data['long_flux']
data['xrsb_diff'] = np.nan

# Loop through each time point and compute difference of xrsa & xrsb
for i in tqdm(range(len(time))):
    ind = np.argmin(np.abs(time-(time[i] + delta_t))) # time[i] + delta_t is the target time for the interval
    # Ensure index is valid and within range
    if ind < len(xrsa):
        data.iloc[i, data.columns.get_loc('xrsa_diff')] = (xrsa[ind] - xrsa[i])
        data.iloc[i, data.columns.get_loc('xrsb_diff')] = (xrsb[ind] - xrsb[i])

xrsa_diff = u.Quantity(data['xrsa_diff'], u.W/(u.m*u.m))
xrsb_diff = u.Quantity(data['xrsb_diff'], u.W/(u.m*u.m))

# Add the differenced columns to the GOES timeseries
goes_ts_orig = goes_ts_orig.add_column('xrsa_diff', xrsa_diff)
goes_ts_orig = goes_ts_orig.add_column('xrsb_diff', xrsb_diff)

# Calculate temperature and emission measure using sunkit_instruments
temp_em_data = calculate_temperature_em_mod(goes_ts_orig, abundance="coronal")
temp_em_df = temp_em_data.to_dataframe().reset_index()  # Convert to DataFrame
temp_em_df.rename(columns={'index': 'time'}, inplace=True)

# Merge the calculated temperature and emission measure with the main DataFrame
data = data.merge(temp_em_df[['temperature', 'emission_measure']], left_index=True, right_index=True, how='inner')

# Define the interval in seconds for calculating rate of change
interval = 300  # 300 seconds = 5 minutes
delta_t = timedelta(seconds=interval)  # Interval as a timedelta object

# Convert time to a NumPy array for performance optimization
time = data['time']#np.array(data.index)

# Define the emission measure as a NumPy array
em = np.array(data['emission_measure'])

# Initialize dEM_dt column
data['dEM_dt'] = np.nan  # Use NaN initially to avoid incorrect values

# Loop through each time point and compute dEM/dt
for i in tqdm(range(len(time))):
    ind = np.argmin(np.abs(time-(time[i] + delta_t))) # time[i] + delta_t is the target time for the interval
    # Ensure index is valid and within range
    if ind < len(em):
        data.iloc[i, data.columns.get_loc('dEM_dt')] = (em[ind] - em[i]) / interval  # (ΔEM / Δt)

# Define FAI conditions
DELTA_EM_THRESHOLD = 1e-10  # Emission measure rate change threshold
TEMP_MIN = 6  # Minimum temperature for FAI condition (in MK)
TEMP_MAX = 20  # Maximum temperature for FAI condition (in MK)

# Apply the FAI condition
data['FAI_flag'] = (
    (data['dEM_dt'] >= DELTA_EM_THRESHOLD) &
    (data['temperature'] >= TEMP_MIN) &
    (data['temperature'] <= TEMP_MAX)
).astype(int)

# Reset index to bring 'time' back as a column
data.reset_index(inplace=True)

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
