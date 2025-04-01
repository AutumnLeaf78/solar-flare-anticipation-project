import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
from goes_chianti_tem_mod import calculate_temperature_em_mod
from tqdm import tqdm
from datetime import timedelta
from astropy import units as u

# **Step 1: Define start and end times for 3-day data retrieval**
tstart = "2024-01-02 12:00"
tend = "2024-01-05 12:00"

# **Step 2: Search and fetch GOES data**
result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"))
goes_files = Fido.fetch(result)  # Fetch all files

# **Step 3: Load all GOES TimeSeries and merge them into one**
goes_ts_list = [TimeSeries(file) for file in goes_files]
goes_ts_combined = TimeSeries(goes_ts_list, concatenate=True)  # Merge all TimeSeries

# **Step 4: Convert merged TimeSeries to a DataFrame**
data = goes_ts_combined.to_dataframe().reset_index()
data.rename(columns={'index': 'time', 'xrsa': 'short_flux', 'xrsb': 'long_flux'}, inplace=True)
data['time'] = pd.to_datetime(data['time'])  # Ensure correct datetime format

# **Step 5: Compute xrsb_diff and xrsa_diff & Handle NaN/Zero Issues**
interval = 300  # 5 minutes
delta_t = timedelta(seconds=interval)  # Interval as a timedelta object
#data['xrsb_diff'] = data['long_flux'].diff(periods=int(interval / 2)).fillna(0)
#data['xrsa_diff'] = data['short_flux'].diff(periods=int(interval / 2)).fillna(0)

# Loop through each time point and compute difference of xrsa & xrsb
time = data['time']
xrsa = data['short_flux']
data['xrsa_diff'] = 0
xrsb = data['long_flux']
data['xrsb_diff'] = 0

for i in tqdm(range(len(time))):
    ind = np.argmin(np.abs(time-(time[i] + delta_t))) # time[i] + delta_t is the target time for the interval
    # Ensure index is valid and within range
    if ind < len(xrsa):
        data.iloc[i, data.columns.get_loc('xrsa_diff')] = (xrsa[ind] - xrsa[i])
        data.iloc[i, data.columns.get_loc('xrsb_diff')] = (xrsb[ind] - xrsb[i])

# **Fix FutureWarning by replacing inplace=True with direct assignment**
data['xrsb_diff'] = data['xrsb_diff'].replace(0, 1e-20)
data['xrsa_diff'] = data['xrsa_diff'].replace(0, 1e-20)

# **Step 6: Convert back to SunPy TimeSeries and Add the Diff Columns**
xrsa_diff = u.Quantity(data['xrsa_diff'], u.W/(u.m**2))
xrsb_diff = u.Quantity(data['xrsb_diff'], u.W/(u.m**2))
goes_ts_combined = goes_ts_combined.add_column('xrsa_diff', xrsa_diff)
goes_ts_combined = goes_ts_combined.add_column('xrsb_diff', xrsb_diff)

# **Step 7: Compute temperature and emission measure**
temp_em_data = calculate_temperature_em_mod(goes_ts_combined, abundance="coronal")
temp_em_df = temp_em_data.to_dataframe().reset_index()
temp_em_df.rename(columns={'index': 'time'}, inplace=True)

# Merge calculated temperature and emission measure
data = data.merge(temp_em_df[['time', 'temperature', 'emission_measure']], on='time', how='inner')

# **Step 8: Compute dEM/dt Efficiently & Handle Divide-by-Zero Errors**
#data['dEM_dt'] = data['emission_measure'].diff(periods=int(interval / 2)).fillna(0)

# Define the emission measure as a NumPy array
em = np.array(data['emission_measure'])

# Initialize dEM_dt column
data['dEM_dt'] = 0  # Use 0 initially to avoid incorrect values

# Loop through each time point and compute dEM/dt
for i in tqdm(range(len(time))):
    ind = np.argmin(np.abs(time-(time[i] + delta_t))) # time[i] + delta_t is the target time for the interval
    # Ensure index is valid and within range
    if ind < len(em):
        data.iloc[i, data.columns.get_loc('dEM_dt')] = (em[ind] - em[i]) / interval  # (ΔEM / Δt)

# **Fix FutureWarning by replacing inplace=True with direct assignment**
data['dEM_dt'] = data['dEM_dt'].replace(0, 1e-20)

# **Step 9: Define FAI conditions**
DELTA_EM_THRESHOLD = 0.5e-5
TEMP_MIN = 6
TEMP_MAX = 20
data['FAI_flag'] = (
    (data['dEM_dt'] >= DELTA_EM_THRESHOLD) &
    (data['temperature'] >= TEMP_MIN) &
    (data['temperature'] <= TEMP_MAX)
).astype(int)

# **Step 10: Reset index for final DataFrame**
data.reset_index(inplace=True)

# **Step 11: Plot the combined 3-day data**

# **Flux Plot**
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['long_flux'], color='blue', label='GOES 1-8 Å Flux')
plt.plot(data['time'], data['short_flux'], color='orange', label='GOES 0.5-4 Å Flux')
plt.scatter(data[data['FAI_flag'] == 1]['time'], data[data['FAI_flag'] == 1]['long_flux'],
            color='red', marker='x', label='FAI Flag')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Flux (W/m²)')
plt.title('GOES X-ray Flux (3-Day Period) with FAI-based Solar Flare Detection')
plt.legend()
plt.grid(True)
plt.show()

# **Emission Measure Plot**
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['emission_measure'], color='green', label='Emission Measure')
plt.xlabel('Time')
plt.ylabel('Emission Measure')
plt.title('Emission Measure vs. Time (3-Day Period)')
plt.legend()
plt.grid(True)
plt.show()

# **Temperature Plot**
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['temperature'], color='purple', label='Temperature (MK)')
plt.xlabel('Time')
plt.ylabel('Temperature (MK)')
plt.title('Temperature vs. Time (3-Day Period)')
plt.ylim(0, 0.2e7)  # Set y-axis range
plt.legend()
plt.grid(True)
plt.show()

# **DEM Plot**
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['dEM_dt'], color='red', label='DEM (dEM/dt)')
plt.xlabel('Time')
plt.ylabel('DEM (dEM/dt)')
plt.title('Differential Emission Measure (DEM) vs. Time (3-Day Period)')
plt.legend()
plt.grid(True)
plt.show()
