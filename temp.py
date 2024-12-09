import matplotlib.pyplot as plt
import numpy as np

# Define the time intervals and the corresponding slopes
time_intervals = [(0, 90), (90, 120), (120, 150)]
slopes = [0.0394, -0.0012, -0.0444]

# Prepare data for the plot
times = []
temperature_changes = []
current_temp_change = 0  # Initialize the temperature change at time 0

for (start, end), slope in zip(time_intervals, slopes):
    # Generate time points for this interval
    time_points = np.linspace(start, end, num=50)  # 50 points for smoothness
    # Calculate temperature changes for this interval
    temp_changes = current_temp_change + slope * (time_points - start)
    # Append to the lists
    times.extend(time_points)
    temperature_changes.extend(temp_changes)
    # Update the current temperature change for the next interval
    current_temp_change = temp_changes[-1]

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(times, temperature_changes, label="Temperature Slope Change", color="b", marker="o")

# Add labels and title
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Temperature Slope Change per Minute", fontsize=12)
plt.title("Dependency of Temperature Slope Change on Time", fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
