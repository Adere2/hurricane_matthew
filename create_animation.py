import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import glob
import os
from datetime import datetime
import wrf
import pandas as pd

# Setup
plt.style.use('dark_background')
output_dir = 'animation_frames'
os.makedirs(output_dir, exist_ok=True)

# Get all WRF output files
wrf_files = sorted(glob.glob('wrfout_d01_2016-10-*'))
print(f"Found {len(wrf_files)} WRF output files")

# Custom colormap for dramatic effect
def create_dramatic_cmap():
    colors = ['#000033', '#000066', '#003399', '#0066CC', '#00CCFF', 
              '#66FF99', '#FFFF00', '#FF9900', '#FF3300', '#CC0000', '#990000']
    return LinearSegmentedColormap.from_list('dramatic', colors, N=256)

def create_wind_cmap():
    colors = ['#000000', '#330066', '#660099', '#9900CC', '#CC00FF',
              '#FF0099', '#FF3366', '#FF6633', '#FF9900', '#FFCC00', '#FFFF00']
    return LinearSegmentedColormap.from_list('wind', colors, N=256)

dramatic_cmap = create_dramatic_cmap()
wind_cmap = create_wind_cmap()

def plot_hurricane_frame(wrf_file, frame_num):
    """Create a single frame of the hurricane animation"""
    
    # Open WRF file
    ncfile = Dataset(wrf_file)
    
    # Get coordinates
    lats, lons = wrf.latlon_coords(wrf.getvar(ncfile, "slp"))
    
    # Extract variables
    slp = wrf.getvar(ncfile, "slp")  # Sea level pressure
    
    # Get 2-meter temperature in Kelvin (its native unit)
    t2_kelvin = wrf.getvar(ncfile, "T2")
    # Manually convert from Kelvin to Celsius
    temp = t2_kelvin - 273.15
    temp.attrs['description'] = '2m Temperature in Celsius'
    temp.attrs['units'] = 'degC'
    
    u10 = wrf.getvar(ncfile, "U10")  # 10m U wind
    v10 = wrf.getvar(ncfile, "V10")  # 10m V wind
    wind_speed = np.sqrt(u10**2 + v10**2) * 1.94384  # Convert to knots
    
    # Get precipitation if available
    try:
        precip = wrf.getvar(ncfile, "RAINNC") + wrf.getvar(ncfile, "RAINC")
    except:
        precip = None
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Get the time value, which is a numpy.datetime64 object in this wrf-python version
    time_val = wrf.extract_times(ncfile, 0)
    # Convert the numpy time to a pandas Timestamp, which supports strftime
    time_str = pd.to_datetime(time_val).strftime('%Y-%m-%d %H:%M UTC')
    # -------------------------------------
    
    # Create figure with dramatic styling
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    
    # Main plot - Hurricane overview
    ax1 = plt.subplot(2, 2, (1, 2), projection=ccrs.PlateCarree())
    ax1.set_facecolor('black')
    
    # Add map features
    ax1.add_feature(cfeature.COASTLINE, color='white', linewidth=0.8)
    ax1.add_feature(cfeature.BORDERS, color='white', linewidth=0.5)
    ax1.add_feature(cfeature.STATES, color='gray', linewidth=0.3)
    ax1.add_feature(cfeature.OCEAN, color='#001122')
    ax1.add_feature(cfeature.LAND, color='#001100')
    
    # Plot sea level pressure with dramatic colors
    pressure_plot = ax1.contourf(lons, lats, slp, levels=50, 
                                cmap=dramatic_cmap, alpha=0.8,
                                transform=ccrs.PlateCarree())
    
    # Add pressure contours
    pressure_contours = ax1.contour(lons, lats, slp, levels=20, 
                                   colors='white', linewidths=0.8,
                                   transform=ccrs.PlateCarree())
    ax1.clabel(pressure_contours, inline=True, fontsize=8, fmt='%d')
    
    # Add wind barbs for dramatic effect
    skip = 6  # Skip every 6th point for clarity
    ax1.barbs(lons[::skip, ::skip], lats[::skip, ::skip], 
             u10[::skip, ::skip], v10[::skip, ::skip],
             transform=ccrs.PlateCarree(), color='cyan', 
             length=6, linewidth=1.5, alpha=0.7)
    
    # Set extent around hurricane
    ax1.set_extent([-85, -65, 20, 35], ccrs.PlateCarree())
    ax1.gridlines(draw_labels=True, alpha=0.3, color='white')
    
    # Wind speed subplot
    ax2 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax2.set_facecolor('black')
    ax2.add_feature(cfeature.COASTLINE, color='white', linewidth=0.5)
    ax2.add_feature(cfeature.OCEAN, color='#001122')
    ax2.add_feature(cfeature.LAND, color='#001100')
    
    wind_plot = ax2.contourf(lons, lats, wind_speed, levels=30,
                            cmap=wind_cmap, alpha=0.9,
                            transform=ccrs.PlateCarree())
    ax2.set_extent([-85, -65, 20, 35], ccrs.PlateCarree())
    ax2.set_title('Wind Speed (knots)', color='white', fontsize=12)
    
    # Temperature subplot
    ax3 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax3.set_facecolor('black')
    ax3.add_feature(cfeature.COASTLINE, color='white', linewidth=0.5)
    ax3.add_feature(cfeature.OCEAN, color='#001122')
    ax3.add_feature(cfeature.LAND, color='#001100')
    
    temp_plot = ax3.contourf(lons, lats, temp, levels=30,
                            cmap='RdYlBu_r', alpha=0.8,
                            transform=ccrs.PlateCarree())
    ax3.set_extent([-85, -65, 20, 35], ccrs.PlateCarree())
    ax3.set_title('Temperature (°C)', color='white', fontsize=12)
    
    # Add colorbars
    cbar1 = plt.colorbar(pressure_plot, ax=ax1, shrink=0.6, pad=0.02)
    cbar1.set_label('Sea Level Pressure (hPa)', color='white')
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
    
    cbar2 = plt.colorbar(wind_plot, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
    
    cbar3 = plt.colorbar(temp_plot, ax=ax3, shrink=0.8, pad=0.02)
    cbar3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')
    
    # Main title
    fig.suptitle(f'Hurricane Matthew - WRF Model\n{time_str}', 
                color='white', fontsize=16, fontweight='bold')
    
    # Find minimum pressure for intensity indication
    min_pressure = float(np.min(slp))
    max_wind = float(np.max(wind_speed))
    
    # Add intensity text
    intensity_text = f'Min Pressure: {min_pressure:.1f} hPa\nMax Wind: {max_wind:.1f} knots'
    fig.text(0.02, 0.95, intensity_text, color='cyan', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    # Save frame
    frame_file = f'{output_dir}/frame_{frame_num:03d}.png'
    plt.savefig(frame_file, dpi=150, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()
    
    ncfile.close()
    
    print(f"Created frame {frame_num}: {time_str} (Min P: {min_pressure:.1f} hPa, Max Wind: {max_wind:.1f} kt)")
    
    return frame_file

def create_animation():
    """Create the complete animation"""
    
    print("Creating hurricane animation frames...")
    frame_files = []
    
    for i, wrf_file in enumerate(wrf_files):
        try:
            frame_file = plot_hurricane_frame(wrf_file, i)
            frame_files.append(frame_file)
        except Exception as e:
            print(f"Error processing {wrf_file}: {e}")
            continue
    
    print(f"\nCreated {len(frame_files)} frames")
    
    return frame_files

if __name__ == "__main__":
    # Check if required modules are available
    try:
        import wrf
        import cartopy
        print("All required modules found. Starting animation creation...")
        create_animation()
    except ImportError as e:
        print(f"Missing required module: {e}")
        print("Please ensure wrf-python, cartopy, and pandas are installed.")
        print("\nInstall required packages with:")
        print("conda install -c conda-forge wrf-python cartopy pandas")
        print("or")
        print("pip install wrf-python cartopy pandas")
