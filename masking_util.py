import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def mask_invalid_data_2d(data_2d, label="", verbose=False):
    """
    Masks NaN and Inf values in a 2D array and prints masking statistics.
    """
    invalid_mask = np.ma.masked_invalid(data_2d)
    masked_indices = np.where(invalid_mask.mask)
    num_masked = len(masked_indices[0])
    
    if num_masked > 0:
        print(f"{label}: {num_masked} elements masked due to NaN or Inf.")
        if verbose:
            for t_idx, f_idx in zip(masked_indices[0], masked_indices[1]):
                print(f"  Masked at Time index: {t_idx}, Freq index: {f_idx}")
                
    return invalid_mask

def mask_invalid_data(data, label="", verbose=False):
    """
    Masks NaN and Inf values in an array of any dimensionality 
    and prints masking statistics.
    """
    invalid_mask = np.ma.masked_invalid(data)
    masked_indices = np.where(invalid_mask.mask)
    num_masked = len(masked_indices[0])
    
    if num_masked > 0:
        
        if verbose:
            print(f"{label}: {num_masked} elements masked due to NaN or Inf.")
            # zip the indices to get coordinates regardless of dimensionality
            #coords = list(zip(*masked_indices))
            #for coord in coords:
            #    print(f"  Masked at index: {coord}")
    return invalid_mask

def label_curve_at_peak(ax, line):
    """
    Finds the peak of a curve from a Line2D object and places 
    its label text above the peak with center alignment and offset.
    Robust against masked data.
    """
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    label_text = line.get_label()
    color = line.get_color()

    # Handle masked arrays or standard arrays with potential NaNs
    if not isinstance(y_data, np.ma.MaskedArray):
        y_data = np.ma.masked_invalid(y_data)

    if y_data.count() > 0:
        # Find the maximum value among unmasked elements
        max_val = np.ma.max(y_data)
        
        # Find the index of the first occurrence of the max value among unmasked elements
        # np.ma.argmax handles the mask correctly
        max_idx = np.ma.argmax(y_data)
        
        peak_x = x_data[max_idx]
        
        # Calculate a vertical offset (2% of the y-axis range)
        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * 0.02
        
        ax.text(peak_x, max_val + offset, f' {label_text}', 
                color=color, fontsize=12, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='center')


def load_freq_band(band_file):
    """
    Parses a frequency band file and returns a tuple:
    (boolean_mask, list_of_ranges)
    """
    print('loading frequency bands from file [%s]'%band_file)
 
    ranges = []
    if band_file and os.path.exists(band_file):
        with open(band_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    lo, hi = float(parts[0]), float(parts[1])
                    ranges.append([lo, hi])
    return ranges

def load_freq_mask(mask_file, freq_array):
    """
    Parses a frequency mask file and returns a boolean array.
    """
    print('loading frequency mask from file [%s]'%mask_file)
    mask = np.zeros(len(freq_array), dtype=bool)
    if mask_file and os.path.exists(mask_file):
        with open(mask_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    lo, hi = float(parts[0]), float(parts[1])
                    mask |= (freq_array >= lo) & (freq_array <= hi)
    else:
            print(f"Warning: Mask file {mask_file} not found.")

    return mask

def get_robust_threshold(data, factor=5.0):
    """Calculate threshold using Median Absolute Deviation (MAD)."""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return med + factor * (mad * 1.4826) # 1.4826 scales MAD to approx std dev

#def get_saturation_mask(data_2d, pwr_factor=10.0, var_factor=10.0, show_hist=False):
def get_saturation_mask(data_2d, MAD_thresh_dB=10.0, show_hist=False):
    """
    Identifies erratic/saturated time windows.
    Robust against CW interference by using MAD-based metrics instead of STD.
    
    Args:
        data_2d: 2D array (time, freq)
        pwr_factor: sensitivity for broadband power surges
        var_factor: sensitivity for spectral shape distortion (using spectral MAD)
    """

    debug = False
   
    # Convert to MaskedArray if it isn't one already to ensure consistent behavior
    if not isinstance(data_2d, np.ma.MaskedArray):
        data_2d = np.ma.masked_invalid(data_2d)

    # 1. Global reference (median over time)
    # Using np.ma.median ensures masked values are ignored
    global_median_spectrum_dB = 10 * np.ma.log10( np.ma.median(data_2d, axis=0) )
    
    # Optional diagnostic plot of the bandpass
    if debug:
        plt.figure(figsize=(8, 4))
        plt.plot(global_median_spectrum_dB, color='black', label='Reference Spectrum')
     
        plt.title("Global Median Spectrum (Reference) ")
        plt.xlabel("Frequency Bin")
        plt.ylabel("Intensity")
        plt.grid(True, alpha=0.3)
        plt.show()


    # 2. Broadband power check (in dB)
    # Power surges are easier to threshold in dB
    #median_pwr = np.ma.median(data_2d, axis=1) 
    #median_pwr_dB = 10 * np.ma.log10(median_pwr)
    
    # 3. Deviation from reference shape (in dB)
    # Using dB residuals makes the MAD sensitive to relative % changes
    data_dB = 10 * np.ma.log10(data_2d)
    
    
    residuals_dB = data_dB - global_median_spectrum_dB[None, :]
    spectral_mad_dB = np.ma.median(np.abs(residuals_dB), axis=1)
    
    # Calculate robust thresholds based on dB metrics
    #pwr_thresh_dB = get_robust_threshold(median_pwr_dB, factor=pwr_factor)
    #var_thresh_dB = get_robust_threshold(spectral_mad_dB, factor=var_factor)
    var_thresh_dB = MAD_thresh_dB
    

    #print('spectral MAD shape:', spectral_mad_dB.shape)
    #print("spectral MAD: ", spectral_mad_dB)
   

    # 4. Optional Diagnostic Histogram
    if show_hist:

        plot_MAD_histogram(spectral_mad_dB, var_thresh_dB) #, label=label
     
    
    # Flag window if Broadband Power is high OR the Spectrum base is erratic
    time_mask =  (spectral_mad_dB > var_thresh_dB)  #(median_pwr_dB > pwr_thresh_dB) |
    masked_indices = np.where(time_mask)
    num_masked = len(masked_indices[0])
    print("Masking saturated time window: %d out of %d time bins are masked"%( num_masked, len(time_mask) ))


    return time_mask

def plot_MAD_histogram(mad_data, threshold, label=""):
    """
    Plots a histogram of the calculated MAD values to help verify thresholds.
    """
    clean_mad = mad_data[np.isfinite(mad_data)]
    
    print('MAD data size: ', mad_data.shape, clean_mad.shape)
    if len(clean_mad) == 0:
        print(f"Warning: No finite MAD data to plot for {label}")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(clean_mad, bins=50, log=True, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.title(f"MAD Distribution - {label}")
    plt.xlabel("MAD (Relative to Global Median)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_masked_pcolormesh(ax, x, y, data_2d, freq_mask=None, time_mask=None, vmin=None, vmax=None, shade_color='gray'):
    """
    Plots pcolormesh with:
    1. Invalid numbers (NaN/Inf) as BLANKS.
    2. Time and Freq masks as SHADED regions.
    Returns the final combined MaskedArray used for profiles.
    """
    # 1. Blank out invalid numbers
    invalid_masked = np.ma.masked_invalid(data_2d)
    
    # 2. Prepare shading overlay
    overlay_mask = np.zeros_like(data_2d, dtype=bool)
    if freq_mask is not None:
        overlay_mask[:, freq_mask] = True
    if time_mask is not None:
        overlay_mask[time_mask, :] = True
        
    # Plot base data (invalids are blanks)
    ax.pcolormesh(x, y, invalid_masked.T, vmin=vmin, vmax=vmax, shading='auto')
    
    # Plot shading layer
    shade_data = np.full_like(data_2d, np.nan)
    shade_data[overlay_mask] = 1.0
    cmap_shade = mcolors.ListedColormap([mcolors.to_rgba(shade_color, alpha=0.4)])
    ax.pcolormesh(x, y, shade_data.T, cmap=cmap_shade, shading='auto', zorder=2)
    
    # Final combined masked array for statistical use
    final_masked = np.ma.array(data_2d, mask=invalid_masked.mask | overlay_mask)
    return final_masked

def get_beam_colors(nAnt, darken_factor=0.8, cmap_name='gist_rainbow'):
    """
    Generates a list of darkened colors for nAnt antennas in a single call.
    """
    #turbo gist_rainbow nipy_spectral, rainbow: green is too light, need to be darkened

    base_cmap = plt.get_cmap(cmap_name)
    # Calculate rainbow color based on beam index
    indices = np.linspace(0, 1, nAnt)
    colors = base_cmap(indices)
    
    # Darken RGB components (first 3 columns), keep Alpha (last column)
    colors[:, :3] *= darken_factor
    #dark_c = (orig_c[0]*0.8, orig_c[1]*0.8, orig_c[2]*0.8, 1.0)
    return colors