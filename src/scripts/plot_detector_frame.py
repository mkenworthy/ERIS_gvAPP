debug=1
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import paths

if debug:
    print('debug')

def get_data(path):
    hdul = fits.open(path)
    return hdul[0].data

data_left = get_data(paths.data / 'ERIS_median_A_position_cuillin.fits.gz')

fig, ax = plt.subplots(1,1,figsize=(10,5))
im1 = ax.imshow(data_left, cmap='viridis', vmin=-2, vmax=1e1, origin='lower')

text_bbox = dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')

ax.text(0.05, 0.95, 'Left PSF', 
               fontsize=11, color='black', verticalalignment='top', bbox=text_bbox)

ax.set_xticks([])  # Removes x-axis ticks and labels
ax.set_yticks([])  # Removes y-axis ticks and labels


plt.savefig(paths.figures / 'combine_science2.pdf', bbox_inches='tight')
