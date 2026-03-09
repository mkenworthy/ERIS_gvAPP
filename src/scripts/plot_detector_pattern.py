debug=0
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import paths

if debug:
    import matplotlib
    matplotlib.use('MacOSX')

def get_data(path):
    hdul = fits.open(path)
    return hdul[0].data

data_left = get_data(paths.data / 'ERIS_median_A_position_cuillin.fits.gz')

fig, ax = plt.subplots(1,1,figsize=(9,5))
im1 = ax.imshow(data_left[50:,100:-50], cmap='viridis', vmin=-2, vmax=7, origin='lower')

text_bbox = dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
arrprop = dict(facecolor='white', shrink=0.05)


ax.annotate('Left-hand coronagraphic PSF', xy=(340,300), xytext=(230,380), fontsize=9,
    va='top', ha='right',
            arrowprops=arrprop, bbox=text_bbox
            )

ax.annotate('Right-hand coronagraphic PSF', xy=(508,178), xytext=(610,120), fontsize=9,
    va='top', ha='left',
            arrowprops=arrprop, bbox=text_bbox
            )


ax.annotate('Right-hand astrometric spot', xy=(485,326), xytext=(544,421), fontsize=9,
    va='top', ha='left',
            arrowprops=arrprop, bbox=text_bbox
            )

ax.annotate('Right-hand astrometric spot', xy=(485,326), xytext=(544,421), fontsize=9,
    va='top', ha='left',
            arrowprops=arrprop, bbox=text_bbox
            )

ax.annotate('Leakage PSF', xy=(424,238), xytext=(575,365), fontsize=9,
    va='top', ha='left',
            arrowprops=arrprop, bbox=text_bbox
            )


ax.annotate('Left-hand astrometric spot', xy=(360,157), xytext=(321,100), fontsize=9,
    va='top', ha='right',
            arrowprops=arrprop, bbox=text_bbox
            )

ax.set_xticks([])  # Removes x-axis ticks and labels
ax.set_yticks([])  # Removes y-axis ticks and labels

plt.savefig(paths.figures / 'detector_pattern.pdf', bbox_inches='tight')

if debug:
    plt.show()