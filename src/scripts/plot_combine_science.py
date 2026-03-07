import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cmocean
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import paths

#original file was cut_stich_plot.py in subdir cut_plot/ sent to MAK from Felix Dannert 2026 Feb 23

# MAK note: original data cube science.fits was around 250 Mb, so this script
# uses the mean image instead saved via the commented out code below.

#path_left = os.path.join(root, 'sci_left.fits')
#path_right = os.path.join(root, 'sci_right.fits')
#path_comb = os.path.join(root, 'science.fits')
#path_comb = os.path.join(root, 'science_mean.fits')

def get_data(path):
    hdul = fits.open(path)
    return hdul[0].data

data_left = get_data(paths.data / 'sci_left.fits.gz')
data_right = get_data(paths.data / 'sci_right.fits.gz')
#data_comb = np.mean(get_data(path_comb), axis=0)
#data_com_out = fits.writeto('science_mean.fits',data_comb)
data_comb = get_data(paths.data / 'science_mean.fits.gz')


def center_crop_square(matrix, crop_size):
    """
    Crops a square numpy matrix to a specific border length, centered.
    """
    current_size = matrix.shape[0]
    
    # Calculate the starting index for the crop
    start = (current_size - crop_size) // 2
    end = start + crop_size
    
    # Slice both dimensions
    return matrix[start:end, start:end]

data_left = center_crop_square(data_left, data_comb.shape[0])
data_right = center_crop_square(data_right, data_comb.shape[0])

layout = """
AC
BC
"""

erot = -7

cmap = cmocean.cm.thermal
# set_bad color for masked/NaN values
cmap.set_bad(cmocean.cm.thermal(0), 1.)

# Create the figure and axes based on the layout
# NEW: Added gridspec_kw to reduce horizontal spacing (wspace) between the columns
fig, axes = plt.subplot_mosaic(layout, figsize=(10, 5), gridspec_kw={'wspace': -0.3}, 
                               #layout='constrained'
                              )

# 3. Plot the matrices using imshow
# Top Left
im1 = axes['A'].imshow(np.log10(data_left), cmap=cmap, vmin=0.1)

# Bottom Left
im2 = axes['B'].imshow(np.log10(data_right), cmap=cmap, vmin=0)

# Right (Double Size)
im3 = axes['C'].imshow(np.log10(data_comb), cmap=cmap, vmin=2)

# --- NEW: Add partial circles (arcs) and lines to the left plots ---
arc_center = ((data_left.shape[0]-2)/2, (data_left.shape[0]-2)/2)  # Center (x, y) in pixel coordinates
arc_radius_px = 70      # Radius in pixels

total = 305 - 165 + 16 - 5
rot = [157 + 180, 157]

start_angle = [rot[0], rot[1]]       # Start angle in degrees
end_angle = [rot[0] + total, rot[1] + total]        # End angle in degrees

start_angle = [154+180+erot, 154+erot]
end_angle = [327+180+erot, 327+erot]

# Add arc and connecting line to Top Left ('A')
arc_a = patches.Arc(arc_center, arc_radius_px * 2, arc_radius_px * 2,
                    theta1=start_angle[0], theta2=end_angle[0],
                    color='white', linewidth=2, linestyle='dashed')
axes['A'].add_patch(arc_a)

# Calculate endpoints and draw line for 'A'
x1_a = arc_center[0] + arc_radius_px * np.cos(np.radians(start_angle[0]))
y1_a = arc_center[1] + arc_radius_px * np.sin(np.radians(start_angle[0]))
x2_a = arc_center[0] + arc_radius_px * np.cos(np.radians(end_angle[0]))
y2_a = arc_center[1] + arc_radius_px * np.sin(np.radians(end_angle[0]))
axes['A'].plot([x1_a, x2_a], [y1_a, y2_a], color='white', linewidth=2, linestyle='dashed')

# Add arc and connecting line to Bottom Left ('B')
arc_b = patches.Arc(arc_center, arc_radius_px * 2, arc_radius_px * 2,
                    theta1=start_angle[1], theta2=end_angle[1],
                    color='white', linewidth=2, linestyle='dashed')
axes['B'].add_patch(arc_b)

# Calculate endpoints and draw line for 'B'
x1_b = arc_center[0] + arc_radius_px * np.cos(np.radians(start_angle[1]))
y1_b = arc_center[1] + arc_radius_px * np.sin(np.radians(start_angle[1]))
x2_b = arc_center[0] + arc_radius_px * np.cos(np.radians(end_angle[1]))
y2_b = arc_center[1] + arc_radius_px * np.sin(np.radians(end_angle[1]))
axes['B'].plot([x1_b, x2_b], [y1_b, y2_b], color='white', linewidth=2, linestyle='dashed')

# --- NEW: Add both arcs and connecting lines to Right ('C') ---
# Scaling the center and radius to fit the larger data_comb plot
scale_c = data_comb.shape[0] / data_left.shape[0]
arc_center_c = ((data_comb.shape[0])/2, (data_comb.shape[1])/2)
arc_radius_px_c = arc_radius_px * scale_c

# Arc 1 (corresponding to Top Left 'A')
arc_c1 = patches.Arc(arc_center_c, arc_radius_px_c * 2, arc_radius_px_c * 2,
                     theta1=start_angle[0], theta2=end_angle[0],
                     color='white', linewidth=2, linestyle='dashed')
axes['C'].add_patch(arc_c1)
x1_c1 = arc_center_c[0] + arc_radius_px_c * np.cos(np.radians(start_angle[0]))
y1_c1 = arc_center_c[1] + arc_radius_px_c * np.sin(np.radians(start_angle[0]))
x2_c1 = arc_center_c[0] + arc_radius_px_c * np.cos(np.radians(end_angle[0]))
y2_c1 = arc_center_c[1] + arc_radius_px_c * np.sin(np.radians(end_angle[0]))
#axes['C'].plot([x1_c1, x2_c1], [y1_c1, y2_c1], color='white', linewidth=2, linestyle='dashed')

# Arc 2 (corresponding to Bottom Left 'B')
arc_c2 = patches.Arc(arc_center_c, arc_radius_px_c * 2, arc_radius_px_c * 2,
                     theta1=start_angle[1], theta2=end_angle[1],
                     color='white', linewidth=2, linestyle='dashed')
axes['C'].add_patch(arc_c2)
x1_c2 = arc_center_c[0] + arc_radius_px_c * np.cos(np.radians(start_angle[1]))
y1_c2 = arc_center_c[1] + arc_radius_px_c * np.sin(np.radians(start_angle[1]))
x2_c2 = arc_center_c[0] + arc_radius_px_c * np.cos(np.radians(end_angle[1]))
y2_c2 = arc_center_c[1] + arc_radius_px_c * np.sin(np.radians(end_angle[1]))
#axes['C'].plot([x1_c2, x2_c2], [y1_c2, y2_c2], color='white', linewidth=2, linestyle='dashed')

# --- NEW: Add rotated rectangle with hatching to Right ('C') ---

# lambda / d in pixels
lod_px = 4e-6/8.2 * 3600 * 180 / np.pi / 0.013

rect_short_side = lod_px  # Defined length for the shorter side
rect_long_side = data_comb.shape[0] * 1.5  # Extends beyond the image (e.g. 600)
rect_angle = 165 + total/2 + erot  # Given angle in degrees

# Calculate the bottom-left corner of the unrotated rectangle so it is centered
rect_x = arc_center_c[0] - rect_short_side / 2
rect_y = arc_center_c[1] - rect_long_side / 2

# Create the rectangle with hatching and alpha=0.5
rect = patches.Rectangle((rect_x, rect_y), rect_short_side, rect_long_side,
                         facecolor='none', edgecolor='white', hatch='//', 
                         alpha=1, linewidth=2)

# Create the transformation: rotate around the center of the image
t_rect = transforms.Affine2D().rotate_deg_around(arc_center_c[0], arc_center_c[1], rect_angle)
rect.set_transform(t_rect + axes['C'].transData)

axes['C'].add_patch(rect)

# Restore the original axis limits for the right plot
axes['C'].set_xlim(-0.5, data_comb.shape[1] - 0.5)
axes['C'].set_ylim(data_comb.shape[0] - 0.5, -0.5)
# ---------------------------------------------------------


# ---- Add titles & inner text boxes -----

# NEW: Added inner text boxes for the left plots
text_bbox = dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')

axes['A'].text(0.05, 0.95, 'Left PSF', transform=axes['A'].transAxes, 
               fontsize=11, color='black', verticalalignment='top', bbox=text_bbox)

axes['B'].text(0.05, 0.95, 'Right PSF', transform=axes['B'].transAxes, 
               fontsize=11, color='black', verticalalignment='top', bbox=text_bbox)

axes['C'].text(0.05, 0.95, 'Combined Science Frame', transform=axes['C'].transAxes, 
               fontsize=11, color='black', verticalalignment='top', bbox=text_bbox)


# --- Delete all ticks and labels ---
for ax in axes.values():
    ax.set_xticks([])  # Removes x-axis ticks and labels
    ax.set_yticks([])  # Removes y-axis ticks and labels
# ----------------------------------------

# Optional: Adjust spacing between plots
# Add padding so tight_layout doesn't override gridspec constraints too aggressively
# plt.tight_layout(w_pad=0.01)

plt.savefig(paths.figures / 'combine_science.pdf', bbox_inches='tight')
