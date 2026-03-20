debug=0

import os
from cProfile import label

import h5py
import numpy as np
import matplotlib.pyplot as plt
from cmocean.cm import thermal, solar, gray
from astropy.io import fits
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.stats import binned_statistic, median_abs_deviation
import pandas as pd
from photutils.aperture import aperture_photometry, CircularAperture, ApertureStats
from astropy.table import Table
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
thermal.set_bad('black')

import paths

if debug:
    import matplotlib
    matplotlib.use('MacOSX')

# saturated data from HR 8799
data_dir = paths.data / 'bra_cont_throughput/'

sat_sci = fits.open(data_dir/'sat_bra_cont.fits')[0].data
unsat_sci = fits.open(data_dir/'unsat_bra_cont_psf.fits')[0].data

# register the images

def get_max_pos(guess, delta, image):
    im_temp = image[guess[1]-delta: guess[1]+delta, guess[0]-delta:guess[0]+delta]
    m = np.unravel_index(np.argmax(im_temp, axis=None), im_temp.shape)
    return np.array((guess[0], guess[1])) + m - delta

def get_min_pos(guess, delta, image):
    im_temp = image[guess[1]-delta: guess[1]+delta, guess[0]-delta:guess[0]+delta]
    m = np.unravel_index(np.nanargmin(im_temp, axis=None), im_temp.shape)
    return np.array((guess[0], guess[1])) + np.array((m[1], m[0])) - delta

def reg_app(image, 
            guess, 
            saturated,
            plot=False):
    position = {}
    
    if not saturated:
        position['left'] = get_max_pos(guess['left'], guess['delta'], image)
        position['right'] = get_max_pos(guess['right'], guess['delta'], image)
    else:
        position['left'] = get_min_pos(guess['left'], guess['delta'], image)
        position['right'] = get_min_pos(guess['right'], guess['delta'], image)
    position['center'] = get_max_pos(guess['center'], guess['delta'], image)
    
    if plot:
        fig, ax = plt.subplots(ncols=3)
        for i, key in enumerate(position.keys()):
            ax[i].imshow(image, cmap=thermal, origin='lower')
            ax[i].scatter(position[key][0], position[key][1], marker='x', color='w')
            ax[i].set_title(key)
            ax[i].set_xlim(position[key][0]-guess['delta'], position[key][0]+guess['delta'])
            ax[i].set_ylim(position[key][1]-guess['delta'], position[key][1]+guess['delta'])
        plt.show()

    return position

guess_unsat = {
    'left': [157, 420],
    'right': [457, 193],
    'center': [307, 307],
    'delta': 10
}

guess_sat = {
    'left': [157, 420],
    'right': [457, 193],
    'center': [307, 307],
    'delta': 10
}


pos_unsat = reg_app(unsat_sci, guess_unsat, saturated=False)
pos_sat = reg_app(sat_sci, guess_sat, saturated=True)

# Calculate radial cuts with aperture photometry

def radial_cut_app(image,
                   position,
                   lod_px,
                   lod_max,
                   side='left',
                   extra_angle=0,
                   samples=100,
                   plot=True,
                   ):
    ap_pos = []
    cut_angle = (
            np.arctan2(position[side][1]-position['center'][1],
                       position[side][0]-position['center'][0])
            + np.pi/2
            + extra_angle
    )
    px_max = lod_max * lod_px
    px_range = np.linspace(0, px_max, samples, endpoint=True)
    for i in px_range:
        ap_pos.append([position[side][0] + i * np.cos(cut_angle),
                       position[side][1] + i * np.sin(cut_angle)])

    ap_pos = np.array(ap_pos)

    px_length = np.abs(np.sqrt((ap_pos[:,0]-position[side][0])**2 + (ap_pos[:, 1]-position[side][1])**2))

    if plot:
        fig, ax = plt.subplots(dpi=150)
        plt.imshow(np.log10(image), origin='lower', cmap=thermal, vmin=0)
        plt.scatter(position[side][0], position[side][1], marker='x', color='w')
        plt.scatter(np.array(ap_pos)[:,0], np.array(ap_pos)[:,1], marker='.', color='w')

        plt.xlim(position[side][0]-px_max, position[side][0]+px_max)
        plt.ylim(position[side][1]-px_max, position[side][1]+px_max)

        plt.show()

    apertures = CircularAperture(ap_pos, r=lod_px/2)

    phot_table = aperture_photometry(image, apertures)
    sums = np.array(phot_table['aperture_sum'])

    stats = ApertureStats(image, apertures)
    stds = np.array([s.std for s in stats])

    if plot:
        fig, ax = plt.subplots(dpi=150)
        ax.plot(px_range / lod_px, sums, color='k')
        ax.fill_between(px_range / lod_px, sums-stds, sums+stds, alpha=0.5, color='k')
        ax.set_yscale('log')
        plt.show()

    return px_length, sums, stats, np.array(ap_pos).T, cut_angle

def get_norm_app(image, position,  lod_px, side='left'):
    apertures = CircularAperture(position[side], r=lod_px/2)
    phot_table = aperture_photometry(image, apertures)
    return phot_table['aperture_sum'][0]


lod_px = 3.96e-6 / 8.2 * 180 * 3600 / np.pi / 0.013

unsat_px, unsat_sum, unsat_std, ap_pos_unsat, ca_unsat = radial_cut_app(image=unsat_sci,
                                                            position=pos_unsat,
                                                            lod_px=lod_px,
                                                            lod_max=20,
                                                            extra_angle=0,
                                                            side='left',
                                                            plot=False)

sat_px, sat_sum, sat_std, ap_pos_sat, ca_sat = radial_cut_app(image=sat_sci,
                                                position=pos_sat,
                                                lod_px=lod_px,
                                                lod_max=20,
                                                extra_angle=0,
                                                side='left',
                                                plot=False)

norm = get_norm_app(unsat_sci, pos_unsat, lod_px)

def get_scaling_factor(sat_sum, unsat_sum, sat_px, unsat_px, lod_px, lod_min=1.5, lod_max=2.):
    mask_sat = np.logical_and(sat_px / lod_px >= lod_min, sat_px / lod_px <= lod_max)
    mask_unsat = np.logical_and(unsat_px / lod_px >= lod_min, unsat_px / lod_px <= lod_max)
    scaling_factor = np.mean(unsat_sum[mask_unsat] / sat_sum[mask_sat])

    return scaling_factor

scaling_factor = get_scaling_factor(sat_sum=sat_sum,
                                    unsat_sum=unsat_sum,
                                    sat_px=sat_px,
                                    unsat_px=unsat_px,
                                    lod_px=lod_px)

norm_sat = norm / scaling_factor

# get radial uncertainties

angles = np.linspace(-81, 81, 400, endpoint=True)

sat_sums = []

for a in angles:
    _, us, _, _, _ = radial_cut_app(image=sat_sci,
                              position=pos_sat,
                              lod_px=lod_px,
                              lod_max=20,
                              extra_angle=a / 180 * np.pi,
                              side='left',
                              plot=False)
    sat_sums.append(us)

unsat_sums = []

for a in angles:
    _, us, _, _, _ = radial_cut_app(image=unsat_sci,
                              position=pos_unsat,
                              lod_px=lod_px,
                              lod_max=20,
                              extra_angle=a / 180 * np.pi,
                              side='left',
                              plot=False)
    unsat_sums.append(us)


# Final Plot

lab_data = pd.read_csv(paths.data / 'lab_performance.csv',
                      header=0)

#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
import os
from pathlib import Path as Pathxx
os.environ["PATH"] += os.pathsep + str(Pathxx.home() / "bin")

#Direct input
plt.rcParams['text.latex.preamble']=r'\usepackage{lmodern} \usepackage{amsmath}'

#Options
params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
 #         'text.latex.unicode': True
          }
plt.rcParams.update(params)


desing_points = {'IWA': [2.2, 1e-4],
                 'OWA': [15, 1e-5],}

# Define the vertices of the upside-down 'T'
verts = [
    (-0.5, 0),  # Left end of the horizontal line
    (0.5, 0),   # Right end of the horizontal line
    (0, 0),     # Center point of the horizontal line
    (0, 1),    # Bottom end of the vertical line
]

# Define the path codes for the vertices
codes = [
    Path.MOVETO,  # Move to the start of the horizontal line
    Path.LINETO,  # Draw line to the end of the horizontal line
    Path.MOVETO,  # Move to the center of the horizontal line
    Path.LINETO,  # Draw line to the bottom of the vertical line
]

# Create the Path object for the upside-down 'T'
upside_down_T = Path(verts, codes)

fig, ax = plt.subplots(ncols=2, dpi=200, figsize=(7.2, 2.2), gridspec_kw={'width_ratios': [1, 0.7]})

# --- THROUGHPUT PLOT ---

# select 4 colors from thermal colormap
colors = gray(np.linspace(0.2, 0.8, 4))

ang_cuts = np.array([10, 35, 60, 82])
cut_btw = 1.3
cut_out = 17.

mask_sat = np.logical_and(sat_px / lod_px > cut_btw, sat_px / lod_px <= cut_out)
mask_unsat = unsat_px / lod_px <= cut_btw + 0.2

vals = {key: [] for key in ang_cuts}
vals_unsat = {key: [] for key in ang_cuts}
max_vals = {}
min_vals = {}
max_vals_unsat = {}
min_vals_unsat = {}
std_vals = {}

for a, us in zip(angles, sat_sums):
    c = np.where(np.abs(a) < ang_cuts)[0][0]
    vals[ang_cuts[c]].append(us)

for a, us in zip(angles, unsat_sums):
    c = np.where(np.abs(a) < ang_cuts)[0][0]
    vals_unsat[ang_cuts[c]].append(us)

sat_mean = np.median(vals[ang_cuts[0]], axis=0)

zorder = 10
for ac in ang_cuts:
    vals[ac] = np.array(vals[ac])
    max_vals[ac] = np.max(np.abs(vals[ac]), axis=0)
    # max_vals[ac] = np.percentile(vals[ac], 66, axis=0)
    min_vals[ac] = np.min(np.abs(vals[ac]), axis=0)
    # min_vals[ac] = np.percentile(np.abs(vals[ac]), 5, axis=0)
    std_vals[ac] = median_abs_deviation(vals[ac], axis=0)

    vals_unsat[ac] = np.array(vals_unsat[ac])
    max_vals_unsat[ac] = np.max(np.abs(vals_unsat[ac]), axis=0)
    min_vals_unsat[ac] = np.min(np.abs(vals_unsat[ac]), axis=0)

    zorder -= 1

xcut_lim = -1
ax[0].fill_between(sat_px[mask_sat] / lod_px,
                # min_vals[ac][mask_sat] / norm_sat,
                   (sat_mean[mask_sat] - std_vals[10][mask_sat]) / norm_sat,
                (sat_mean[mask_sat] + std_vals[10][mask_sat]) / norm_sat,
                alpha=0.3,
                lw=0,
                color='tab:orange',
                zorder=zorder)

ax[0].plot(sat_px[mask_sat] / lod_px, sat_mean[mask_sat] / norm_sat, color='tab:orange', zorder=12, label=r'On-sky measurement (4 $\mu$m)')
ax[0].plot(unsat_px[mask_unsat] / lod_px, unsat_sum[mask_unsat] / norm, color='tab:orange', zorder=12)

ax[0].plot(lab_data['measured_x'], lab_data['measured_y'], c='dodgerblue',
    label='Lab measurement (2 $\\mu$m)', zorder=13)

ax[0].legend(loc='upper right', fontsize=8)
ax[0].set_yscale('log')
ax[0].set_ylim(3e-6, 2)
ax[0].set_xlim(ax[0].get_xlim()[0], 18)

ax[0].axvline(cut_btw, color='k', ls='--', zorder=15, c='gray')
# at bottom of image, next to line add arrow pointing to the right, no text
ax[0].annotate('', xy=(cut_btw-0.2, 1e-5), xytext=(cut_btw-2, 1e-5), arrowprops=dict(facecolor='gray', arrowstyle='<-', zorder=15, edgecolor='gray'), zorder=15)
ax[0].annotate('', xy=(cut_btw+0.2, 1e-5), xytext=(cut_btw+2, 1e-5), arrowprops=dict(facecolor='gray', arrowstyle='<-', zorder=15, edgecolor='gray'), zorder=15)

ax[0].text(cut_btw-1.3, 1.5e-5, 'PSF\nreference', color='gray', fontsize=7, zorder=15, ha='left', va='bottom', rotation=90)
ax[0].text(cut_btw+0.2, 1.5e-5, 'Science\nintegration', color='gray', fontsize=7, zorder=15, ha='left', va='bottom', rotation=0)

ax[0].scatter(desing_points['IWA'][0], desing_points['IWA'][1], marker=upside_down_T, color='tab:orange', s=100, zorder=16)
ax[0].text(desing_points['IWA'][0]+0.2, desing_points['IWA'][1], 'IWA', color='tab:orange', fontsize=6, ha='left', va='bottom', zorder=16)

ax[0].scatter(desing_points['OWA'][0], desing_points['OWA'][1], marker=upside_down_T, color='tab:orange', s=100, zorder=16)
ax[0].text(desing_points['OWA'][0]+0.2, desing_points['OWA'][1], 'OWA', color='tab:orange', fontsize=6, ha='left', va='bottom', zorder=16)


# --- APP IMAGE ---
size = pos_sat['left'][0]

# cut image
sat_sci_cut = sat_sci[pos_sat['left'][1]-size:pos_sat['left'][1]+size,
                  pos_sat['left'][0]-size:pos_sat['left'][0]+size]

img = ax[1].imshow(np.log10(np.abs(sat_sci / norm_sat)), cmap=thermal, origin='lower', vmin=np.log10(3e-6), vmax=0)

# add colorbar to right of image
cbar = plt.colorbar(img, ax=ax[1], fraction=0.046, pad=0.25, label=r'$\mathrm{log_{10}}$(Raw contrast)')


# rotate the above line by 10 deg
rot = np.array((ang_cuts[0], -ang_cuts[0])) / 180 * np.pi
rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

ap_pos_sat_rot = np.dot(rot_mat[:, :, 0], ap_pos_sat - np.array([pos_sat['left'][0], pos_sat['left'][1]]).reshape(2, 1)) + np.array([pos_sat['left'][0], pos_sat['left'][1]]).reshape(2, 1)
ax[1].plot([ap_pos_sat_rot[0, 0], ap_pos_sat_rot[0, -1]],
              [ap_pos_sat_rot[1, 0], ap_pos_sat_rot[1, -1]],
              color='tab:orange', lw=1, solid_capstyle='round')

ap_pos_sat_rot = np.dot(rot_mat[:, :, 1], ap_pos_sat - np.array([pos_sat['left'][0], pos_sat['left'][1]]).reshape(2, 1)) + np.array([pos_sat['left'][0], pos_sat['left'][1]]).reshape(2, 1)
ax[1].plot([ap_pos_sat_rot[0, 0], ap_pos_sat_rot[0, -1]],
              [ap_pos_sat_rot[1, 0], ap_pos_sat_rot[1, -1]],
              color='tab:orange', lw=1, solid_capstyle='round')
# ax[1].scatter(ap_pos_sat[0], ap_pos_sat[1], marker='.', color='w', s=1)

# plot wedge from position_sat['left']
patches = []
ac_prev = 82

ac_prev = 0
ac = 10
wedge = Wedge(pos_sat['left'], sat_px.max(), ca_sat*180/np.pi-ac, ca_sat*180/np.pi-ac_prev, ec='none', facecolor='gray', alpha=0.5)
ax[1].add_patch(wedge)
wedge = Wedge(pos_sat['left'], sat_px.max(), ca_sat*180/np.pi+ac_prev, ca_sat*180/np.pi+ac, ec='none', facecolor='gray', alpha=0.5)
ax[1].add_patch(wedge)
ac_prev = ac

ax[1].set_xlim(pos_sat['left'][0]-size, pos_sat['left'][0]+size)
ax[1].set_ylim(pos_sat['left'][1]-size, pos_sat['left'][1]+size)

lod_mrk = 15

ax[1].set_xticks([pos_sat['left'][0] - lod_mrk * lod_px, pos_sat['left'][0], pos_sat['left'][0] + lod_mrk * lod_px])
ax[1].set_xticklabels([-lod_mrk, 0, lod_mrk])
ax[1].set_yticks([pos_sat['left'][1] - lod_mrk * lod_px, pos_sat['left'][1], pos_sat['left'][1] + lod_mrk * lod_px])
ax[1].set_yticklabels([-lod_mrk, 0, lod_mrk])

# ticks and label on right
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")

ax[1].set_ylabel(r'$\Delta$DEC [$\lambda/D$]')
ax[1].set_xlabel(r'$\Delta$RA [$\lambda/D$]')


ax[0].set_xlabel(r'Angular separation [$\lambda/D$]')
ax[0].set_ylabel('Raw contrast')

plt.tight_layout()
plt.savefig(paths.figures / 'raw_contrast_bracont_v2.pdf')

if debug:
    plt.show()

