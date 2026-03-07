from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import paths


pupil_grid = make_pupil_grid(2448)
focal_grid = make_focal_grid(4, 60)

prop = FraunhoferPropagator(pupil_grid, focal_grid)

vAPP_phase = Field(read_fits(paths.data / 'ERIS_final_gvAPP.fits.gz').ravel(), pupil_grid)
telescope_pupil = Field(read_fits(paths.data / 'ERIS_final_amplitude.fits.gz').ravel(), pupil_grid)

# Defining the properties of the vAPP
fast_axis_orientation = vAPP_phase / 2
phase_retardation = np.pi
circularity = 0

# Making the vAPP
vAPP_pol = PhaseRetarder(phase_retardation, fast_axis_orientation, circularity)

angle = np.arctan2(-0.7,-25)

grid = make_pupil_grid(2448)
PG = Field(2*np.pi*25*(np.cos(angle)*grid.x+np.sin(angle)*grid.y),grid) %  (2*np.pi)

#retardation = np.linspace(0, 2 * np.pi, 37, endpoint=True)
wavefront = Wavefront(telescope_pupil, 1, input_stokes_vector=(1, 0, 0, 0))
vAPP_pol = PhaseRetarder(170*np.pi/180., fast_axis_orientation, circularity)

wavefront_out = vAPP_pol.forward(wavefront)

vAPP_PSF = prop(wavefront_out).intensity

gs_kw = dict(width_ratios=[1.8, 1])
fig, axd = plt.subplots(1,2, figsize=(9, 2.5),
                        width_ratios=[1.0,2.5],
                              layout="constrained")

cmaps = ['RdBu_r', 'viridis']

# remove the grating
fast_axis_orientation = np.angle((np.exp(1j*(vAPP_phase-PG))))

pcm1 = imshow_field(fast_axis_orientation, mask=telescope_pupil, cmap='twilight', ax=axd[0])

pcm2 = imshow_field(np.log10(vAPP_PSF / vAPP_PSF.max()), vmin=-5, ax=axd[1])

import matplotlib.ticker as mticker

cbar1 = fig.colorbar(pcm1, ax=axd[0],
                     label='phase',
                    ticks=[-3.141,-3.141/2,0,3.141/2,3.141],
                    format=mticker.FixedFormatter(["$-\\pi$","$-\\pi/2$","0","$\\pi/2$","$\\pi$"]),
                     location='left'
                    )
cbar2 = fig.colorbar(pcm2, ax=axd[1], label='Normalised intensity',
                    ticks=[0,-1,-2,-3,-4],
                    format=mticker.FixedFormatter(["$10^{0}$","$10^{-1}$","$10^{-2}$","$10^{-3}$","$10^{-4}$"])
                    )

ff=12

axd[0].set_yticks([-0.5,0,0.5])
axd[0].set_xticks([-0.5,0,0.5])

axd[1].set_ylim(-30,30)
axd[1].set_xlim(-60,60)

td = dict(fontsize=10,
          color='white')
cc = dict(ha='center',
         va='center')

axd[1].annotate('Coronagraphic PSF', xy=(-25,0), xycoords='data',
            xytext=(-55,-20),
            va='top', ha='left',
            arrowprops=dict(facecolor='white', shrink=0.1), **td)


axd[1].annotate('Coronagraphic PSF', xy=(25,0), xycoords='data',
            xytext=(55,20),
            va='bottom', ha='right',
            arrowprops=dict(facecolor='white', shrink=0.1), **td)


axd[1].annotate('Astrometric spots', xy=(0,25), xycoords='data',
            xytext=(10,-20),
            va='bottom', ha='left',
            arrowprops=dict(facecolor='white', shrink=0.1), **td)


axd[1].annotate('Astrometric spots', xy=(0,0), xycoords='data',
            xytext=(10,-20),
            va='bottom', ha='left',
            arrowprops=dict(facecolor='white', shrink=0.1), **td)


axd[1].annotate('Astrometric spots', xy=(0,-25), xycoords='data',
            xytext=(10,-20),
            va='bottom', ha='left',
            arrowprops=dict(facecolor='white', shrink=0.1), **td)


axd[1].annotate('Leakage PSF', xy=(0,0), xycoords='data',
            xytext=(-10,20),
            va='bottom', ha='right',
            arrowprops=dict(facecolor='white', shrink=0.1), **td)


axd[1].text(-25,8,"Dark hole", **td, **cc)
axd[1].text( 25,-8,"Dark hole", **td, **cc)

axd[0].set_xlabel("$[x/D]$",fontsize=ff)
axd[0].set_ylabel("$[y/D]$",fontsize=ff)
axd[1].set_xlabel("$[\\lambda/D]$",fontsize=ff)
axd[1].set_ylabel("$[\\lambda/D]$",fontsize=ff)

plt.savefig(paths.figures / 'image_phase_psf.pdf')



