from hcipy import *
from matplotlib import pyplot as plt
import numpy as np 
from matplotlib.gridspec import GridSpec
import paths

def create_image(vAPP,gvAPP,amp):
    fig = plt.figure(figsize = (5*1.2,6*1.2))
    gs = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2:])


    imshow_field(((np.pi + vAPP)%(2*np.pi)-np.pi),cmap = 'RdBu',origin='lower',ax=ax1,mask = amp)
    ax1.set_xlabel('[x/D]')
    ax1.set_ylabel('[y/D]')

    imshow_field(gvAPP%(2*np.pi)-np.pi,cmap = 'RdBu',origin='lower',ax=ax2,mask = amp)
    ax2.set_xlabel('[x/D]')
    ax2.set_yticks([])

    #calculaate PSF
    focal_grid = make_focal_grid(8,60)
    prop = FraunhoferPropagator(grid,focal_grid)
    wf = Wavefront(np.sqrt(amp),input_stokes_vector=(1,0,0,0))
    vAPP_optic = LinearRetarder(np.pi*0.92,gvAPP/2)
    PSF = prop(vAPP_optic.forward(wf)).I

    imshow_field(np.log10(PSF/PSF.max()),vmin = -5,vmax = 0,ax=ax3)
    ax3.set_ylim(-30,30)
    ax3.set_xlabel(r'x [$\lambda/D$]')
    ax3.set_ylabel(r'y [$\lambda/D$]')

    plt.savefig(paths.figures/'image_phase_psf.pdf')

phasename =paths.data/'ERIS_final_gvAPP.fits.gz'
ampname =  paths.data/'ERIS_final_amplitude.fits.gz'

phase = read_fits(phasename)
amp = read_fits(ampname)

Nx = phase.shape[0]

grid = make_pupil_grid(Nx)

angle = np.arctan2(-0.7,-25)

PG = Field(2*np.pi*25*(np.cos(angle)*grid.x+np.sin(angle)*grid.y),grid) % (2*np.pi)

gvAPP = Field(phase.ravel(),grid)
vAPP = gvAPP - PG
amp = Field(amp.ravel(),grid)

create_image(vAPP,gvAPP,amp)

