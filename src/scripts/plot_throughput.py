# MAK note: this works locally but because of astroquery cacheing weirdness, it doesn't work on showyourwork.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.svo_fps import SvoFps
import paths

params = {#'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          # 'text.latex.unicode': True,
          }
plt.rcParams.update(params)

data_sky = pd.read_csv(paths.data / 'throughput.csv', index_col=0)
data_man = pd.read_csv(paths.data / 'manufacturer_tp.csv', skiprows=1)
data_source = pd.read_csv(paths.data / 'throughput_source.csv', index_col=0, skiprows=0)

filter_properties = SvoFps.get_filter_list(facility='Paranal', instrument='Eris')

filts = ['Paranal/ERIS.H2_cont', 'Paranal/ERIS.H2-1-OS',
       'Paranal/ERIS.Br-g', 'Paranal/ERIS.K_peak',
       'Paranal/ERIS.IB242', 'Paranal/ERIS.IB248',
       'Paranal/ERIS.Br-a-cont', 'Paranal/ERIS.Br-a']

filts_long = ['Paranal/ERIS.K', 'Paranal/ERIS.Lp', 'Paranal/ERIS.Mp']

filt_wls = {}
filts_long_wls = {}

for f_svo in filts:
    filt_wls[f_svo] = [
        float(filter_properties[filter_properties['filterID'] == f_svo]['WavelengthEff'] * 1e-4),
        float(filter_properties[filter_properties['filterID'] == f_svo]['WidthEff'] * 1e-4)
    ]

for f_svo in filts_long:
    filts_long_wls[f_svo] = [
        float(filter_properties[filter_properties['filterID'] == f_svo]['WavelengthEff'] * 1e-4),
        float(filter_properties[filter_properties['filterID'] == f_svo]['WidthEff'] * 1e-4)
    ]

filts_long_tpts = {}
filts_tpts = {}
for f in filts_long:
    mask = np.logical_and(data_man['X'] / 1e3 > filts_long_wls[f][0] - filts_long_wls[f][1] / 2,
                          data_man['X'] / 1e3 < filts_long_wls[f][0] + filts_long_wls[f][1] / 2)
    
    filts_long_tpts[f] = data_man[mask]['Y'].mean()
    
for f in filts:
    mask = np.logical_and(data_man['X'] / 1e3 > filt_wls[f][0] - filt_wls[f][1] / 2,
                          data_man['X'] / 1e3 < filt_wls[f][0] + filt_wls[f][1] / 2)
    
    filts_tpts[f] = data_man[mask]['Y'].mean()


fig, ax = plt.subplots(dpi=200, figsize=(3.55, 2.))

lw=0.75

filt = 'Br-a'
ax.errorbar(x=data_sky['wl'][filt], 
            y=data_sky['tpt'][filt] * 100, 
            xerr=data_sky['wl_err'][filt], 
            yerr=data_sky['tpt_err'][filt] * 100,
            capsize=0,
            c='tab:orange',
            label='On-sky',
            lw=lw)


ax.plot(data_man['X'] / 1e3, 
        data_man['Y'],
        c='k',
        label='Lab',
        lw=lw)

label = 'Internal Source'
for filt in ['Br-g', 'K-peak', 'IB242', 'IB248']:
    ax.errorbar(x=data_source['wl'][filt],
                y=data_source['tpt'][filt] * 100,
                xerr=data_source['wl_err'][filt],
                yerr=data_source['tpt_err'][filt] * 100,
                capsize=0,
                lw=lw,
                label=label,
                c='tab:blue')
    if label is not None:
        label = None

for f in filts:
    ax.axvspan(filt_wls[f][0] - filt_wls[f][1] / 2, 
               filt_wls[f][0] + filt_wls[f][1] / 2, 
               alpha=0.2,
               color='dodgerblue',
               lw=0)
#     
# for f in filts_long:
#     ax.hlines(filts_long_tpts[f],
#                 filts_long_wls[f][0] - filts_long_wls[f][1] / 2,
#                 filts_long_wls[f][0] + filts_long_wls[f][1] / 2,
#                 color='dodgerblue',
#                 lw=0.5)

ax.set_xlabel(r'Wavelength [$\mathrm{\mu m}$]')
ax.set_ylabel(r'Transmission [\%]')

ax.legend()

fig.savefig(paths.figures / 'throughput.pdf', bbox_inches='tight')
