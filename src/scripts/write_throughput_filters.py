# MAK note: this works locally but because of astroquery cacheing weirdness, it doesn't work on showyourwork.

import os

import numpy as np
import pandas as pd
from astroquery.svo_fps import SvoFps
import paths

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

import pickle

list_of_dicts = [filts_tpts,filts_long_tpts,filt_wls,filts_long_wls]

with open(paths.data / 'throughputs.pkl', 'wb') as f:
    pickle.dump(list_of_dicts, f)
print('written throughput filters to throughputs.pkl')

