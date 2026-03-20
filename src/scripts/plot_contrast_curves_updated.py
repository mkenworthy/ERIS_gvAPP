debug=0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths

if debug:
    import matplotlib
    matplotlib.use('MacOSX')

# contrast curves by Markus

cc_bracont = pd.read_csv(paths.data/'cc_br-a-cont/Contrast_curves_Br-a-cont_new.csv')
cc_bracont_err = pd.read_csv(paths.data/'cc_br-a-cont/Contrast_curves_Br-a-cont_errors_new.csv')
cc_bra = pd.read_csv(paths.data/'cc_br-a_new/Contrast_curves_Br-a.csv')
cc_bra_err = pd.read_csv(paths.data/'cc_br-a_new/Contrast_curves_Br-a_errors.csv')
cc_brg = pd.read_csv(paths.data/'cc_br-g_new/contrast_curves_Br-g.csv')
cc_brg_err = pd.read_csv(paths.data/'cc_br-g_new/contrast_curves_Br-g_errors.csv')

cc_curves_to_plot = {
    'br-a-cont':[cc_bracont,cc_bracont_err,
        {'color':'tab:orange'},
                 r'Br-$\alpha$-cont',
                 53
                ],
    'br-a':[cc_bra,cc_bra_err,
        {'color':'dodgerblue'},r'Br-$\alpha$',
            15
           ],
    'br-g':[cc_brg,cc_brg_err,
        {'color':'g'},r'Br-$\gamma$',
            48
           ],
}

# Contrast Curves by Pengyu

uncertainty = [0.15, 2.5, 16, 50, 84, 97.5, 99.85]
columns = ['separation [px]'] + list(np.array(uncertainty).astype(str))

trap_kpeak = pd.read_csv(paths.data/'TRAP_reduction_pengyu/bin_m1dark_trap_percentile_uncertainty_values_f03_kpeak.txt',header=None,delimiter=' ',names=columns)
trap_bracont = pd.read_csv(paths.data/'TRAP_reduction_pengyu/trap_percentile_uncertainty_values_f03_Br-a-cont.txt',header=None,delimiter=' ',names=columns)

filter_params = {
    'br-a-cont':[39646.41,1045.74],
    'br-a':[40512.02,253.03],
    'br-g':[21723,220],
    'K-peak':[21970.78,978.21]
}

pixscale=13/1e3
kpeak_lambda_eff = filter_params['K-peak'][0]
kpeak_lambda_eff_width = filter_params['K-peak'][1]
lod_kpeak = kpeak_lambda_eff/1e10/8.2/np.pi*180*60*60

kpeak_tot_time=360*63*2*0.25/60 # 5*60

trap_kpeak['separation [$FWHM$]'] = trap_bracont['separation [px]']*pixscale/lod_kpeak
trap_kpeak['Best'] = trap_kpeak['50.0']
trap_kpeak_err = pd.DataFrame(data=(trap_kpeak['84.0']-trap_kpeak['16.0'])/2,columns=['Best'])

trap_df = pd.DataFrame(data=np.array([[kpeak_tot_time*60/0.25,0.25]]),columns=['ESO DET NDIT','ESO DET SEQ1 DIT'])

cc_curves_to_plot['K-peak'] = [trap_kpeak,trap_kpeak_err,
        {'color':'k'},r'K-peak',189]


#import os
#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
# import os
# from pathlib import Path as Pathxx
# os.environ["PATH"] += os.pathsep + str(Pathxx.home() / "bin")

# #Direct input
# plt.rcParams['text.latex.preamble']=r'\usepackage{lmodern} \usepackage{amsmath}'

#Options
params = {#'text.usetex' : True,
          'font.size' : 8,
          #'font.family' : 'lmodern',
          # 'text.latex.unicode': True,
          }
plt.rcParams.update(params)
plt.rcParams['axes.axisbelow'] = True


fontsize=12
lw=2

def flux_to_mag(flux):
    return -2.5*np.log10(flux)
def mag_to_c(mag):
    return 10**(-mag/2.5)

def c_to_mag(c):
    return -2.5*np.log10(c)

plt.figure(figsize=(4,2.5))
#plt.figure(figsize=(8,5))
for key in ['br-g','br-a-cont','br-a']:
    mask_valid = cc_curves_to_plot[key][0]['Best'] > 1e-6
    plt.fill_between(x=cc_curves_to_plot[key][0]['separation [$FWHM$]'][mask_valid],y1=cc_curves_to_plot[key][0]['Best'][mask_valid]-cc_curves_to_plot[key][1]['Best'][mask_valid],y2=cc_curves_to_plot[key][0]['Best'][mask_valid]+cc_curves_to_plot[key][1]['Best'][mask_valid],**cc_curves_to_plot[key][2],alpha=0.3)
    tot_time = cc_curves_to_plot[key][4]
    label = r'%s @%.2f(%.2f)$\mu$m, $t_{\mathrm{tot}}$=%imin' % (cc_curves_to_plot[key][3],filter_params[key][0]/1e4,filter_params[key][1]/1e4,tot_time)
    plt.plot(cc_curves_to_plot[key][0]['separation [$FWHM$]'][mask_valid],cc_curves_to_plot[key][0]['Best'][mask_valid],**cc_curves_to_plot[key][2],lw=lw,label=label)
# plt.grid()
kpeak_lambda_eff
label = r'%s @%.2f(%.2f)$\mu$m, $t_{\mathrm{tot}}$=%imin' % ('K-peak',kpeak_lambda_eff/1e4,kpeak_lambda_eff_width/1e4,kpeak_tot_time)
plt.plot(trap_kpeak['separation [px]']*pixscale/lod_kpeak,trap_kpeak['50.0'],color='k',label=label)
plt.fill_between(x=trap_kpeak['separation [px]']*pixscale/lod_kpeak,y1=trap_kpeak['16.0'],y2=trap_kpeak['84.0'],alpha=0.3,color='k')

# first y-axis
plt.yscale('log')
ax = plt.gca()
plt.ylabel(r'$5\sigma_{\scriptscriptstyle\mathcal{N}}$ post-proc. contrast')

y_lim_mag = np.array([12.5,2.5])
ax.set_ylim(mag_to_c(y_lim_mag))
# second y-axis
twin_ax =plt.twinx()

twin_ax.set_yticks(minor=False,ticks=np.arange(13))
twin_ax.set_yticks(minor=True,ticks=np.arange(0,13,0.5))
twin_ax.set_ylabel(r'$5\sigma_{\scriptscriptstyle\mathcal{N}}$ post-proc. contrast [$\Delta$mag]')

# x-axis
ax.set_xticks(ticks=np.arange(20))
ax.set_xticks(minor=True,ticks=np.arange(0,20,0.5))
ax.set_xlabel(r'Angular separation [$\lambda$/D]')
ax.set_xlim((0,20))

# both
ax.tick_params(which='both',axis='both')
twin_ax.tick_params(which='both',axis='both')

twin_ax.set_ylim(y_lim_mag)

ax.legend()
plt.tight_layout()
plt.savefig(paths.figures / 'contrast_curves_updated.pdf',dpi=500)
if debug:
      plt.show()