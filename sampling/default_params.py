#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file lists every scan parameter with its default value.
#

import multiprocessing as mp


default_param_dict = {
                        'analyses' : 'all',
                        'bkg_unc' : None,
                        'bkg_unc_samples' : 500, 
                        'bkg_yields' : None,
                        'buffer_size' : 100,
                        'cluster' : False,
                        'CR_center' : 'obs',
                        'CR_sigma' : 0.02,
                        'fit_bkg' : False,
                        'input_folder' : '../data/',
                        'keep_files' : False,
                        'low_lim_samples' : 30,
                        'output_folder' : '../tables/',
                        'points' : 1000, 
                        'processes' : mp.cpu_count(), 
                        'remove_channels' : [],
                        'removeCRsVRs' : False,
                        'scan_criterion' : 'mu1',
                        'scan_limits' : None,
                        # settings the hardcoded scan_limits were probed with; the
                        # sampler recomputes the limits when this run would exceed them
                        'scan_limits_context' : None,
                        'scans' : 1, 
                        'sig_rel_unc' : 0.0,
                        'signal_leakage_CR' : True,
                        'signal_leakage_CR_sign' : 'both',
                        'signal_leakage_CR_spread' : 0.10,
                        'signal_leakage_VR' : False,
                        'signal_leakage_VR_sign' : 'both',
                        'signal_leakage_VR_spread' : 0.10,
                        'spey_verbose_lvl' : 1,
                        'SR_sigma' : 0.05,
                        'start_method' : 'default',
                        'VR_center' : 'obs',
                        'VR_sigma' : 0.02,
                        }

