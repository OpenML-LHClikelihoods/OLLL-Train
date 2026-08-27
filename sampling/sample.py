#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file is used to generate likelihood samples.
#

import os, sys
if 'NUMEXPR_MAX_THREADS' not in os.environ:
    os.environ['NUMEXPR_MAX_THREADS'] = '8' # silence numpy warning
import warnings
import yaml
from os.path import join, dirname, isfile, basename
import spey
import numpy as np
import time
import multiprocessing as mp
import json
from utils import *
from likelihood import *
from datetime import datetime, timedelta
import time
import copy
import random
from tqdm_loggable.auto import tqdm
from collections import OrderedDict
from misc import *
import tarfile
import gc
import jax
import argparse
from scipy.linalg import LinAlgError
from default_params import default_param_dict
jax.config.update('jax_platforms', 'cpu')
silence_spey_banner()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# YAML parsing
#
def include_handler(data):
    """Resolve an ``include`` key within a parsed YAML document.

    If ``data`` is a dict containing an ``"include"`` key, the file it
    points to is loaded as YAML and merged into ``data`` (with the
    ``"include"`` key itself removed). Non-dict inputs are returned
    unchanged.

    Args:
        data: A single parsed YAML document (typically a ``dict``), as
            produced by ``yaml.load``/``yaml.full_load_all``.

    Returns:
        The input ``data``, with any ``include`` directive resolved and
        merged in (if applicable). Non-dict inputs are returned as-is.

    Raises:
        FileNotFoundError: If the file referenced by ``include`` does not exist.
        ValueError: If the included file does not contain a dictionary.
    """
    if not isinstance(data, dict):
        return data

    # Check if the "include" key exists
    if "include" in data:
        file_path = data["include"]
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Included file {file_path} not found.")
        with open(file_path, 'r') as f:
            included_data = yaml.load(f, Loader=yaml.FullLoader)  # Using FullLoader directly
            if not isinstance(included_data, dict):
                raise ValueError(f"Included file {file_path} must contain a dictionary.")
            del data["include"]  # Remove the `include` key
            # The card is merged ON TOP, so it wins wherever both set a key.
            # That is deliberate - a card describes the analysis and should not be
            # silently contradicted - but it means a value written in the
            # parameter document simply has no effect, with nothing to show for
            # it. sig_rel_unc is the usual casualty. Report the collisions.
            # capture the document's own values BEFORE the merge overwrites them
            overridden = {k: data[k] for k, v in included_data.items()
                          if k in data and data[k] != v}
            data.update(included_data)  # Merge included data into the main dictionary
            if overridden:
                label = data.get('analysis') or data.get('name') or file_path
                card = os.path.basename(file_path)
                # the include is resolved before main() binds the module logger,
                # so fall back to printing when it is not there yet
                say = globals().get('logger')
                say = say.warning if say is not None else (lambda m: print('[WARNING]', m))
                for k in sorted(overridden):
                    say(f"[{label}] '{k}' is set in the parameter file, but {card} sets it "
                        f"too and the card wins: {k}={included_data[k]!r} is used, not "
                        f"{overridden[k]!r}. Change it in the card, or include one that "
                        f"already has the value you want.")
    return data

def load_yaml_with_includes(file_path):
    """Load all YAML documents from a file, resolving ``include`` directives.

    Args:
        file_path (str): Path to the YAML file to load. The file may
            contain multiple ``---``-separated documents.

    Returns:
        list: The list of parsed YAML documents, each passed through
        :func:`include_handler` to resolve any ``include`` key.
    """
    with open(file_path, 'r') as f:
        documents = yaml.full_load_all(f)  # Using FullLoader directly
        return [include_handler(doc) for doc in documents]

#
# MAIN
#
def main(logger, param_file, starting_points_file, starting_points_file_index):
    """Drive the full likelihood-sampling pipeline for one or more analyses.

    Loads and merges the YAML parameter file(s) (including any
    ``include`` directives) into a global parameter dictionary, then for
    each requested analysis: unpacks input archives if needed, loads the
    background workspace and signal patchset, determines background
    yields/uncertainties (either from the parameter file or by fitting),
    determines observed yields, computes scan limits and the minimal
    allowed (most negative) signal per bin, generates MCMC starting
    points, and finally runs the parallel MCMC scan across signal
    patches via :class:`likelihood.ScanWrapper`. Results are merged,
    written to CSV, and accompanying metadata is saved as JSON in the
    output directory for each analysis.

    Args:
        logger (logging.Logger): Logger instance; note that this
            argument is immediately overwritten inside the function by
            ``logging.getLogger("main_logger")``, so the caller's logger
            configuration is only used indirectly (whatever handlers are
            attached to the ``"main_logger"`` logger by the caller).
        param_file (str): Path to the YAML parameter file describing the
            analyses to sample and their configuration.
        starting_points_file (str or None): Optional path to a CSV file
            with pre-computed MCMC starting points to use instead of
            generating new ones.
        starting_points_file_index (int or None): Optional row index to
            select from ``starting_points_file`` when it contains
            multiple candidate starting points.

    Raises:
        ValueError: On malformed or missing YAML parameter documents, or
            invalid parameter values (e.g. mismatched channel/bin counts,
            non-positive scan/point/process counts, negative signal
            relative uncertainty).
        FileNotFoundError: If ``param_file`` is missing, or if an
            ``include``-referenced YAML file cannot be found.
        RuntimeError: If the parameter file cannot be parsed at all.
        PermissionError: If the output directory for a given analysis
            cannot be created due to insufficient permissions.
    """
    logger = logging.getLogger("main_logger")
    yaml_docs = None
    param_docs = []
    # default values of parameters
    # deepcopy: otherwise every assignment below would mutate the module-level
    # default_param_dict and leak into any later call of main()
    global_param_dict = copy.deepcopy(default_param_dict)

    if os.path.isfile(param_file):
        yaml_docs = load_yaml_with_includes(param_file)

        # consitency checks
        if len(yaml_docs) == 0:
            mes = 'No yaml documents found inside the param file!'
            logger.critical(mes)
            raise ValueError(mes)
        elif len(yaml_docs) == 1:
            if yaml_docs[0].get('analyses') is not None:
                mes = 'Parameter should contain the list of analyses to read and subsequently their parameters in seperate yaml documents!'
                logger.critical(mes)
                raise ValueError(mes)
            else:
                # there is only one analysis, parse it
                logger.warning('Parameter file seems to contain a single analysis. Loading..')
                name = yaml_docs[0].get('name')
                global_param_dict['analyses'] = [name]
                param_docs = yaml_docs
        else:
            if yaml_docs[0].get('analyses') is None:
                logger.warning('List of analyses not provided! I will scan all available.')
                global_param_dict['analyses'] = 'all'
            else:
                for k,v in yaml_docs[0].items():
                    global_param_dict[k] = v
            # the first document always holds the global settings; the remaining
            # ones describe the individual analyses
            param_docs = yaml_docs[1:]
    else:
        mes = 'Parameter file "parameters.yaml" is missing!'
        logger.critical(mes)
        raise FileNotFoundError(mes)
    if yaml_docs is None:
        mes = 'Unable to parse parameter file. Aborting.'
        logger.critical(mes)
        raise RuntimeError(mes)

    del yaml_docs

    # Warn about keys the code never reads. Silent typos (e.g. "SR_sigma'" or
    # "scan") used to make settings vanish without a trace.
    check_unknown_parameters([global_param_dict] + param_docs, logger)

    spey.set_log_level(global_param_dict['spey_verbose_lvl'])
    if global_param_dict['analyses'] == 'all':
        logger.info(f'Param file loaded. All {len(param_docs)} analyses in the file will be sampled.')
    else:
        logger.info(f'Param file loaded. {len(global_param_dict["analyses"])} analyses to be sampled.')
        if len(list(set(global_param_dict['analyses']))) > len(param_docs):
            logger.warning(f'The list of analyses to sample is longer than the list of parameter documents!')


    #############################################
    # create a directory for likelihood tables
    #############################################
    # honour output_folder instead of always creating ../tables: pointing the
    # parameter file somewhere else used to still create an empty ../tables
    tables_dir = global_param_dict['output_folder']
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir, exist_ok=True)
        logger.info('Created parent directory for likelihood tables.')
    else:
        logger.info('Parent directory for likelihood tables already exists.')

    #########################
    # loop over the analyses
    #########################
    global_param_dict['merged'] = False # add metadata with info about the post-sampling merging of results
    for analysis_index in range(len(param_docs)):
        analysis = param_docs[analysis_index]
        param_dict = copy.deepcopy(global_param_dict)
        for k,v in analysis.items():
            param_dict[k] = v
        name = param_dict['analysis']
        if global_param_dict['analyses']=='all' or name in global_param_dict['analyses']:
            ###############################
            # sample the selected analysis
            ###############################
            # wall-clock, not process CPU time: the scan runs in a multiprocessing
            # pool, so time.process_time() only sees this process and badly
            # under-reports both the total and the seconds-per-point figure.
            analysis_time = time.perf_counter_ns()
            logger.info(f'Working on analysis {name}.')
            # check for the input folder and unpack the archive if needed
            input_folder = join(param_dict['input_folder'], name)
            if not os.path.isdir(input_folder):
                archive_path = join(param_dict['input_folder'], name+'.tar.gz')
                if os.path.exists(archive_path):
                    logger.info(f'Unpacking {name}.tar.gz')
                    try:
                        with tarfile.open(archive_path, 'r:gz') as tar:
                            tar.extractall(path=param_dict['input_folder'])
                    except Exception as e:
                        logger.error("Couldn't unpack archive with likelihood model! Message: " +str(e))
                        continue
                else:
                    logger.error(f'Input files not found for {name}.')
                    continue

            # set the random seed
            if 'seed' not in param_dict.keys():
                param_dict['seed'] = int(time.time()) % (2**32-1)

            # set up slurm computations 
            if param_dict['cluster']:
                logger.info('Running in the cluster mode!')
                slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
                slurm_task_id = int(slurm_task_id) if slurm_task_id and slurm_task_id.isdigit() else 0
                param_dict['seed'] = slurm_task_id + param_dict['seed']

            logger.info(f'Using random seed value: {param_dict["seed"]}')
            set_global_determinism(seed=param_dict['seed'])

            try:
                dir_index = 0
                while True:
                    if not param_dict['cluster']:
                        dirpath = join(param_dict['output_folder'], "{}-{}".format(name, dir_index) )
                    else:
                        dirpath = join(param_dict['output_folder'], "{}-{}-{}".format(name, dir_index, slurm_task_id) )

                    if not os.path.isdir(dirpath):
                        break
                    else:
                        dir_index += 1
                os.makedirs(dirpath)
                logger.info('Created directory for likelihood tables.')
            except PermissionError:
                # cannot create directory
                mes = 'Cannot create output directory. Permission denied.'
                logger.critical(mes)
                raise PermissionError(mes)

            bkgfiles = param_dict['bkgfiles']
            if bkgfiles is None:
                mes = 'Background files missing in the parameter file!'
                logger.critical(mes)
                raise ValueError(mes)                
            
            ####################
            # Load the patches
            ####################
            patchsets = param_dict['patchsets']
            if patchsets is None:
                mes = 'Provide at least one patchset in the parameter file!'
                logger.critical(mes)
                raise ValueError(mes)
            
            ######################################################################
            # Special case when one background file corresponds to many patchsets
            ######################################################################
            if len(patchsets) > 1 and len(bkgfiles) == 1:
                # we simply emulate provision of multiple background files
                bkgfiles = [ bkgfiles[0] ] * len(patchsets)
                # works for 1-1 and 1-many
                param_dict['bkg_yields'] = [param_dict['bkg_yields']] * len(patchsets)
                param_dict['bkg_unc'] = [param_dict['bkg_unc']] * len(patchsets)
            elif len(patchsets) == 1 and len(bkgfiles) == 1:
                 param_dict['bkg_yields'] = [ param_dict['bkg_yields'] ]
                 param_dict['bkg_unc'] = [ param_dict['bkg_unc'] ]
            # Check if each patchset is matched with background file
            if len(patchsets) != len(bkgfiles):
                    mes = f'Number of patchsets ({len(patchsets)}) does not match the number of background files ({len(bkgfiles)})!'
                    logger.critical(mes)
                    raise ValueError(mes)
            
            for file_pair_index, (bkgfile, patchset_var) in enumerate(zip(bkgfiles, patchsets)):
                if isinstance(patchset_var, str):
                    patchset_path = patchset_var
                elif isinstance(patchset_var, list):
                    patchset_path = patchset_var[0]
                    if not patchset_var[1]:
                        mes = f'Skipping {patchset_path}'
                        logger.warning(mes)
                        continue
                else:
                    mes = f'Patchset entries should be strings or lists, but {type(patchset_var)} provided!'
                    logger.critical(mes)
                    raise TypeError(mes)


                # restart the clock for every (background, patchset) pair, so the
                # timing reported at the end of the pair covers that pair only
                analysis_time = time.perf_counter_ns()
                logger.info(f"Analysing {basename(bkgfile)} and {basename(patchset_path)} files [{file_pair_index+1}/{len(bkgfiles)}].")
                suffix = basename(bkgfile).split('.')[0]+'-'+basename(patchset_path).split('.')[0]

                # load the channels for given pair of files
                if len(param_dict['bkgfiles']) == 1:
                    if len(param_dict['channels']) > 1:
                        logger.warning(f'There are more groups of channels ({len(param_dict["channels"])}) specified than input background files. I will ignore additional channel groups.')
                        channels = OrderedDict(param_dict['channels'][file_pair_index])
                    else:
                        channels = OrderedDict(param_dict['channels'][0])
                else:
                    channels = OrderedDict(param_dict['channels'][file_pair_index])
                SRs = [k for k,v in channels.items() if v=='SR']


                with open(join(param_dict['input_folder'], name, patchset_path), 'r') as fpatch:
                    patch_json = json.load(fpatch)
                patchset_spec = pyhf.PatchSet(patch_json)
                logger.info(f'{len(patchset_spec.patches)} patches in the analysis.')
                logger.info(f'{len(channels.keys())} channels found with {len(SRs)} signal regions: \n\t\t\t'+'\n\t\t\t'.join(SRs))
               
                ##############################
                # determine background values
                ##############################
                logger.info(f"Getting the background yields.")
                with open(join(param_dict['input_folder'], name, bkgfile)) as serialized:
                    bkg_spec = json.load(serialized)
                # bkgonly = pyhf.Workspace(bkg_spec)
                stat_wrapper = spey.get_backend("pyhf")
                signal_patch = patch_json["patches"][0]["patch"]
                interpreter = WorkspaceInterpreter(bkg_spec)
                signal_map, modifiers = interpreter.patch_to_map(signal_patch=signal_patch)
                for key, item in signal_map.items():
                    interpreter.inject_signal(key, item)
                stat_wrapper = spey.get_backend("pyhf")
                full_background_model = stat_wrapper(
                                                    analysis=name,
                                                    background_only_model=bkg_spec,
                                                    signal_patch=interpreter.make_patch(),
                                                )
                _, model_bkg, data_bkg = full_background_model.backend.model(expected=spey.ExpectationType.apriori)
                
                # extract channels
                if set(model_bkg.config.channel_nbins.keys()) != set(channels.keys()):
                    mes = 'Channels found in the background file does not match channels provided in the parameter file!'
                    logger.critical(mes)
                    raise ValueError(mes)

                bins_names = []
                bins_is_signal = []
                channels_and_bins = []
                for k,v in model_bkg.config.channel_nbins.items():
                    channels_and_bins.append((k, channels[k], v))
                    for bin_index in range(v):
                        bins_names.append(k+'-{}'.format(bin_index))
                        bins_is_signal.append(k in SRs)
                bins_is_signal = np.array(bins_is_signal)
                bkg_yields = list(zip(bins_names, data_bkg[:len(bins_is_signal)]))

                # Store channel_nbins before deleting model_bkg - needed for robust mapping
                channel_nbins = OrderedDict(model_bkg.config.channel_nbins)

                del data_bkg
                del model_bkg

                # Create bin names in the YAML file's channel ordering. This is needed by
                # BOTH the 'read from file' and the 'fit' branches below (the printed tables
                # and the observed yields follow the user's ordering), so it lives here.
                input_bins_ordered = []
                bins_is_signal_ordered = []
                for ch_name in channels.keys():
                    n_bins = channel_nbins.get(ch_name)
                    if n_bins is None:
                        logger.warning(f"Channel {ch_name} from input file not found in model")
                        continue
                    for bin_index in range(n_bins):
                        input_bins_ordered.append(f'{ch_name}-{bin_index}')
                        bins_is_signal_ordered.append(ch_name in SRs)

                if not param_dict['fit_bkg']:
                    logger.info(f"Loading background yields and uncertainties from the file.")
                    file_data_bkg = param_dict['bkg_yields'][file_pair_index]
                    file_data_bkg_unc = param_dict['bkg_unc'][file_pair_index]

                    if file_data_bkg is None or file_data_bkg_unc is None:
                        mes = "'fit_bkg' is disabled but 'bkg_yields'/'bkg_unc' are missing from the parameter file! " \
                              "Provide them in the analysis card or set 'fit_bkg : True'."
                        logger.critical(mes)
                        raise ValueError(mes)

                    # Validate lengths
                    if len(input_bins_ordered) != len(file_data_bkg) or len(input_bins_ordered) != len(file_data_bkg_unc):
                        mes = f'There is a mismatch between the number of recognised bins ({len(input_bins_ordered)}), ' \
                              f'the number of bkg yields ({len(file_data_bkg)}), and the number of bkg uncertainties ({len(file_data_bkg_unc)}). Please proceed carefully.'
                        logger.error(mes)
                        raise ValueError(mes)
                    
                    # Create lookup dictionaries mapping bin name -> value
                    bkg_yield_dict = OrderedDict(zip(input_bins_ordered, file_data_bkg))
                    bkg_unc_dict = OrderedDict(zip(input_bins_ordered, file_data_bkg_unc))
                    
                    # Assign values in MODEL's bin ordering (bins_names)
                    bkg_yields = []
                    bkg_unc = []
                    for bin_name in bins_names:
                        if bin_name not in bkg_yield_dict:
                            mes = f'Bin {bin_name} from model not found in input file data'
                            logger.error(mes)
                            raise ValueError(mes)
                        bkg_yields.append((bin_name, bkg_yield_dict[bin_name]))
                        bkg_unc.append((bin_name, bkg_unc_dict[bin_name]))
                    # There might be a difference between user defined order of data and model's
                    bkg_yields_ordered = []
                    bkg_unc_ordered = []
                    for bin_name in input_bins_ordered:
                        bkg_yields_ordered.append((bin_name, bkg_yield_dict[bin_name]))
                        bkg_unc_ordered.append((bin_name, bkg_unc_dict[bin_name]))
                    logger.info(f"Successfully mapped {len(bkg_yields)} background yields and uncertainties")
                else:
                    ########################################
                    # determine background uncertainties
                    ########################################
                    full_background_model.backend.manager.backend = "jax"
                    # need to convert full background model to simplified in order to get systematic uncertainty on B
                    converter = spey.get_backend("pyhf.simplify")
                    logger.info(f"Estimating background uncertainty ...")

                    n_bkg_unc = param_dict['bkg_unc_samples']
                    # spey renamed the simplified-pdf keys after 0.2.5
                    # ('default_pdf.*' -> 'default.*'), so keep both spellings and
                    # fall through to the other one if this install rejects the first
                    convert_to_candidates = correlated_background_keys()
                    # control_region_indices=[val for val in list(interpreter.channels) if not val in SRs]
                    # this might fail so we make a loop
                    simplified_background_model = None
                    for trial in range(3):
                        try:
                            simplified_background_model = converter(
                                                    statistical_model=full_background_model,
                                                    convert_to=convert_to_candidates[0],
                                                    control_region_indices=list(interpreter.channels), #control_region_indices,
                                                    number_of_samples=n_bkg_unc,

                                                )
                            break
                        except LinAlgError as e:
                            mes = f'Conversion to simplified model failed: ' + repr(e)
                            logger.error(mes)
                            if trial == 2:
                                raise
                            else:
                                mes = f'Trying again...'
                                logger.warning(mes)
                        except Exception as e:
                            # unknown conversion key -> this spey version uses the other spelling
                            if len(convert_to_candidates) > 1 and 'conversion' in repr(e).lower():
                                rejected = convert_to_candidates.pop(0)
                                logger.warning(f"This spey version does not know '{rejected}', "
                                               f"trying '{convert_to_candidates[0]}' instead.")
                                continue
                            raise
                    if simplified_background_model is None:
                        mes = 'Could not build the simplified background model.'
                        logger.critical(mes)
                        raise RuntimeError(mes)

                    cov_matrix = simplified_background_model.backend.covariance_matrix
                    del simplified_background_model
                    bkg_unc = list( zip( bins_names, list(np.sqrt(np.diag(cov_matrix))) ) )
                    del converter
                    # bkg_yields/bkg_unc are in the MODEL's bin ordering here; re-express them
                    # in the user's ordering so the tables below can be printed either way.
                    bkg_yield_dict = OrderedDict(bkg_yields)
                    bkg_unc_dict = OrderedDict(bkg_unc)
                    bkg_yields_ordered = [(bin_name, bkg_yield_dict[bin_name]) for bin_name in input_bins_ordered]
                    bkg_unc_ordered = [(bin_name, bkg_unc_dict[bin_name]) for bin_name in input_bins_ordered]
                ##############################
                # determine the observed yields
                ##############################
                logger.info(f"Geting the observed number of events.")
                workspace_obs, model_obs, data_obs = full_background_model.backend.model(expected=spey.ExpectationType.observed)
                obs_yields = list(zip(bins_names, data_obs))
                obs_yields_ordered = []
                obs_data_dict = {}
                obs_bin_iter = 0
                # NOTE: do not rebind channel_nbins here - it holds the model's full
                # channel -> nbins map and is still needed further down.
                for channel_name, n_bins_in_channel in model_obs.config.channel_nbins.items():
                    for nb in range(n_bins_in_channel):
                        bin_name = channel_name + f'-{nb}'
                        obs_data_dict[bin_name] = data_obs[obs_bin_iter + nb]
                    obs_bin_iter += n_bins_in_channel
                for bin_name in input_bins_ordered:
                    obs_yields_ordered.append([bin_name, obs_data_dict[bin_name]])
                print_yield_table(input_bins_ordered, bins_is_signal_ordered, bkg_yields_ordered, bkg_unc_ordered, obs_yields_ordered, logger)
                
                
                #################################################################
                # Check the validity of scan, points, and processes parameters
                #################################################################
                if param_dict['scans'] < 1:
                    mes = f"Number of scans has to be at least 1, but {param_dict['scans']} is provided!"
                    logger.critical(mes)
                    raise ValueError(mes)
                if param_dict['points'] < 1:
                    mes = f"Number of scans has to be at least 1, but {param_dict['points']} is provided!"
                    logger.critical(mes)
                    raise ValueError(mes)
                if param_dict['processes'] < 1:
                    mes = f"Number of scans has to be at least 1, but {param_dict['processes']} is provided!"
                    logger.critical(mes)
                    raise ValueError(mes)
                    
                if param_dict['sig_rel_unc'] < 0:
                    mes = f"Signal relative uncertainty cannot be negative! It is set to {param_dict['sig_rel_unc']}"
                    logger.critical(mes)
                    raise ValueError(mes)
                elif param_dict['sig_rel_unc'] > 1.0:
                    mes = f"Signal relative uncertainty set to {param_dict['sig_rel_unc']}! Please check this is intended!"
                    logger.warning(mes)

                ##############################
                # determine limits of the scan
                ##############################
                logger.info(f"Setting initial scan limits on S.")

                mask = get_mask(len(obs_yields), channels_and_bins, True, param_dict['signal_leakage_CR'], param_dict['signal_leakage_VR'])
                nSmin, nSmax, central_values = get_scan_limits(bkg_yields, 
                                                                bkg_unc, 
                                                                obs_yields, 
                                                                channels_and_bins,
                                                                param_dict['signal_leakage_CR'], 
                                                                param_dict['signal_leakage_VR'],
                                                                param_dict['signal_leakage_CR_spread'],
                                                                param_dict['signal_leakage_VR_spread'],
                                                                param_dict['signal_leakage_CR_sign'],
                                                                param_dict['signal_leakage_VR_sign'],
                                                                param_dict['CR_center'],
                                                                param_dict['VR_center'],
                                                                logger
                                                                )
                nSobs = get_obs_signal(bkg_yields, obs_yields)
                nSmin_orig = copy.deepcopy(nSmin)

                ##############################
                # Prepare channels for removal
                ##############################
                # resolved BEFORE find_min_S, so bins belonging to removed channels
                # can be skipped there instead of being probed and then discarded
                if param_dict['removeCRsVRs']:
                    if (param_dict['signal_leakage_CR'] or param_dict['signal_leakage_VR']):
                        mes = f"Asked to remove CRs/VRs but signal leakage enabled! I don't know what to do so I abort!"
                        logger.critical(mes)
                        raise ValueError(mes)
                    elif param_dict['remove_channels'] is None:
                        channels_to_be_removed = []
                        for c, sr, b in channels_and_bins:
                            if sr != 'SR':
                                channels_to_be_removed.append(c)
                        param_dict['remove_channels'] = channels_to_be_removed
                        del channels_to_be_removed
                    else:
                        for c, sr, b in channels_and_bins:
                            if sr != 'SR':
                                param_dict['remove_channels'] =  param_dict['remove_channels'] + [c]
                # check if the channels to be removed are available
                param_dict['remove_channels'] = [c for c in param_dict['remove_channels'] if c in channels]
                for channel_name in param_dict['remove_channels']:
                    logger.warning(f'Removing channel: {channel_name}')

                #############################################################
                # Scan limits: hardcoded in the card, or computed from yields
                #############################################################
                hardcoded_limits = load_scan_limits(param_dict['scan_limits'],
                                                    file_pair_index,
                                                    len(patchsets),
                                                    input_bins_ordered,
                                                    bins_names,
                                                    central_values,
                                                    basename(patchset_path),
                                                    logger)
                if hardcoded_limits is not None:
                    # Stored limits are only valid for the configuration they were
                    # probed with: a different signal uncertainty, or a leakage spread
                    # wider than the harvested one, puts the scan outside the range
                    # that was actually verified. Fall back to computing them.
                    if not check_scan_limits_context(param_dict.get('scan_limits_context'),
                                                     param_dict, basename(patchset_path), logger):
                        hardcoded_limits = None

                if hardcoded_limits is not None:
                    nSmin, nSmax = hardcoded_limits
                    param_dict['scan_limits_source'] = 'parameter card'
                    # report the settings the STORED limits were built for, not this
                    # run's: they are only equal because the check above passed, and
                    # with no context block they are unknown
                    limits_ctx = param_dict.get('scan_limits_context')
                    if limits_ctx:
                        provenance = (f"computed for sig_rel_unc={limits_ctx.get('sig_rel_unc', '?')}, "
                                      f"CR/VR spread {limits_ctx.get('signal_leakage_CR_spread', '?')}/"
                                      f"{limits_ctx.get('signal_leakage_VR_spread', '?')}")
                    else:
                        provenance = "settings UNKNOWN, NOT checked against this run"
                    logger.info(f"Scan limits mode: LOADED from the parameter card "
                                f"({len(bins_names)} bins, {provenance}). "
                                f"Skipping the lower-limit probe.")
                    # The stored box covers every bin, including ones this run pins or
                    # removes, so that it stays reusable. Re-impose this run's choices.
                    nSmin, nSmax, _ = apply_region_pinning(nSmin, nSmax, channels_and_bins,
                                                           param_dict['signal_leakage_CR'],
                                                           param_dict['signal_leakage_VR'],
                                                           logger)
                    # nothing was probed, so the "initial" limits are the loaded ones
                    nSmin_orig = copy.deepcopy(nSmin)
                else:
                    param_dict['scan_limits_source'] = 'computed'
                    # Only bins that the sampler will actually vary need a probed lower
                    # limit. A pinned bin (signal leakage off -> sigma 0) never moves off
                    # its central value, and a removed bin is dropped from the model
                    # before the likelihood is evaluated, so probing either one only buys
                    # the +1e-4 safety margin a bin that stays at zero must not get.
                    probe_mask = get_probe_mask(mask, channels_and_bins, param_dict['remove_channels'])
                    n_probed = int(np.sum(probe_mask))
                    logger.info(f"Scan limits mode: COMPUTED from yields "
                                f"({n_probed} of {len(bins_names)} bins probed, "
                                f"low_lim_samples={param_dict['low_lim_samples']}, "
                                f"sig_rel_unc={param_dict['sig_rel_unc']}).")
                    logger.info(f"Setting the absolute lower limit on S ...")
                    nSmin = find_min_S(param_dict['low_lim_samples'], bkg_spec, stat_wrapper, nSmin,
                                       channels_and_bins, logger, probe_mask=probe_mask,
                                       sig_rel_unc=param_dict['sig_rel_unc'])
                print_limit_table(bins_names, nSmin, nSmax, central_values, logger)
                # find mu_SIG limits for max likelihood calculation
                mu_min, mu_max = find_mu_limits(nSmin, nSmax, central_values, logger)
                poi_index = full_background_model.backend.config().poi_index
                mu_min_orig, mu_max_orig = full_background_model.backend.config().suggested_bounds[poi_index]
                mu_min = mu_min_orig if mu_min_orig < mu_min else mu_min
                mu_max = mu_max_orig if mu_max_orig > mu_max else mu_max
                mu_bounds = (mu_min, mu_max)
                logger.info(f'Mu bounds estimated: ({mu_min}, {mu_max})')
                del full_background_model

                logger.info(f"Generating initial points.")

                p0s = generate_starting_points(nSmin, nSmax, central_values,
                                    mask=mask,
                                    n=param_dict['scans'], 
                                    start_method=param_dict['start_method'], 
                                    channels_and_bins=channels_and_bins,
                                    starting_points_file=starting_points_file,
                                    starting_points_file_index=starting_points_file_index 
                                    )
                # When using input file with total yields, one has to subtract central values
                # if starting_points_file is not None:
                #     if len(p0s[0]) != len(central_values):
                #         raise IndexError(f'Size mismatch! Starting points: {len(p0s[0])}, expected: {len(central_values)}!')
                #     if len(p0s) > 1:
                #         raise NotImplementedError(f'Expected a single starting point from file, but encountered {len(p0s)}.')
                #     p0s[0] = p0s[0] - central_values

                ####################
                ######  SCAN  ######
                ####################
                
                logger.info('Saving metadata.')
                metadata = create_metadata(param_dict,
                                            bkg_yields, 
                                            bkg_unc, 
                                            obs_yields, 
                                            central_values+nSmin, 
                                            central_values+nSmax, 
                                            central_values+nSmin_orig,
                                            logger)
                # save metadata in case of scan failure
                metadata_path = join(dirpath, 'metadata.json')
                with open(metadata_path, "w") as outfile: 
                    json.dump(metadata, outfile)
                del nSmin_orig
                logger.info('Preparing the scan.')
                sigmas = calculate_sigmas(nSmin, nSmax, mask, param_dict['SR_sigma'], param_dict['CR_sigma'], param_dict['VR_sigma'], channels_and_bins, logger)

                scan_wrapper = ScanWrapper(param_dict['points'],
                                        bkg_spec, 
                                        sigmas,
                                        channels_and_bins, 
                                        central_values, 
                                        param_dict['buffer_size'], 
                                        nSmin, 
                                        param_dict['scan_criterion'],
                                        mu_bounds,
                                        seed=param_dict['seed'],
                                        remove_channels=param_dict['remove_channels'],
                                        sig_rel_unc=param_dict['sig_rel_unc'],
                                        logger=logger
                                        )
                inputs = [(p, join(dirpath, f'table-{suffix}-{scan_index}.csv')) for scan_index, p in enumerate(p0s)]
                del p0s
                gc.collect()
                logger.info(f"Running {len(inputs)} scans, {param_dict['points']} points each, with {param_dict['processes']} processes ...")

                nLL_max_results = []
                # pool
                with mp.Pool(processes=param_dict['processes'], maxtasksperchild=1) as pool:
                    for nLL_max in tqdm(pool.imap_unordered(scan_wrapper, inputs), total=len(inputs), colour='GREEN'):
                        nLL_max_results.append(nLL_max)

                # merge the results
                logger.info('Merging the results.')
                merged_file_path, min_values, max_values = merge_results([em[1] for em in inputs], keep_files = param_dict['keep_files'], suffix=suffix, logger=logger)
                logger.info(f'Results saved to {merged_file_path}')
                # update the metadata
                logger.info('Updating the metadata.')
                metadata = update_metadata(metadata, min_values, max_values, nLL_max_results[0])
                with open(metadata_path, "w") as outfile: 
                    json.dump(metadata, outfile, indent = 4)
                os.rename(metadata_path, merged_file_path.replace('.csv', '.json'))
                # calculate time of the scan (wall-clock, see analysis_time above)
                final_time = time.perf_counter_ns() - analysis_time
                time_per_point = np.round( (final_time//10**9) / ( param_dict['points'] * param_dict['scans']), 3)
                time_string = get_time_string(final_time)
                logger.info(f'Finished scan for {param_dict["analysis"]}. It took {time_string} in total, on average {time_per_point} seconds per point.')


if __name__ == "__main__":
    """Main function for script execution."""
    parser = argparse.ArgumentParser(description="Run the script with a parameter file and optional log directory.")
    parser.add_argument("param_file", nargs="?", default="parameters.yaml", help="Path to the YAML file with parameters.")
    parser.add_argument("--log_dir", default="logs", help="Directory where logs should be saved.")
    parser.add_argument("--start", default=None, help="CSV file with starting points.")
    parser.add_argument("--start_index", default=None, help="Row to choose from the input file")

    args = parser.parse_args()

    # Initialize logger
    logger = setup_logger(args.log_dir)
    logger.info('Welcome to NLL sampler by Rafal Maselek (https://orcid.org/0000-0002-5558-8249)')
    logger.info(f"Using log directory: {args.log_dir}")

    mp.set_start_method('spawn', force=True)
    logger.info("Process spawned.")

    logger.info(f"Loading the parameter file: {args.param_file}")


    start_time = datetime.now()
    if 'pyhf' not in spey.AvailableBackends():
        mes = 'PyHF backend was not found! Aborting.'
        logger.critical(mes)
        raise RuntimeError(mes)
    try:
        main(logger, args.param_file, args.start, args.start_index)
    except Exception as e:
        mes = f'Program failed: ' + repr(e)
        logger.critical(mes)
        raise Exception(mes)
    end_time = datetime.now()

    logger.info("[END] Program finished! Total time of execution {}".format(end_time - start_time))

