#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file implements functions used by sample.py to perform likelihood sampling.
#

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ.pop('TF_CONFIG', None)
import numpy as np
from math import isinf, isnan
from timeit import default_timer as timer
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow_probability as tfp
from misc import *
import pyhf
import spey
from spey_pyhf.helper_functions import WorkspaceInterpreter
import os
from tqdm_loggable.auto import tqdm as progressbar
import warnings
import random
import gc


#: Value written in place of a likelihood that came back NaN or infinite.
#: Such a row carries no usable information, so merge_results drops it.
NAN_PLACEHOLDER = 1e10
#: Number of likelihood columns at the end of every results row.
N_LIKELIHOOD_COLUMNS = 8

#: Criteria that name a single likelihood directly.
EXPLICIT_CRITERIA = ('nLL_obs_mu1', 'nLL_exp_mu1', 'LL_obs_mu1', 'LL_exp_mu1')
#: Every criterion accepted by :class:`ScanWrapper` ('mu1' picks one of the
#: nLL_*_mu1 criteria at random for each scan).
VALID_CRITERIA = EXPLICIT_CRITERIA + ('mu1',)


def set_seeds(seed):
    """Seed all relevant random number generators for reproducibility.

    Sets the seed used by the Python hash algorithm, the standard
    library ``random`` module, TensorFlow, and NumPy, so that a run can
    be reproduced deterministically given the same seed value.

    Args:
        seed (int): Seed value applied to all random number generators.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed):
    """Configure TensorFlow (and other RNGs) for fully deterministic execution.

    In addition to seeding the random number generators via
    :func:`set_seeds`, this forces TensorFlow to use deterministic
    operations and restricts it to single-threaded inter-/intra-op
    parallelism, which is required to get bit-reproducible results
    across runs.

    Args:
        seed (int): Seed value forwarded to :func:`set_seeds`.
    """
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    set_seeds(seed=seed)



def build_signal_modifiers(bin_vals, sig_rel_unc):
    """Build the modifier list attached to an injected signal.

    Used by BOTH the scan and the lower-limit probe, so the two cannot drift
    apart: whatever uncertainty the scan evaluates the likelihood with, the
    probe has to respect when deciding how negative the signal may go.

    Args:
        bin_vals (numpy.ndarray): Signal yields for the channel's bins.
        sig_rel_unc (float): Relative uncertainty on the injected signal. When
            negligible, only the ``lumi`` and ``mu_SIG`` modifiers are used.

    Returns:
        list[dict]: Modifiers for ``WorkspaceInterpreter.inject_signal``.
    """
    base = [
        {
            "data": None,
            "name": "lumi",
            "type": "lumi"
        },
        {
            "data": None,
            "name": "mu_SIG",
            "type": "normfactor"
        }
    ]
    if abs(sig_rel_unc) <= 1e-17:
        return base
    return [
        {
            "name": "Wolfgang_unc",
            "type": "histosys",
            "data": {
                "hi_data": bin_vals * (1.0+sig_rel_unc),
                "lo_data": [ float(np.max([0.0, bval*(1.0-sig_rel_unc)])) for bval in bin_vals]
                },
        },
    ] + base


def find_min_S(niter, bkg_spec, stat_wrapper, nSmin, channels_and_bins, logger, probe_mask=None,
               sig_rel_unc=0.0):
    """Find, for each bin, the most negative injectable signal that keeps the likelihood well-defined.

    Starting from a candidate lower bound on the signal yield per bin
    (``nSmin``), this iteratively bisects the injected signal fraction
    (``mu``) so that both the expected (apriori) and observed
    likelihoods at ``poi_test=1.0`` remain finite (not NaN/inf) for a
    ``pyhf``/``spey`` statistical model built from ``bkg_spec`` with the
    signal injected. This yields the true minimal (most negative)
    signal yield allowed per bin, which is used later to bound the MCMC
    scan.

    Args:
        niter (int): Number of bisection iterations to perform per bin.
            Values below 1 disable negative signal injection in signal
            regions (a warning is logged).
        bkg_spec (dict): Background-only ``pyhf`` workspace specification.
        stat_wrapper (callable): ``spey`` statistical model backend/wrapper
            (e.g. obtained via ``spey.get_backend("pyhf")``) used to build
            the statistical model for each candidate signal injection.
        nSmin (numpy.ndarray): Initial candidate lower limits on the
            signal yield, one entry per bin (flattened across channels).
        channels_and_bins (list[tuple]): List of ``(channel_name,
            channel_type, n_bins)`` tuples describing the channel layout
            and their number of bins, in the same order as ``nSmin``.
        logger (logging.Logger): Logger used to report progress and
            warnings when a stable lower limit cannot be found for a bin.
        probe_mask (numpy.ndarray, optional): Boolean per-bin mask selecting
            the bins to probe. Bins marked ``False`` - pinned bins, whose step
            size is 0, and bins of removed channels, which never reach the
            likelihood - keep their incoming ``nSmin`` and cost no model
            evaluations. Skipping them matters beyond speed: the ``+1e-4``
            safety margin applied to a probed bin would otherwise put a
            positive floor under a bin that must hold exactly zero signal.
            Defaults to probing every bin.
        sig_rel_unc (float, optional): Relative uncertainty on the injected
            signal. The probe applies the same ``histosys`` modifier the scan
            uses, so the limit it returns stays valid once that uncertainty is
            in play; with a negative signal the up-variation is
            ``S*(1+sig_rel_unc)``, i.e. more negative than the nominal.
            Defaults to 0.0 (no uncertainty).

    Returns:
        numpy.ndarray: Array of the same shape as ``nSmin`` containing
        the refined minimal signal yield allowed for each bin.
    """
    if niter < 1:
        logger.warning(f"The value of the 'low_lim_samples' parameter is {niter} < 1! No negative signal in SRs will be injected.")
    
    #pyhf.set_backend('tensorflow')
    up_lim = np.zeros(nSmin.shape, dtype=float)
    minimalS = np.zeros(nSmin.shape, dtype=float)

    if probe_mask is None:
        probe_mask = np.ones(nSmin.shape, dtype=bool)
    else:
        probe_mask = np.asarray(probe_mask, dtype=bool)
        if probe_mask.shape != nSmin.shape:
            mes = f'[ERROR] probe_mask has shape {probe_mask.shape} but {nSmin.shape} expected!'
            logger.critical(mes)
            raise ValueError(mes)
    # bins that are not probed keep the limit they came in with
    minimalS[~probe_mask] = nSmin[~probe_mask]

    # Bins whose FIRST probe (mu=1, the full analytic candidate) came back
    # non-finite. Bisecting down from there is the intended behaviour - the
    # candidate is an estimate and the probe exists to shrink it - but the
    # count is worth reporting, because a bin that cannot take its analytic
    # floor gives up part of the negative range the scan was meant to cover,
    # and a run where MOST bins do that is usually an input problem rather
    # than physics. Kept as a summary, not a per-bin error.
    reduced_at_mu1 = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nbins = int(np.sum(probe_mask))
        ii = 0

        with progressbar(total=nbins*niter) as pbar:
            interpreter = WorkspaceInterpreter(bkg_spec)
            # iterate over all channels and bins
            for jj in range(len(channels_and_bins)):
                c, sr, b = channels_and_bins[jj]
                # extract signal value for given channel
                bin_vals = np.array(nSmin[ii:ii+b])
                # get the default signal for injection
                inject_vals = np.zeros(bin_vals.shape)
                # iterate over all bins in that channel
                for bb in range(b):
                    if not probe_mask[ii+bb]:
                        # pinned or removed: leave inject_vals[bb] at 0 so this bin
                        # contributes no signal while the other bins are probed
                        continue
                    # Optimise the signal fraction for this bin, starting at the
                    # full analytic candidate (mu=1) and bisecting on a bracket:
                    #   mu_fail = the smallest fraction KNOWN to break the model
                    #   mu_ok   = the largest fraction KNOWN to work
                    # Keeping both ends is what makes this a bisection. Halving
                    # the current mu on every failure, as this loop used to do,
                    # can step BELOW a fraction already proven good - and since
                    # the limit was taken from the most recent success rather
                    # than the best one, the probe could hand back a floor it had
                    # already bettered earlier in the same loop.
                    mu = 1.0
                    mu_fail = 1.0
                    mu_ok = None
                    mu_new = None
                    # perform niter steps to find the lower limit on signal
                    for nn in range(niter):
                        # Probe THIS bin against the background-only model and
                        # nothing else. A single interpreter reused across the
                        # loop keeps every injection it has been given, so by
                        # the time the loop reaches a late channel all the
                        # earlier ones are already sitting at their most
                        # negative signal. That joint configuration can be
                        # invalid even when every bin on its own is fine, and
                        # then every remaining bin fails for a reason that has
                        # nothing to do with its own limit - the result depends
                        # on the channel iteration order. Rebuilding here keeps
                        # each bin's limit its own.
                        interpreter = WorkspaceInterpreter(bkg_spec)
                        # set signal for given bin
                        inject_vals[:] = 0.0
                        inject_vals[bb] = np.round(bin_vals[bb] * mu+1e-4, 4)
                        # inject the signal to all bins, with the SAME modifiers the
                        # scan will use - with a relative signal uncertainty the
                        # histosys up-variation of a negative signal is more negative
                        # than the nominal, so the limit has to be probed with it
                        interpreter.inject_signal(c, inject_vals,
                                                  modifiers=build_signal_modifiers(inject_vals, sig_rel_unc))
                        statistical_model = stat_wrapper(
                                                    background_only_model=interpreter.background_only_model,
                                                    signal_patch=interpreter.make_patch(),
                                                    )
                        statistical_model.backend.manager.backend = "tensorflow"
                        nLL_exp_mu1 = statistical_model.likelihood(poi_test=1.0, expected='apriori')
                        #nLL_exp_mu1 = tf.squeeze(nLL_exp_mu1).numpy()
                        nLL_obs_mu1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
                        #nLL_obs_mu1 = tf.squeeze(nLL_obs_mu1).numpy()
                        
                        # print(c, mu, bin_vals[bb] * mu, nLL_exp_mu1, nLL_obs_mu1, inject_vals)
                        if isnan(nLL_exp_mu1) or isnan(nLL_obs_mu1) or isinf(nLL_exp_mu1) or isinf(nLL_obs_mu1):
                            if nn == 0:
                                reduced_at_mu1.append((f'{c}-{bb}', float(bin_vals[bb]), ii + bb))
                            mu_fail = mu
                            # with nothing proven good yet there is no bracket to
                            # bisect, so fall back to halving until something works
                            mu = mu / 2.0 if mu_ok is None else (mu_ok + mu_fail) / 2.0
                        else:
                            # only ever move the limit further out, never back in
                            if mu_ok is None or mu > mu_ok:
                                minimalS[ii+bb] = inject_vals[bb]
                                mu_ok = mu
                            mu_new = mu
                            if np.isclose(mu, 1.0, atol=1e-3):
                                break
                            mu = (mu_ok + mu_fail) / 2.0
                    pbar.update(niter)

                    if mu_new is None:
                        logger.warning(f'Not able to find lower limit on negative signal in {c}-{bb}. No negative signal in this bin will be used. Consider increasing "low_lim_samples" parameter.')
                        minimalS[ii+bb] = up_lim[ii+bb]
                ii += b
    del interpreter, inject_vals

    _report_reduced_at_mu1(logger, reduced_at_mu1, minimalS, int(np.sum(probe_mask)))
    return minimalS


def _report_reduced_at_mu1(logger, reduced, minimalS, nprobed):
    """Summarise the bins that could not take their full analytic floor.

    ``mu=1`` is the candidate ``get_scan_limits`` derives from the card's
    background yields, their uncertainties and the observed counts. The probe
    is there precisely to walk that candidate down when the model does not
    hold at it, so a failure at ``mu=1`` is not an error - but it does mean
    the scan explores less negative signal than intended, and it happens for
    two very different reasons that are worth telling apart:

      * physics - the card holds a post-fit background above the workspace's
        own nominal template, so the analytic floor asks for a total yield the
        template cannot reach. Expect a handful of bins.
      * inputs  - bkg_yields/bkg_unc do not belong to this workspace, or the
        channel-to-bin mapping slipped. Expect most bins to fail.

    The ratio of affected bins is what separates them, so it is reported.
    """
    if not reduced:
        return
    kept = []
    for name, cand, idx in reduced:
        kept.append(abs(minimalS[idx] / cand) if cand else 0.0)
    frac = len(reduced) / max(nprobed, 1)
    worst = min(kept) if kept else 0.0
    median = float(np.median(kept)) if kept else 0.0
    order = sorted(zip(reduced, kept), key=lambda rk: rk[1])
    head = ', '.join(f'{n} {k:.0%}' for (n, _, _), k in order[:3])
    logger.warning(
        f'Lower limit reduced in {len(reduced)} of {nprobed} bins ({frac:.0%}); they keep '
        f'{median:.0%} of the target range at the median, {worst:.0%} at worst. This is normal. '
        f'Worst: {head}.')
    if frac > 0.5:
        logger.warning(
            f'Over half the bins needed that - usually an input problem rather than physics. '
            f'Check that bkg_yields/bkg_unc belong to this workspace and follow its channel order.')


class NewStateWrapper():
    """Callable proposal generator for the random-walk Metropolis MCMC sampler.

    Given the previous MCMC state, proposes a new state by sampling from
    a diagonal multivariate normal distribution centered on the previous
    state, then truncates the proposal so that no component falls below
    the allowed minimal signal yield.
    """
    def __init__(self, std, minimalS_allowed):
        """Store the per-dimension step size and the truncation floor.

        Args:
            std (numpy.ndarray): Per-dimension standard deviations
                (step sizes) used for the proposal distribution.
            minimalS_allowed (numpy.ndarray): Per-dimension lower bounds;
                proposed values below these are clipped up to them.
        """
        #self.nSmin = tf.convert_to_tensor(nSmin, dtype=float)
        #self.nSmax = tf.convert_to_tensor(nSmax, dtype=float)
        self.minimalS_allowed = minimalS_allowed
        self.std = std
        self.dims = np.shape(std)[0]
    def __call__(self, prev_state, seed):
        """Draw a new proposal state from the previous MCMC state.

        Args:
            prev_state (array-like): Current state of the MCMC chain
                (signal yields per bin).
            seed: Random seed forwarded to the TensorFlow Probability
                sampler for reproducibility.

        Returns:
            numpy.ndarray: Proposed new state, with any component below
            ``minimalS_allowed`` clipped to that lower bound.
        """
        #a = tf.convert_to_tensor(a, dtype=float)
        new_state_dist = tfp.distributions.MultivariateNormalDiag(loc=prev_state, scale_diag=self.std*np.ones(self.dims))
        new_state = new_state_dist.sample(seed=seed)
        new_state = np.where(new_state>=self.minimalS_allowed, new_state, self.minimalS_allowed)
        del new_state_dist
        return new_state


class LikelihoodCalculatorWrapper():
    """Callable target log-probability function for the MCMC scan.

    Instances of this class keep track of the background specification
    and construct a signal patch from a vector of signal yields (``S``)
    on each call, computing the (negative) log-likelihood used as the
    MCMC target density. It also buffers computed likelihoods and
    yields, periodically flushing them to ``output_file``, and caches
    the mu=0 and maximum likelihoods (computed once, on the first call).
    """
    # create a callable object that will keep information about the background and construct patches from the S yields
    def __init__(self, bkg_spec, channels_and_bins, central_values, output_file, buff_size, \
                criterion, mu_bounds, seed, remove_channels, sig_rel_unc, logger):
        """Initialize the wrapper and write the output CSV header.

        Args:
            bkg_spec (dict): Background-only ``pyhf`` workspace specification.
            channels_and_bins (list[tuple]): List of ``(channel_name,
                channel_type, n_bins)`` tuples describing the analysis
                channel layout.
            central_values (numpy.ndarray): Central (background) yields
                per bin, added to the sampled signal yields when saving
                results.
            output_file (str): Path to the CSV file where per-sample
                yields and likelihoods will be written. Created fresh;
                raises if it already exists.
            buff_size (int): Number of samples to accumulate in memory
                before flushing to ``output_file``.
            criterion (str): Which quantity to return (negated) as the
                MCMC target log-probability. One of ``'nLL_obs_mu1'``,
                ``'nLL_exp_mu1'``, ``'LL_obs_mu1'``, ``'LL_exp_mu1'``.
            mu_bounds (tuple): ``(mu_min, mu_max)`` bounds on the signal
                strength parameter used during likelihood maximization.
            seed (int): Random seed applied at the start of every call
                for reproducibility.
            remove_channels (list[str] or None): Names of channels to
                exclude (e.g. control/validation regions) from the fit
                and from the saved output columns.
            sig_rel_unc (float): Relative uncertainty on the injected
                signal yield, used to add a ``histosys`` modifier when
                non-zero.
            logger (logging.Logger): Logger used for progress, debug and
                error messages.

        Raises:
            FileExistsError: If ``output_file`` already exists.
        """
        self._bkg_spec = bkg_spec
        self._channels_and_bins = channels_and_bins
        self._counter = 0
        self._output_file = output_file
        self._buff_size = buff_size
        self._central_values = central_values
        self._criterion = criterion
        if self._criterion not in EXPLICIT_CRITERIA:
            mes = f'[ERROR] Wrong criterion passed to LikelihoodCalculatorWrapper: {self._criterion}! ' \
                  f'Expected one of {EXPLICIT_CRITERIA}.'
            logger.critical(mes)
            raise ValueError(mes)
        self._mu_bounds = mu_bounds
        self._remove_channels = [] if remove_channels is None else remove_channels
        self._sig_rel_unc = sig_rel_unc
        self.logger = logger
        # write the first line with header names
        bin_no = 0
        try:
            line = ''
            for c, sr, b in channels_and_bins:
                    for val in range(b):
                        if c not in self._remove_channels:
                            line = line + f'{c}-{val},'
                        bin_no += 1
            line = line + 'nLL_exp_mu0,nLL_exp_mu1,nLL_obs_mu0,nLL_obs_mu1,nLLA_exp_mu0,nLLA_exp_mu1,nLLA_obs_mu0,nLLA_obs_mu1\n'
            with open(self._output_file, 'w') as fout:
                fout.write(line)
        except FileExistsError:
            mes = '[ERROR] The output file {} already exists!'
            self.logger.critical(mes)
            raise FileExistsError(mes)

        self._bin_no = bin_no
        # handle the removed channels
        # NOTE: the mask is indexed by BIN, not by channel, so every bin of a
        # removed channel has to be switched off (channels may hold >1 bin).
        self._mask = np.ones(shape=bin_no, dtype=bool)
        bin_offset = 0
        for c, sr, b in channels_and_bins:
            if c in self._remove_channels:
                self._bin_no = self._bin_no - b
                self._mask[bin_offset:bin_offset + b] = False
            bin_offset += b

        if int(np.sum(self._mask)) != self._bin_no:
            mes = f'[ERROR] Bin bookkeeping mismatch after removing channels: ' \
                  f'mask keeps {int(np.sum(self._mask))} bins but {self._bin_no} expected!'
            self.logger.critical(mes)
            raise ValueError(mes)

        self._results = np.empty(shape=(self._buff_size, self._bin_no+8), dtype=float)
        self.nLL_exp_mu0 = None
        self.nLL_obs_mu0 = None
        self.nLLA_exp_mu0 = None
        self.nLLA_obs_mu0 = None
        self._stat_wrapper = spey.get_backend("pyhf")
        self._seed = seed

        gc.enable()
        gc.set_threshold(350, 5, 5)


    def calculate_Lmu0(self, S_yields):
        """Compute and cache the mu=0 (background-only) likelihoods.

        Builds a statistical model with the given signal yields injected,
        then computes the expected (apriori) and observed negative
        log-likelihoods, as well as their Asimov counterparts, all at
        ``poi_test=0.0``. Results are cached on the instance
        (``nLL_exp_mu0``, ``nLL_obs_mu0``, ``nLLA_exp_mu0``,
        ``nLLA_obs_mu0``) for reuse in subsequent calls.

        Args:
            S_yields (array-like): Signal yields per bin to inject before
                computing the likelihoods.
        """
        self.logger.debug('Calculating nLL for mu=0.')
        interpreter = WorkspaceInterpreter(self._bkg_spec)
        interpreter = self.inject_signal(interpreter, S_yields)
        for channel_name in self._remove_channels:
            interpreter.remove_channel(channel_name)
        
        my_patch = interpreter.make_patch()
        self.logger.debug('First patch: {}'.format(my_patch))

        statistical_model = self._stat_wrapper(
                                            background_only_model=interpreter.background_only_model,
                                            signal_patch=my_patch,
                                        )
        statistical_model.backend.manager.backend = "tensorflow"
        nLL_exp_mu0 = statistical_model.likelihood(poi_test=0.0, expected='apriori')
        nLL_exp_mu0 = self.check_for_nan(nLL_exp_mu0, 'nLL_exp_mu0')
        self.nLL_exp_mu0 = nLL_exp_mu0
        nLL_obs_mu0 = statistical_model.likelihood(poi_test=0.0, expected='observed')
        nLL_obs_mu0 = self.check_for_nan(nLL_obs_mu0, 'nLL_obs_mu0')
        self.nLL_obs_mu0 = nLL_obs_mu0
        # Asimov likelihood
        nLLA_exp_mu0 = statistical_model.asimov_likelihood(poi_test=0.0, expected='apriori')
        nLLA_exp_mu0 = self.check_for_nan(nLLA_exp_mu0, 'nLLA_exp_mu0')
        self.nLLA_exp_mu0 = nLLA_exp_mu0
        nLLA_obs_mu0 = statistical_model.asimov_likelihood(poi_test=0.0, expected='observed')
        nLLA_obs_mu0 = self.check_for_nan(nLLA_obs_mu0, 'nLLA_obs_mu0')
        self.nLLA_obs_mu0 = nLLA_obs_mu0

        del interpreter, statistical_model
        gc.collect()

    def calculate_Lmax(self, S_yields):
        """Compute the maximum-likelihood estimates for the given signal yields.

        Builds a statistical model with the given signal yields injected,
        restricts the signal-strength parameter bounds to
        ``self._mu_bounds``, and maximizes the likelihood (and its Asimov
        counterpart) for both expected (apriori) and observed data.
        Failures in any of the four maximizations are caught and logged,
        with ``[None, None]`` substituted for that result. The combined
        results are stored in ``self.nLL_max`` as a list of
        ``[nLL_exp_max, nLL_obs_max, nLLA_exp_max, nLLA_obs_max]``.

        Args:
            S_yields (array-like): Signal yields per bin to inject before
                maximizing the likelihood.
        """
        self.logger.info('Calculating maximum likelihood.')
        stat_wrapper = spey.get_backend("pyhf")
        interpreter = WorkspaceInterpreter(self._bkg_spec)
        interpreter = self.inject_signal(interpreter, S_yields)
        for channel_name in self._remove_channels:
            interpreter.remove_channel(channel_name)
        
        my_patch = interpreter.make_patch()
        self.logger.debug('First patch: {}'.format(my_patch))

        statistical_model = self._stat_wrapper(
                                            background_only_model=interpreter.background_only_model,
                                            signal_patch=my_patch,
                                        )
        par_bounds = statistical_model.backend.config().suggested_bounds
        poi_index = statistical_model.backend.config().poi_index
        par_bounds[poi_index] = self._mu_bounds

        try:
            nLL_exp_max = statistical_model.maximize_likelihood(expected=spey.ExpectationType.apriori, par_bounds=par_bounds, )
        except Exception as e:
            self.logger.error('Failed to calculate nLL_exp_max: '+str(e))
            nLL_exp_max = [None, None]
        
        try:    
            nLL_obs_max = statistical_model.maximize_likelihood(par_bounds=par_bounds, )
        except Exception as e:
            self.logger.error('Failed to calculate nLL_obs_max: '+str(e))
            nLL_obs_max = [None, None]
        
        try:                            
            nLLA_exp_max = statistical_model.maximize_asimov_likelihood(test_statistics="qmutilde", expected=spey.ExpectationType.apriori, par_bounds=par_bounds,)
        except Exception as e:
            self.logger.error('Failed to calculate nLLA_exp_max: '+str(e))
            nLLA_exp_max = [None, None]
        
        try:
            nLLA_obs_max = statistical_model.maximize_asimov_likelihood(test_statistics="qmutilde", par_bounds=par_bounds)
        except Exception as e:
            self.logger.error('Failed to calculate nLLA_obs_max: '+str(e))
            nLLA_obs_max = [None, None]

        self.nLL_max = [list(nLL_exp_max), list(nLL_obs_max), list(nLLA_exp_max), list(nLLA_obs_max)]

    def clear_buffer(self):
        """Reset the in-memory results buffer and sample counter.

        Discards the current results array and allocates a fresh empty
        buffer of shape ``(buff_size, n_bins + 8)``, resetting the
        internal sample counter to zero.
        """
        del self._results
        self._results = np.empty(shape=(self._buff_size, self._bin_no+8), dtype=float)
        self._counter = 0

    def inject_signal(self, interpreter, S_yields):
        """Inject signal yields (with appropriate modifiers) into each channel.

        Iterates over ``self._channels_and_bins``, slicing the
        corresponding signal yields out of ``S_yields`` for each channel,
        and injects them into the workspace via ``interpreter``. If
        ``self._sig_rel_unc`` is non-negligible, a ``histosys`` modifier
        encoding the relative signal uncertainty is added alongside the
        standard ``lumi`` and ``mu_SIG`` (``normfactor``) modifiers;
        otherwise only the latter two are used.

        Args:
            interpreter (WorkspaceInterpreter): Workspace interpreter used
                to inject the signal and build the resulting patch.
            S_yields (array-like): Signal yields per bin, in the same
                flattened order as ``self._channels_and_bins``.

        Returns:
            WorkspaceInterpreter: The same ``interpreter`` instance, with
            the signal injected into every channel.
        """
        ii = 0
        for c, sr, b in self._channels_and_bins:
            bin_vals = np.array(S_yields[ii:ii + b])
            interpreter.inject_signal(
                c,
                bin_vals,
                modifiers=build_signal_modifiers(bin_vals, self._sig_rel_unc),
            )

            ii += b
        return interpreter

    def save_results(self, counter=None):
        """Append the buffered results to the output CSV file and clear the buffer.

        Args:
            counter (int, optional): Number of valid rows in the results
                buffer to write out. Defaults to ``self._counter`` (the
                number of samples accumulated so far).
        """
        if counter is None:
            counter = self._counter
        with open(self._output_file, 'a') as fout:
                np.savetxt(fout, self._results[:counter], fmt="%+010.8f", delimiter=',')
        self.clear_buffer()

    def check_for_nan(self, likelihood, name):
        """Sanitize a likelihood value, substituting a large finite value for NaN/inf.

        Args:
            likelihood (float): Likelihood value to check.
            name (str): Human-readable name of the quantity, used in the
                logged error message if ``likelihood`` is not finite.

        Returns:
            float: ``likelihood`` unchanged, or ``1e10`` if it was NaN or
            infinite (``-inf`` is mapped to ``-1e10``).
        """
        if isnan(likelihood):
            self.logger.error(f'[ERROR] {name} is {likelihood}! I will write it as +{NAN_PLACEHOLDER:.0e}')
            return np.float64(NAN_PLACEHOLDER)
        elif isinf(likelihood):
            replacement = np.float64(-NAN_PLACEHOLDER) if likelihood < 0 else np.float64(NAN_PLACEHOLDER)
            self.logger.error(f'[ERROR] {name} is {likelihood}! I will write it as {replacement:+.0e}')
            return replacement
        else:
            return likelihood
    
    def __call__(self, S_yields):
        """Compute the MCMC target log-probability for a proposed signal state.

        On the first call, caches the mu=0 likelihoods (via
        :meth:`calculate_Lmu0`) and the maximum likelihoods (via
        :meth:`calculate_Lmax`). On every call, injects ``S_yields`` into
        the workspace, computes the mu=1 expected/observed likelihoods
        (and their Asimov counterparts), buffers the resulting yields and
        likelihoods (flushing to disk via :meth:`save_results` once the
        buffer is full), and returns the quantity selected by
        ``self._criterion`` as the value to be used by the MCMC sampler.

        Args:
            S_yields (array-like): Proposed signal yields per bin for
                this MCMC step.

        Returns:
            float: The (possibly negated) likelihood or log-likelihood
            selected by ``self._criterion``, used as target
            log-probability by the random-walk Metropolis sampler.

        Raises:
            ValueError: If ``self._criterion`` is not one of the
                recognized criterion strings.
        """
        set_seeds(self._seed)
        if self.nLL_exp_mu0 is None:
            # first call
            self.calculate_Lmu0(S_yields)
            self.calculate_Lmax(S_yields)
        interpreter = WorkspaceInterpreter(self._bkg_spec)
        interpreter = self.inject_signal(interpreter, S_yields)
        for channel_name in self._remove_channels:
            interpreter.remove_channel(channel_name)

        new_patch = interpreter.make_patch()
        # self.logger.warning(new_patch)
        statistical_model = self._stat_wrapper(
                                            background_only_model=interpreter.background_only_model,
                                            signal_patch=new_patch,
                                        )
        statistical_model.backend.manager.backend = "tensorflow"
        nLL_exp_mu1 = statistical_model.likelihood(poi_test=1.0, expected='apriori')       
        nLL_exp_mu1 = self.check_for_nan(nLL_exp_mu1, 'nLL_exp_mu1')
        nLL_obs_mu1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
        nLL_obs_mu1 = self.check_for_nan(nLL_obs_mu1, 'nLL_obs_mu1')
        # Asimov likelihoods
        nLLA_exp_mu1 = statistical_model.asimov_likelihood(poi_test=1.0, expected='apriori')       
        nLLA_exp_mu1 = self.check_for_nan(nLLA_exp_mu1, 'nLLA_exp_mu1')
        nLLA_obs_mu1 = statistical_model.asimov_likelihood(poi_test=1.0, expected='observed')
        nLLA_obs_mu1 = self.check_for_nan(nLLA_obs_mu1, 'nLLA_obs_mu1')
        likelihoods_to_save = [self.nLL_exp_mu0, nLL_exp_mu1, self.nLL_obs_mu0, nLL_obs_mu1, \
                                self.nLLA_exp_mu0, nLLA_exp_mu1, self.nLLA_obs_mu0, nLLA_obs_mu1]
        yields_to_save = list(np.array((self._central_values+S_yields))[self._mask])
        self._results[self._counter] = np.array( yields_to_save + likelihoods_to_save, dtype=float)
        self._counter += 1
        del interpreter, statistical_model
        if self._counter == self._buff_size:
            self.save_results()
            gc.collect()
        if self._criterion == 'nLL_obs_mu1':
            return -1*nLL_obs_mu1
        elif self._criterion == 'nLL_exp_mu1':
            return -1*nLL_exp_mu1
        elif self._criterion == 'LL_obs_mu1':
            return nLL_obs_mu1
        elif self._criterion == 'LL_exp_mu1':
            return nLL_exp_mu1
        else:
            mes = f'[ERROR] Wrong criterion passed to LikelihoodCalculatorWrapper: {self._criterion}!'
            self.logger.critical(mes)
            raise ValueError(mes)

    def get_counter(self):
        """Return the number of results currently held in the in-memory buffer.

        Returns:
            int: Number of samples accumulated since the last flush.
        """
        return self._counter

def scan(p0, N, stds, minimalS_allowed, bkg_spec, channels_and_bins, central_values, output_file, buff_size, criterion, mu_bounds, seed, remove_channels, sig_rel_unc, logger):
    """Run a single random-walk Metropolis MCMC chain over signal yields.

    Builds a :class:`LikelihoodCalculatorWrapper` as the target
    log-probability function and a :class:`NewStateWrapper` as the
    proposal function, then runs ``N-1`` steps of TensorFlow
    Probability's ``RandomWalkMetropolis`` sampler starting from ``p0``.
    Any results still buffered in memory after the chain finishes are
    flushed to disk.

    Args:
        p0 (array-like): Initial state (signal yields per bin) for the chain.
        N (int): Total number of MCMC samples desired for the chain
            (``N-1`` transition steps are run in addition to ``p0``).
        stds (numpy.ndarray): Per-dimension proposal standard deviations,
            passed to :class:`NewStateWrapper`.
        minimalS_allowed (numpy.ndarray): Per-dimension lower bounds on
            the signal yield, passed to :class:`NewStateWrapper`.
        bkg_spec (dict): Background-only ``pyhf`` workspace specification.
        channels_and_bins (list[tuple]): List of ``(channel_name,
            channel_type, n_bins)`` tuples describing the channel layout.
        central_values (numpy.ndarray): Central (background) yields per bin.
        output_file (str): Path to the CSV file where sampled yields and
            likelihoods are written.
        buff_size (int): Number of samples to buffer before flushing to disk.
        criterion (str): Which likelihood quantity to use as MCMC target;
            forwarded to :class:`LikelihoodCalculatorWrapper`.
        mu_bounds (tuple): ``(mu_min, mu_max)`` bounds on the signal
            strength used during likelihood maximization.
        seed (int): Random seed used to seed RNGs and the MCMC sampler.
        remove_channels (list[str] or None): Channels to exclude from the fit.
        sig_rel_unc (float): Relative uncertainty on the injected signal yield.
        logger (logging.Logger): Logger for progress and error messages.

    Returns:
        list: The cached maximum-likelihood results
        (``target_log_prob_fn.nLL_max``), i.e. a list of
        ``[nLL_exp_max, nLL_obs_max, nLLA_exp_max, nLLA_obs_max]``
        computed on the first call of the target log-probability function.
    """
    set_seeds(seed)
    target_log_prob_fn = LikelihoodCalculatorWrapper(bkg_spec, channels_and_bins, central_values, output_file, buff_size, criterion, mu_bounds, seed, remove_channels, sig_rel_unc, logger) 
    new_state_fn_truncated = NewStateWrapper(stds, minimalS_allowed)   
    RandomWalkMH=tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn, new_state_fn=new_state_fn_truncated, name=None)
    tfp.mcmc.sample_chain(
            num_results=int(N-1),
            num_burnin_steps=0,
            current_state=p0,
            kernel=RandomWalkMH,
            trace_fn=None,
            return_final_kernel_results = False,
            parallel_iterations=1,
            seed=seed,
            )
    final_counter = target_log_prob_fn.get_counter()
    if final_counter != 0:
        target_log_prob_fn.save_results(final_counter)
    return target_log_prob_fn.nLL_max

    

def calculate_sigmas(nSmin, nSmax, mask, SR_sigma, CR_sigma, VR_sigma, channels_and_bins, logger=None):
    """Compute per-bin MCMC proposal standard deviations from the scan range.

    For each channel, the proposal standard deviation is set to a
    fraction (``SR_sigma``, ``CR_sigma``, or ``VR_sigma`` depending on the
    channel type) of the full scan range (``nSmax - nSmin``) for its
    bins. Bins masked out (signal leakage disabled) are given a standard
    deviation of zero so they are not varied by the sampler.

    Args:
        nSmin (numpy.ndarray): Lower limits on the signal yield per bin.
        nSmax (numpy.ndarray): Upper limits on the signal yield per bin.
        mask (numpy.ndarray): Boolean mask; ``False`` entries correspond
            to bins that should not be varied (std forced to 0).
        SR_sigma (float): Fraction of the scan range used as std for
            signal-region (``'SR'``) bins.
        CR_sigma (float): Fraction of the scan range used as std for
            control-region (``'CR'``) bins.
        VR_sigma (float): Fraction of the scan range used as std for
            validation-region (``'VR'``) bins.
        channels_and_bins (list[tuple]): List of ``(channel_name,
            channel_type, n_bins)`` tuples describing the channel layout.
        logger (logging.Logger, optional): Logger used to report a
            critical message if an unrecognized channel type is
            encountered. If ``None``, the error is raised without being
            logged first.

    Returns:
        numpy.ndarray: Array of per-bin proposal standard deviations,
        same shape as ``nSmax``.

    Raises:
        ValueError: If a channel type other than ``'SR'``, ``'CR'``, or
            ``'VR'`` is encountered.
    """
    stds = np.empty(nSmax.shape, dtype=float)
    deltaS = nSmax-nSmin
    ii = 0
    for c, t, b in channels_and_bins:
        val = None
        if t=='SR':
            val = SR_sigma
        elif t=='CR':
            val = CR_sigma
        elif t=='VR':
            val = VR_sigma
        else:
            mes = f"[ERROR] Wrong type of the channel provided: {t}!"
            if logger is not None:
                logger.critical(mes)
            raise ValueError(mes)
        stds[ii:ii+b] = val * deltaS[ii:ii+b]
        ii += b
    stds[~mask] = 0.0 # when signal leakage is disabled, do not vary CRs/VRs
    return stds


class ScanWrapper():
    """Callable wrapper that runs a single MCMC scan, suitable for use with ``multiprocessing``.

    Bundles all the parameters needed by :func:`scan` (background
    specification, proposal standard deviations, channel layout, output
    settings, etc.) so that an instance can be called with just a
    ``(starting_point, output_file)`` pair, e.g. via
    ``multiprocessing.Pool.imap_unordered``.
    """
    def __init__(self, N,  bkg_spec, sigmas, channels_and_bins, central_values, buff_size, minimalS_allowed, criterion, mu_bounds, seed, remove_channels=None, sig_rel_unc=0.0, logger=None):
        """Store the fixed parameters shared by all scans launched via this wrapper.

        Args:
            N (int): Number of MCMC samples to generate per scan.
            bkg_spec (dict): Background-only ``pyhf`` workspace specification.
            sigmas (numpy.ndarray): Per-bin proposal standard deviations
                (e.g. from :func:`calculate_sigmas`).
            channels_and_bins (list[tuple]): List of ``(channel_name,
                channel_type, n_bins)`` tuples describing the channel layout.
            central_values (numpy.ndarray): Central (background) yields per bin.
            buff_size (int): Number of samples to buffer before flushing to disk.
            minimalS_allowed (numpy.ndarray): Per-bin lower bounds on the
                signal yield allowed during the scan.
            criterion (str): Selection of which likelihood the sampler
                should target; one of ``'nLL_obs_mu1'``, ``'nLL_exp_mu1'``,
                ``'LL_obs_mu1'``, ``'LL_exp_mu1'``, or ``'mu1'`` (in which
                case one of the ``nLL_*_mu1`` criteria is chosen at random
                on each call).
            mu_bounds (tuple): ``(mu_min, mu_max)`` bounds on the signal
                strength used during likelihood maximization.
            seed (int): Random seed used to seed RNGs for each scan.
            remove_channels (list[str], optional): Channels to exclude
                from the fit. Defaults to ``None``.
            sig_rel_unc (float, optional): Relative uncertainty on the
                injected signal yield. Defaults to ``0.0``.
            logger (logging.Logger, optional): Logger to use; if ``None``,
                a new one is created via ``setup_logger()``.
        """
        self._N = N
        self._bkg_spec = bkg_spec
        self._channels_and_bins = channels_and_bins
        self._buff_size = buff_size
        self._minimalS_allowed = minimalS_allowed
        self._criterion = criterion
        if self._criterion not in VALID_CRITERIA:
            mes = f'[ERROR] Wrong criterion passed to ScanWrapper: {self._criterion}! ' \
                  f'Expected one of {VALID_CRITERIA}.'
            if logger is not None:
                logger.critical(mes)
            raise ValueError(mes)
        self._stds = sigmas
        self._central_values = central_values
        self._mu_bounds = mu_bounds
        self._seed = seed
        self._remove_channels = remove_channels
        self._sig_rel_unc = sig_rel_unc
        self.nLL_max = None
        if logger is None:
            self.logger = setup_logger()
        else:
            self.logger = logger

    
    def __call__(self, dat):
        """Run one MCMC scan for a given starting point and output file.

        Resolves the effective criterion (randomly choosing between
        expected and observed mu=1 criteria if ``self._criterion ==
        'mu1'``), then delegates to :func:`scan` with all stored
        parameters. Only the first call's maximum-likelihood result is
        retained in ``self.nLL_max``.

        Args:
            dat (tuple): ``(p0, output_file)`` where ``p0`` is the
                starting state (signal yields per bin) for the chain and
                ``output_file`` is the path of the CSV file to write
                results to.

        Returns:
            list or None: The maximum-likelihood results
            (``[nLL_exp_max, nLL_obs_max, nLLA_exp_max, nLLA_obs_max]``)
            if this is the first call on this instance (``self.nLL_max``
            was ``None``), otherwise ``None`` implicitly.

        Raises:
            ValueError: If ``self._criterion`` is not a recognized value.
        """
        set_seeds(self._seed)
        p0, output_file = dat
        if self._criterion in EXPLICIT_CRITERIA:
            criterion = self._criterion
        elif self._criterion == 'mu1':
            # str(), because np.random.choice returns a numpy array/str, and the
            # criterion is later compared against plain Python strings.
            criterion = str(np.random.choice(['nLL_exp_mu1', 'nLL_obs_mu1']))
        else:
            mes = f'[ERROR] Wrong criterion passed to ScanWrapper: {self._criterion}!'
            self.logger.critical(mes)
            raise ValueError(mes)


        nLL_max = scan(p0, self._N, self._stds, self._minimalS_allowed, self._bkg_spec, \
            self._channels_and_bins, self._central_values, output_file, \
            self._buff_size, criterion, self._mu_bounds, self._seed, self._remove_channels, \
            self._sig_rel_unc, self.logger)
        gc.collect()
        if self.nLL_max is None:
            self.nLL_max = nLL_max
            return self.nLL_max




