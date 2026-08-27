#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file provides some functions used in sample.py.
#

import numpy as np 
from prettytable import PrettyTable
from os.path import dirname, join, isfile, basename
import os
from misc import *
import copy
import pandas as pd
from name_dict import analysis_name_dict

def get2nd(arr):
    """Take the value out of a list of ``(bin_name, value)`` pairs.

    Args:
        arr (list[tuple]): Pairs of bin name and value, in bin order.

    Returns:
        numpy.ndarray: The values alone, in the same order.
    """
    return np.array([b for a,b in arr])


#: Keys that are legal in a parameter document but are not scan parameters,
#: so they do not appear in ``default_param_dict``.
STRUCTURAL_PARAM_KEYS = frozenset({
    'analysis',     # name of the analysis described by this document
    'name',         # alternative spelling used by single-analysis files
    'include',      # resolved (and removed) while loading the YAML
    'bkgfiles',     # provided by the analysis card
    'patchsets',    # provided by the analysis card
    'channels',     # provided by the analysis card
    'seed',         # optional, otherwise derived from the clock
    'merged',       # metadata flag added by the sampler itself
})


def correlated_background_keys():
    """Return the accepted spellings of spey's correlated-background pdf key.

    spey renamed the simplified-likelihood conversion keys after 0.2.5
    (``default_pdf.correlated_background`` -> ``default.correlated_background``).
    The name matching the installed version comes first.

    Returns:
        list[str]: Candidate ``convert_to`` keys, most likely first.
    """
    old_key, new_key = 'default_pdf.correlated_background', 'default.correlated_background'
    try:
        import spey
        version = tuple(int(part) for part in str(spey.__version__).split('.')[:3])
    except Exception:
        return [old_key, new_key]
    return [old_key, new_key] if version <= (0, 2, 5) else [new_key, old_key]


def check_unknown_parameters(param_docs, logger):
    """Warn about parameter keys that the sampler does not recognise.

    A mistyped key (``SR_sigma'``, ``scan`` instead of ``scans``, ...) is
    silently ignored by the rest of the code, so the run quietly uses the
    default value instead. This walks every parameter document and reports
    anything that is neither a known scan parameter nor a structural key.

    Args:
        param_docs (list[dict]): Parameter documents to check (the merged
            global settings followed by the per-analysis documents).
        logger (logging.Logger): Logger used to emit the warnings.

    Returns:
        list[str]: The sorted list of unrecognised keys that were found.
    """
    from default_params import default_param_dict

    known = set(default_param_dict.keys()) | STRUCTURAL_PARAM_KEYS
    unknown = set()
    for doc in param_docs:
        if not isinstance(doc, dict):
            continue
        label = doc.get('analysis') or doc.get('name') or 'global settings'
        for key in doc.keys():
            if key not in known:
                unknown.add(key)
                logger.warning(f"Unknown parameter '{key}' in [{label}] - it will be IGNORED. Check for a typo.")
    return sorted(unknown)


def get_mask(shape, channels_and_bins, scan_SRs, scan_CRs, scan_VRs):
    """Build a per-bin boolean mask selecting whole region types.

    Args:
        shape (int): Total number of bins, i.e. the length of the mask.
        channels_and_bins (list[tuple]): ``(channel, type, n_bins)`` per
            channel, in bin order; ``type`` is ``'SR'``, ``'CR'`` or ``'VR'``.
        scan_SRs (bool): Include the bins of signal regions.
        scan_CRs (bool): Include the bins of control regions.
        scan_VRs (bool): Include the bins of validation regions.

    Returns:
        numpy.ndarray: Boolean array of length ``shape``, True for the bins
        whose region type was selected.
    """
    mask = np.empty(shape, dtype=bool)
    ii = 0
    for c, t, b in channels_and_bins:
        fill_with = False
        if (t=='SR' and scan_SRs) or (t=='CR' and scan_CRs) or (t=='VR' and scan_VRs):
            fill_with = True
        mask[ii:ii+b] = fill_with
        ii += b
    return mask 


def load_scan_limits(scan_limits, patchset_index, n_patchsets, input_bins_ordered,
                     bins_names, central_values, patchset_label, logger):
    """Load hardcoded scan limits for one patchset from the parameter card.

    ``scan_limits`` mirrors the structure of ``patchsets``: one entry per
    patchset, each either ``None`` (compute the limits as usual) or a list of
    ``[min, max]`` pairs, one per bin, **in the same channel order as the rest
    of the card** (i.e. ``input_bins_ordered``).

    The stored values are TOTAL yields - the same numbers the sampler prints in
    its "Scan limits" table and stores as ``lower_limits``/``upper_limits`` in
    the metadata - so they are converted back to signal offsets relative to
    ``central_values`` and returned in the MODEL's bin order.

    Args:
        scan_limits (list or None): The ``scan_limits`` parameter as read from
            the card.
        patchset_index (int): Index of the patchset currently being processed.
        n_patchsets (int): Total number of patchsets, for shape validation.
        input_bins_ordered (list[str]): Bin names in the card's channel order.
        bins_names (list[str]): Bin names in the model's channel order.
        central_values (numpy.ndarray): Central yields per bin, model order.
        patchset_label (str): Name of the patchset, for log messages.
        logger (logging.Logger): Logger for the mode and validation messages.

    Returns:
        tuple or None: ``(nSmin, nSmax)`` as signal offsets in the model's bin
        order, or ``None`` when this patchset has no hardcoded limits.

    Raises:
        ValueError: If the entry is malformed - wrong nesting, wrong number of
            bins, non-numeric values, ``min > max``, or a central value lying
            outside its own limits.
    """
    if scan_limits is None:
        return None
    if not isinstance(scan_limits, list):
        mes = f"'scan_limits' must be a list with one entry per patchset, got {type(scan_limits).__name__}!"
        logger.critical(mes)
        raise ValueError(mes)
    if len(scan_limits) != n_patchsets:
        mes = f"'scan_limits' has {len(scan_limits)} entries but there are {n_patchsets} patchsets! " \
              f"Use null for the patchsets whose limits should be computed."
        logger.critical(mes)
        raise ValueError(mes)

    entry = scan_limits[patchset_index]
    if entry is None:
        return None
    if not isinstance(entry, list):
        mes = f"'scan_limits' entry for {patchset_label} must be a list of [min, max] pairs " \
              f"or null, got {type(entry).__name__}!"
        logger.critical(mes)
        raise ValueError(mes)
    if len(entry) != len(input_bins_ordered):
        mes = f"'scan_limits' for {patchset_label} has {len(entry)} bins but the analysis has " \
              f"{len(input_bins_ordered)}! The list must follow the channel order used by the rest of the card."
        logger.critical(mes)
        raise ValueError(mes)

    limits_by_bin = {}
    for bin_name, pair in zip(input_bins_ordered, entry):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            mes = f"'scan_limits' for {patchset_label}, bin {bin_name}: expected a [min, max] pair, got {pair!r}!"
            logger.critical(mes)
            raise ValueError(mes)
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            mes = f"'scan_limits' for {patchset_label}, bin {bin_name}: values must be numbers, got {pair!r}!"
            logger.critical(mes)
            raise ValueError(mes)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            mes = f"'scan_limits' for {patchset_label}, bin {bin_name}: values must be finite, got {pair!r}!"
            logger.critical(mes)
            raise ValueError(mes)
        if lo > hi:
            mes = f"'scan_limits' for {patchset_label}, bin {bin_name}: min ({lo}) is above max ({hi})!"
            logger.critical(mes)
            raise ValueError(mes)
        limits_by_bin[bin_name] = (lo, hi)

    missing = [b for b in bins_names if b not in limits_by_bin]
    if missing:
        mes = f"'scan_limits' for {patchset_label} is missing these model bins: {missing[:5]}" \
              f"{' ...' if len(missing) > 5 else ''}"
        logger.critical(mes)
        raise ValueError(mes)

    # reorder from the card's channel order into the model's, then convert the
    # stored TOTAL yields into signal offsets around the central values
    lows = np.array([limits_by_bin[b][0] for b in bins_names], dtype=float)
    highs = np.array([limits_by_bin[b][1] for b in bins_names], dtype=float)

    outside = [(b, lo, cv, hi) for b, lo, cv, hi in zip(bins_names, lows, central_values, highs)
               if not (lo - 1e-6 <= cv <= hi + 1e-6)]
    if outside:
        b, lo, cv, hi = outside[0]
        mes = f"'scan_limits' for {patchset_label}: central value of bin {b} ({cv}) lies outside its " \
              f"limits [{lo}, {hi}] ({len(outside)} such bins). The limits are TOTAL yields, not signal offsets."
        logger.critical(mes)
        raise ValueError(mes)

    return np.round(lows - central_values, 4), np.round(highs - central_values, 4)


#: Tolerance for comparing a run's settings against the harvest context.
_CONTEXT_TOL = 1e-12


def check_scan_limits_context(context, param_dict, patchset_label, logger):
    """Decide whether hardcoded limits are still valid for this run's settings.

    Stored limits are only as good as the configuration they were probed with.
    Three settings invalidate them outright:

    * **signal uncertainty** - the probe injects the same ``histosys`` modifier
      the scan uses, so any change to ``sig_rel_unc`` can move the floor;
    * **a larger leakage spread** - a CR/VR bin's range is ``obs * spread``, so
      asking for a wider spread than was stored pushes the scan past the
      range that was actually verified;
    * **channel removal** (``removeCRsVRs`` / ``remove_channels``) - dropping a
      channel changes the likelihood itself, so a floor probed against the full
      model was never verified against the reduced one.

    A *smaller* spread is not unsafe - the stored box merely covers more than
    this run asked for - so it warns instead of forcing a recomputation. A
    region whose leakage is switched off is not checked at all: its bins are
    pinned on load and never move.

    Note the asymmetry between pinning and removal. Pinning keeps the channel in
    the workspace and freezes its bins, so the stored box still covers the
    run; removal builds a different model. That is why the harvest is always run
    with nothing removed, and why a run that removes channels recomputes.

    The check is HARD: limits are used only when every setting that matters for
    this run can be positively verified. A missing ``scan_limits_context``, or a
    context that does not record a setting this run depends on, means the limits
    are recomputed rather than trusted.

    Args:
        context (dict or None): ``scan_limits_context`` from the card. ``None``
            means the card predates the guard or was written by hand; there is
            nothing to verify against, so the limits are recomputed.
        param_dict (dict): This run's parameters.
        patchset_label (str): Patchset name, for the log messages.
        logger (logging.Logger): Logger for the verdict.

    Returns:
        bool: ``True`` if the stored limits may be used, ``False`` if they must
        be recomputed.
    """
    if context is None:
        logger.warning(f"Not using the card's scan limits for {patchset_label}: it has no "
                       f"'scan_limits_context', so they cannot be checked. Computing them "
                       f"now, which can be slow. See README.")
        return False
    if not isinstance(context, dict):
        mes = f"'scan_limits_context' must be a mapping, got {type(context).__name__}!"
        logger.critical(mes)
        raise ValueError(mes)

    reasons = []
    # Notes that only make sense if the limits actually get used. Collected
    # rather than logged straight away: saying 'these stay valid' and then
    # 'these do not apply' two lines later is worse than saying nothing.
    advisories = []

    stored_sig = context.get('sig_rel_unc')
    current_sig = param_dict['sig_rel_unc']
    if stored_sig is None:
        reasons.append('sig_rel_unc not recorded in the card')
    elif abs(float(stored_sig) - float(current_sig)) > _CONTEXT_TOL:
        reasons.append(f'sig_rel_unc {stored_sig}->{current_sig}')

    for region, leak_key, spread_key in (('CR', 'signal_leakage_CR', 'signal_leakage_CR_spread'),
                                         ('VR', 'signal_leakage_VR', 'signal_leakage_VR_spread')):
        if not param_dict[leak_key]:
            continue        # region is pinned on load, its spread is irrelevant
        stored = context.get(spread_key)
        current = param_dict[spread_key]
        if stored is None:
            # this run leaks signal into the region, so its spread matters and
            # an unrecorded one cannot be verified
            reasons.append(f'{region} spread not recorded in the card')
            continue
        if float(current) > float(stored) + _CONTEXT_TOL:
            reasons.append(f'{region} spread {stored}->{current}')
        elif float(current) < float(stored) - _CONTEXT_TOL:
            advisories.append(f'{region} spread is {current} here but {stored} in the card, so '
                              f'the card\'s limits are wider than this run needs. Harmless.')

    # Channel removal is REPORTED, not treated as invalidating. find_min_S takes
    # the raw workspace and never applies remove_channels - only ScanWrapper
    # does, during the scan - so the lower-limit search evaluates the full model
    # whatever the run removes. Recomputing on removal therefore reproduces the
    # stored numbers exactly (measured on 1911.06660: 2.9001/24.5 and 2.0001/17
    # either way) at the cost of a full probe, which is hours on a large
    # workspace. Worth saying out loud, because the scan itself DOES drop the
    # channels, so the floor is proven against a slightly different likelihood
    # than the one being sampled.
    run_flag = bool(param_dict.get('removeCRsVRs'))
    run_removed = sorted(param_dict.get('remove_channels') or [])
    stored_flag = bool(context.get('removeCRsVRs'))
    stored_removed = sorted(context.get('remove_channels') or [])
    if (run_flag or run_removed) and (stored_flag != run_flag or stored_removed != run_removed):
        n = len(run_removed)
        advisories.append(
            f"Using the card's limits although this run removes "
            f'{run_removed[0] if n == 1 else f"{n} channels"}. The lower-limit search always '
            f'uses the full model, so recomputing them here would give the same numbers; the '
            f'scan itself does drop the channels.')

    # low_lim_samples is a RESOLUTION, not a validity condition. The probe
    # bisects the candidate floor, so more samples can only find a floor at
    # least as low as fewer samples did - never an invalid one. A run asking
    # for more than was stored therefore gets a usable but coarser box, and
    # on the bins where the probe had to bisect at all that costs real negative
    # range. Worth saying out loud; not worth an hours-long recomputation.
    stored_lls = context.get('low_lim_samples')
    current_lls = param_dict.get('low_lim_samples')
    if (stored_lls is not None and current_lls is not None
            and int(current_lls) > int(stored_lls)):
        advisories.append(
            f"Using the card's limits, computed with low_lim_samples={stored_lls} while this run "
            f'asks for {current_lls}. A finer search would only widen a few bins slightly; set '
            f'low_lim_samples={stored_lls} to match the card.')

    if not reasons:
        for note in advisories:
            logger.warning(note)

    if reasons:
        logger.warning(f"Not using the card's scan limits for {patchset_label} (card->run: "
                       + ', '.join(reasons) + '). Computing them now, which can be slow. '
                       f'See README to store them in the card.')
        return False
    return True


def apply_region_pinning(nSmin, nSmax, channels_and_bins, signal_leakage_CR, signal_leakage_VR, logger):
    """Collapse the range of regions whose signal leakage is switched off.

    Hardcoded ``scan_limits`` are harvested with leakage enabled everywhere and
    nothing removed, so that one stored box stays valid whatever a later run
    decides to pin or drop. This re-imposes the current run's choices on the
    loaded box, exactly as :func:`get_scan_limits` would have: a region without
    leakage gets a range of +/-1e-10, so its bins never move off their central
    value.

    Args:
        nSmin (numpy.ndarray): Lower signal offsets per bin, model order.
        nSmax (numpy.ndarray): Upper signal offsets per bin, model order.
        channels_and_bins (list[tuple]): ``(channel, type, n_bins)`` tuples.
        signal_leakage_CR (bool): Whether CR bins may carry signal.
        signal_leakage_VR (bool): Whether VR bins may carry signal.
        logger (logging.Logger): Logger for the summary message.

    Returns:
        tuple: ``(nSmin, nSmax, n_pinned)`` with the pinned bins collapsed.
    """
    nSmin = np.array(nSmin, dtype=float, copy=True)
    nSmax = np.array(nSmax, dtype=float, copy=True)
    n_bins = nSmin.shape[0]
    pinned_types = []
    if not signal_leakage_CR:
        pinned_types.append('CR')
    if not signal_leakage_VR:
        pinned_types.append('VR')

    n_pinned = 0
    if pinned_types:
        bin_offset = 0
        for c, t, b in channels_and_bins:
            if t in pinned_types:
                nSmin[bin_offset:bin_offset + b] = -1e-10
                nSmax[bin_offset:bin_offset + b] = 1e-10
                n_pinned += b
            bin_offset += b
    if n_pinned:
        logger.info(f'Pinning {n_pinned} of {n_bins} loaded bins '
                    f'({"/".join(pinned_types)} leakage disabled for this run).')
    return nSmin, nSmax, n_pinned


def get_probe_mask(scan_mask, channels_and_bins, remove_channels):
    """Return the bins whose lower limit on S actually needs to be probed.

    A bin is worth probing only if the sampler will vary it. Two kinds are
    excluded:

    * **pinned** bins - signal leakage is disabled for their region, so
      :func:`~likelihood.calculate_sigmas` gives them a step size of 0 and
      they stay at their central value for the whole chain;
    * **removed** bins - their channel is dropped from the model before the
      likelihood is evaluated, so whatever is injected there is discarded.

    Both keep their incoming ``nSmin`` (essentially zero) instead of the
    probed ``nSmin + 1e-4``, which would otherwise put a positive floor under
    a bin that must hold exactly zero signal.

    Args:
        scan_mask (numpy.ndarray): Boolean per-bin mask from :func:`get_mask`;
            ``False`` marks a pinned bin.
        channels_and_bins (list[tuple]): ``(channel_name, channel_type, n_bins)``
            tuples describing the channel layout.
        remove_channels (list[str] or None): Channels dropped from the fit.

    Returns:
        numpy.ndarray: Boolean mask, ``True`` for bins that should be probed.
    """
    probe_mask = np.array(scan_mask, dtype=bool, copy=True)
    if remove_channels:
        bin_offset = 0
        for c, t, b in channels_and_bins:
            if c in remove_channels:
                probe_mask[bin_offset:bin_offset + b] = False
            bin_offset += b
    return probe_mask


def print_yield_table(bins_names, bins_is_signal, bkg_yields, bkg_unc, obs_yields, logger):
    """Log the per-bin background, uncertainty and observed count as a table.

    Args:
        bins_names (list[str]): Bin names, in bin order.
        bins_is_signal (list): Per-bin marker of whether the bin is in an SR.
        bkg_yields (list[tuple]): ``(bin_name, background)`` pairs.
        bkg_unc (list[tuple]): ``(bin_name, uncertainty)`` pairs.
        obs_yields (list[tuple]): ``(bin_name, observed)`` pairs.
        logger (logging.Logger): Logger the table is written to.

    Returns:
        str: The rendered table, for storing in the run metadata.
    """
    logger.info('Yields:')
    table = PrettyTable()
    table.add_column('BIN', bins_names, align='l', valign='t')
    table.add_column('SR', bins_is_signal, align='c', valign='t')
    table.add_column('BKG', [np.round(b,1) for a,b in bkg_yields], align='r', valign='t')
    table.add_column('∆BKG', [np.round(b,1) for a,b in bkg_unc], align='r', valign='t')
    table.add_column('OBS', [np.round(b,1) for a,b in obs_yields], align='r', valign='t')
    logger.info('\n'+table.get_string())
    return table.get_string()


def print_limit_table(bins_names, nSmin, nSmax, central_values, logger):
    """Log the scan box as a table of TOTAL yields.

    ``nSmin``/``nSmax`` are signal offsets; the table shows them added to the
    central value, which is what the card stores as ``scan_limits``.

    Args:
        bins_names (list[str]): Bin names, in bin order.
        nSmin (numpy.ndarray): Lower signal offset per bin.
        nSmax (numpy.ndarray): Upper signal offset per bin.
        central_values (numpy.ndarray): Central yield per bin.
        logger (logging.Logger): Logger the table is written to.

    Returns:
        str: The rendered table, for storing in the run metadata.
    """
    logger.info('Scan limits:')
    table = PrettyTable()
    table.add_column('BIN', bins_names, align='l', valign='t')
    table.add_column('MIN', np.round(central_values+nSmin, 1), align='r', valign='t')
    table.add_column('MAX', np.round(central_values+nSmax, 1), align='r', valign='t')
    logger.info('\n'+table.get_string())
    return table.get_string()


def get_time_string(ns_t):
    """Render a nanosecond duration as "H hours M minutes S seconds".

    Args:
        ns_t (int): Duration in nanoseconds.

    Returns:
        str: Human-readable duration.
    """
    s_t = ns_t//10**9
    m_t = s_t // 60
    h_t = m_t // 60
    left_s = s_t % 60
    left_m = m_t - 60*h_t
    return f"{h_t} hours {left_m} minutes {left_s} seconds"

def get_obs_signal(bkg_yields, obs_yields):
    """Return the observed-minus-expected yields, bin by bin.

    Args:
        bkg_yields (list[tuple]): ``(bin_name, value)`` pairs of background yields.
        obs_yields (list[tuple]): ``(bin_name, value)`` pairs of observed yields.

    Returns:
        list[tuple]: ``(bin_name, [obs - bkg, ...])`` pairs, in the order of
        ``bkg_yields``.
    """
    obs_signal_yields = []
    for bkg_name, bkg_values in bkg_yields:
        for obs_name, obs_values in obs_yields:
            if obs_name == bkg_name:
                if not isinstance(obs_values, list):
                    obs_values = [obs_values]
                    bkg_values = [bkg_values]
                entry = (obs_name, [ a-b for a,b in zip(obs_values,bkg_values) ] )
                obs_signal_yields.append(entry)
                break
    return obs_signal_yields

def get_scan_limits(bkg_yields, bkg_unc, obs_yields, channels_and_bins, signal_leakage_CR, signal_leakage_VR, CRs_scan_spread, VRs_scan_spread, CR_scan_sign, VR_scan_sign, CR_center_type, VR_center_type, logger):
    """Derive the per-bin scan box from the yields, in closed form.

    Signal regions get ``nsMax = obs - (B - 5*dB)`` (floored at 0) and
    ``nsMin = -B + dB``, falling back to ``obs - B`` or ``-B`` when that would
    be non-negative; ``dB`` is first clipped to ``3*sqrt(B)`` because an
    oversized uncertainty otherwise opens the box down to zero background.
    Control and validation regions ignore ``B`` entirely: they are centred on
    the observed count and range over ``+/- obs * spread``, or collapse to
    ``+/-1e-10`` when their signal leakage is switched off.

    The result is only a candidate lower bound - :func:`likelihood.find_min_S`
    checks whether the model can actually be evaluated there.

    Args:
        bkg_yields (list[tuple]): ``(bin_name, background)`` pairs, bin order.
        bkg_unc (list[tuple]): ``(bin_name, uncertainty)`` pairs, same order.
        obs_yields (list[tuple]): ``(bin_name, observed)`` pairs, same order.
        channels_and_bins (list[tuple]): ``(channel, type, n_bins)`` per channel.
        signal_leakage_CR (bool): Allow signal in control regions.
        signal_leakage_VR (bool): Allow signal in validation regions.
        CRs_scan_spread (float): CR range as a fraction of the observed count.
        VRs_scan_spread (float): VR range as a fraction of the observed count.
        CR_scan_sign (str): ``'both'``, ``'positive'`` or ``'negative'``.
        VR_scan_sign (str): ``'both'``, ``'positive'`` or ``'negative'``.
        CR_center_type (str): ``'obs'`` or ``'exp'``; steers a warning only.
        VR_center_type (str): ``'obs'`` or ``'exp'``; steers a warning only.
        logger (logging.Logger): Logger for the clipping and spread warnings.

    Returns:
        tuple: ``(nSmin, nSmax, central_values)`` - the signal offsets per bin,
        rounded to 4 decimals, and the central yield each is measured from
        (``B`` for an SR bin, the observed count otherwise).

    Raises:
        ValueError: If the bin order differs between B, dB and obs.
    """
    B, deltaB, obs = get2nd(bkg_yields), get2nd(bkg_unc), get2nd(obs_yields)

    # safety check
    for a,b,c in zip(bkg_yields, bkg_unc, obs_yields):
        if a[0]!=b[0] or a[0]!=c[0]:
            mes =  '[ERROR] order of channels is mixed between B, ∆B and Obs!'
            logger.critical(mes)
            raise ValueError(mes)

    # sometime the fitting procedure fails and returns too large blg unc 
    uncertainty_too_big = deltaB > 3 * np.sqrt(B)
    all_bins_names = [f"{c}-{b}" for c, _, binN in channels_and_bins for b in range(binN) ]
    if np.sum(uncertainty_too_big) > 0:
        bins_names = [name for name, too_big in zip(all_bins_names, uncertainty_too_big) if too_big]
        logger.warning(f'Background uncertainty seems too large. For getting the scan limits, I will clip it to 3√B for {bins_names}.')
        deltaB[uncertainty_too_big] = 3*np.sqrt(B[uncertainty_too_big])
    
    # calculate upper and lower limits
    nsMax = obs - (B-5*deltaB)
    nsMax = np.where(nsMax > 0.0, nsMax, 0.0)
    
    nsMin = -B+deltaB
    nsMin = np.where( nsMin > obs-B, obs-B, nsMin )
    nsMin = np.where(nsMin < 0.0, nsMin, -B)

    if CR_center_type == 'obs':
        nSobs_abs = np.abs(obs-B) # get the observed signal
    elif CR_center_type == 'exp':
        nSobs_abs = np.zeros(obs.shape) # use 0 signal
    else:
        raise ValueError(f"Central values for CRs should be either 'exp' or 'obs' but '{CR_center_type}' provided")

    CRmask = get_mask(len(B), channels_and_bins, False, True, False)
    if signal_leakage_CR:
        deltaS_CR = obs[CRmask]*CRs_scan_spread
        fluctuations = deltaS_CR - nSobs_abs[CRmask]
        if np.sum(fluctuations < 0.0, dtype=int) > 0:
            logger.warning('Fluctuations in the CRs are too small to account for the difference between expected and observed yields!')
            low_index = np.argmin(fluctuations)
            low_fluct_val = np.abs(nSobs_abs[CRmask][low_index]/obs[CRmask][low_index]) if obs[CRmask][low_index] > 0.0 else 0.0
            # low_index indexes the CR-masked arrays, so the name has to be looked
            # up in the CR-masked list of bin names, not in the full one.
            CR_bins_names = [name for name, keep in zip(all_bins_names, CRmask) if keep]
            logger.warning(f'Consider increasing the "signal_leakage_CR_spread" to at least {low_fluct_val} (based on {CR_bins_names[low_index]}).')
        # set the limits
        if CR_scan_sign == 'both':
            nsMax[CRmask] = deltaS_CR
            nsMin[CRmask] = -1.0*deltaS_CR  
        elif CR_scan_sign == 'positive':
            nsMax[CRmask] = deltaS_CR
            nsMin[CRmask] = -1e-10
        elif CR_scan_sign == 'negative':
            nsMax[CRmask] = 1e-10
            nsMin[CRmask] = -1.0*deltaS_CR
        else:
            mes =  f'[ERROR] Wrong value for the signal_leakage_CR_sign parameter: {CR_scan_sign}'
            logger.critical(mes)
            raise ValueError(mes)
    else:
        nsMax[CRmask] = 1e-10
        nsMin[CRmask] = -1e-10

    VRmask = get_mask(len(B), channels_and_bins, False, False, True)
    if signal_leakage_VR:
        # overwrite
        if VR_center_type == 'obs':
            nSobs_abs = np.abs(obs-B) # get the observed signal
        elif VR_center_type == 'exp':
            nSobs_abs = np.zeros(obs.shape) # use 0 signal
        else:
            raise ValueError(f"Central values for VRs should be either 'exp' or 'obs' but '{VR_center_type}' provided")

        deltaS_VR = obs[VRmask]*VRs_scan_spread
        fluctuations = deltaS_VR - nSobs_abs[VRmask]
        if np.sum(fluctuations < 0.0, dtype=int) > 0:
            logger.warning('Fluctuations in the VRs are too small to account for the difference between expected and observed yields!')
            low_index = np.argmin(fluctuations)
            low_fluct_val = np.abs(nSobs_abs[VRmask][low_index]/obs[VRmask][low_index]) if obs[VRmask][low_index] > 0.0 else 0.0
            # low_index indexes the VR-masked arrays, so the name has to be looked
            # up in the VR-masked list of bin names, not in the full one.
            VR_bins_names = [name for name, keep in zip(all_bins_names, VRmask) if keep]
            logger.warning(f'Consider increasing the "signal_leakage_VR_spread" to at least {low_fluct_val} (based on {VR_bins_names[low_index]}).')
        # set the limits
        if VR_scan_sign == 'both':
            nsMax[VRmask] = deltaS_VR
            nsMin[VRmask] = -deltaS_VR
        elif VR_scan_sign == 'positive':
            nsMax[VRmask] = deltaS_VR
            nsMin[VRmask] = -1e-10
        elif VR_scan_sign == 'negative':
            nsMax[VRmask] = 1e-10
            nsMin[VRmask] = -deltaS_VR
        else:
            mes =  f'[ERROR] Wrong value for the signal_leakage_VR_sign parameter: {VR_scan_sign}'
            logger.critical(mes)
            raise ValueError(mes)
    else:
        nsMax[VRmask] = 1e-10
        nsMin[VRmask] = -1e-10
        
    SRmask = get_mask(len(B), channels_and_bins, True, False, False)
    CRandVRmask = CRmask + VRmask
    central_values = np.empty(len(SRmask))
    for cc in range(len(central_values)):
        if SRmask[cc]:
            central_values[cc] = B[cc]
        elif CRandVRmask[cc]:
             central_values[cc] = obs[cc]
        else:
            mes =  f'[ERROR] wrong mask on position {cc}!'
            logger.critical(mes)
            raise ValueError(mes)

    assert central_values.shape[0] == nsMax.shape[0]
    return np.round(nsMin,4), np.round(nsMax,4), central_values


def find_mu_limits(nSmin, nSmax, central_values, logger):
    """Estimate the mu_SIG range the scan box can support.

    Only bins that can actually move constrain mu. A pinned bin - CR/VR
    leakage switched off, so its range is the +/-1e-10 stub - carries no
    information about how far mu may go, but it dominates both expressions
    below if it is left in: its offset cancels the 1e-10 guard term, so the
    lower bound divides by zero and the upper bound collapses to 0.5. Such
    bins are therefore excluded rather than guarded against.
    """
    nSmin = np.asarray(nSmin, dtype=float)
    nSmax = np.asarray(nSmax, dtype=float)
    central_values = np.asarray(central_values, dtype=float)

    # the +/-1e-10 stub a pinned bin carries is orders of magnitude below any
    # real limit, so this separates pinned bins from bins with genuine room
    MOVES = 1e-8
    can_go_down = nSmin < -MOVES
    can_go_up = nSmax > MOVES
    n_pinned = int(np.sum(~(can_go_down | can_go_up)))
    if n_pinned:
        logger.debug(f'{n_pinned} pinned bins excluded from the mu bounds estimate.')

    if np.any(can_go_down):
        # likewise verbatim: central/(nSmin + 1e-10) over the bins that move
        mu_min = np.max(central_values[can_go_down]/(nSmin[can_go_down] + 1e-10))
    else:
        logger.warning('No bin can take a negative signal, so mu has no lower bound to '
                       'estimate. Falling back to 0.')
        mu_min = 0.0

    if np.any(can_go_up):
        # the original expression, kept verbatim (1e-10 guard included) so that
        # a run without pinned bins gets exactly the number it got before
        up = nSmax[can_go_up]
        cen = central_values[can_go_up]
        x = (up + cen)/(up + 1e-10)
        mu_max = np.min(x - cen/(up + 1e-10))
    else:
        logger.warning('No bin can take a positive signal, so mu has no upper bound to '
                       'estimate. Falling back to 1.')
        mu_max = 1.0
    logger.debug(f'Mu bounds initial estimate: ({mu_min}, {mu_max})')
    logger.debug(f'Central values: {central_values}')
    logger.debug(f'nSmin: {nSmin}')
    logger.debug(f'Central nSmax: {nSmax}')
    if mu_min >= mu_max:
        logger.warning(f'mu_min ({mu_min}) >= mu_max ({mu_max}) ! Setting initial limits to (0, 1).')
        mu_min = 0
        mu_max = 1
    return mu_min, mu_max


def generate_starting_points(nsMin, nsMax, central_values, mask, n=1, start_method='default', channels_and_bins=None, logger=None, starting_points_file=None, starting_points_file_index=None 
):
    """Produce the initial MCMC states, one per chain.

    Args:
        nsMin (numpy.ndarray): Lower signal offset per bin.
        nsMax (numpy.ndarray): Upper signal offset per bin.
        central_values (numpy.ndarray): Central yield per bin, used to convert
            an external file of TOTAL yields into signal offsets.
        mask (numpy.ndarray): Boolean per bin; masked-out bins start at 0.
        n (int, optional): Number of starting points. Defaults to 1.
        start_method (str, optional): ``'random'`` (uniform in the box),
            ``'gauss'`` (narrow normal around zero signal), ``'edges'``
            (a corner of the box) or ``'default'`` (zero signal).
        channels_and_bins (list[tuple], optional): ``(channel, type, n_bins)``
            per channel; needed to name the columns of an external file.
        logger (logging.Logger, optional): Logger; one is created if omitted.
        starting_points_file (str, optional): CSV of TOTAL yields to start from
            instead of generating points.
        starting_points_file_index (int, optional): Row of that file to use.

    Returns:
        numpy.ndarray: Array of shape ``(n, len(nsMin))`` of signal offsets.
    """
    #
    # some useful functions
    #
    def populate_randomly(n, mask, nsMin, nsMax):
        """Draw ``n`` points uniformly inside the box; masked bins stay at 0."""
        random_points = np.empty(shape=(n, len(nsMin)))
        for pp in range(n):
            random_points[pp] = np.array([np.random.uniform(nsMin[j],nsMax[j]) if mask[j] else 0.0 for j in range(len(nsMax))])
        return random_points

    def populate_gauss(n, mask, nsMin, nsMax):
        """Draw ``n`` points from a narrow normal at zero signal, clipped at nsMin."""
        centers = np.zeros(len(mask))
        sigmas = 0.01 * (nsMax-nsMin)
        sigmas = np.where(sigmas > 1.0, 1.0, sigmas)
        cov = np.diag(sigmas**2)        
        random_points = np.random.multivariate_normal(centers, cov, size=n)
        random_points = np.clip(random_points, nsMin, np.inf)
        random_points[:, ~mask] = 0.0
        return random_points

    def populate_with_edges(n, mask, nsMin, nsMax):
        """Draw ``n`` points on the corners of the box; masked bins stay at 0."""
        edge_points = np.empty(shape=(n, len(nsMin)))
        for pp in range(n):
            edge_points[pp] = np.array([np.random.choice([nsMin[j],nsMax[j]]) if mask[j] else 0.0 for j in range(len(nsMax))])
        return edge_points
    #
    # the actual generation
    #
    if logger is None:
        logger = setup_logger()
    
    if starting_points_file is not None:
        try:
            logger.warning('Using external file for starting points. Make sure it contains TOTAL yields.')
            logger.info(f'Attempting to read starting points from {starting_points_file}')
            table = pd.read_csv(starting_points_file)

            # 1 if pandas only saw one column and it contains ";" in its name, re-read assuming ";" is the separator
            if len(table.columns) == 1 and ";" in table.columns[0]:
                table = pd.read_csv(starting_points_file, sep=";")

            if starting_points_file_index is not None:
                idx = int(starting_points_file_index)
                sub = table.iloc[idx, :]
                # if you grabbed a single row (Series), turn it back into a 1×N DataFrame
                if isinstance(sub, pd.Series):
                    table = sub.to_frame().T
                else:
                    table = sub
            
            expected_cols = [ f"{channel}-{b}" for channel, sr, bins in channels_and_bins for b in range(bins)]

            # check for missing
            missing = set(expected_cols) - set(table.columns)
            if missing:
                logger.error(f"Missing columns in starting‐points file: {sorted(missing)}. I will use some default values for these.")
                for ii, colname in enumerate(expected_cols):
                    if colname in missing:
                        table.insert(ii, colname, central_values[ii], allow_duplicates=False)
                new_missing = set(expected_cols) - set(table.columns)
                if new_missing:
                    raise ValueError(f'Failure to add default values for columns: {new_missing}')

            # drop extras and reorder
            table = table[expected_cols]

            # extract raw numpy (no index, no names)
            points = table.to_numpy(dtype=float) - central_values
            logger.info('Starting points read!')
            return points

        except FileNotFoundError:
            raise FileNotFoundError('The file with starting points does not exist!')
        except Exception as e:
            raise e

    elif start_method == 'default':
        points = np.array([list(nsMin), list(nsMax), np.zeros(nsMin.shape)]) #nsMin is negative!
        points[:, ~mask] = 0.0 # do not start off-centre in regions that are not scanned
        if n > 3:
            random_points = populate_randomly(n-3, mask, nsMin, nsMax)
            points = np.concatenate([points, random_points], axis=0)
            assert len(points) == n
            return points
        else:
            return np.array(points[:n])
    elif start_method == 'random':
        points = populate_randomly(n, mask, nsMin, nsMax)
        assert len(points) == n
        return points 
    elif start_method == 'fine-tune':
        points = populate_gauss(n, mask, nsMin, nsMax)
        assert len(points) == n
        return points 
    elif start_method == 'edges':
        # plain min(), so the 2**nbins term does not overflow numpy's integer types
        maxiter = min(n, 2**len(nsMin))
        types_of_channels = set([ em[1] for em in channels_and_bins])
        if 'SR' not in types_of_channels:
            mes =  f'[ERROR] No signal regions provided! ({types_of_channels})'
            logger.critical(mes)
            raise ValueError(mes)
        else:
            edge_points = populate_with_edges(maxiter, mask, nsMin, nsMax)
            if n <= maxiter:
                assert len(edge_points) == n
                return edge_points
            else:
                # more points requested than distinct edges available: top up with random ones
                generate_n = n - maxiter
                random_points = populate_randomly(generate_n, mask, nsMin, nsMax)
                points = np.concatenate([edge_points, random_points], axis=0)
                assert len(points) == n
                return points
    else:
        mes =  f"[ERROR] Wrong start method ({start_method}). I will use the 'default' option."
        logger.error(mes)
        return generate_starting_points(nsMin, nsMax, central_values, mask, n=n, start_method='default',
                                        channels_and_bins=channels_and_bins, logger=logger)


def find_placeholder_rows(data):
    """Flag rows whose likelihood columns hold the NaN/inf placeholder.

    When a likelihood comes back NaN or infinite,
    :meth:`~likelihood.LikelihoodCalculatorWrapper.check_for_nan` writes
    ``+/-NAN_PLACEHOLDER`` instead. Such a row carries no usable likelihood, and
    left in place it would also poison the per-column min/max recorded in the
    metadata.

    Args:
        data (numpy.ndarray): Result rows, ``(n_rows, n_bins + 8)``.

    Returns:
        numpy.ndarray: Boolean mask, ``True`` for rows to discard.
    """
    from likelihood import NAN_PLACEHOLDER, N_LIKELIHOOD_COLUMNS
    if data.size == 0:
        return np.zeros(len(data), dtype=bool)
    likelihoods = data[:, -N_LIKELIHOOD_COLUMNS:]
    # compare with a little slack: the value survives a round trip through the
    # CSV, but exact float equality is a poor thing to rely on
    return np.any(np.abs(likelihoods) >= NAN_PLACEHOLDER * (1.0 - 1e-9), axis=1)


def merge_results(infiles, keep_files=True, suffix='', logger=None):
    """Concatenate the per-scan CSV chunks into one result file.

    Rows carrying the NaN placeholder that marks a failed likelihood are
    dropped, so the merged file contains only usable points.

    Args:
        infiles (list[str]): Per-scan CSV files to merge.
        keep_files (bool, optional): Keep the inputs after merging.
            Defaults to True.
        suffix (str, optional): Extra tag in the output filename.
        logger (logging.Logger, optional): Logger; one is created if omitted.

    Returns:
        tuple: ``(outpath, n_rows, min_values, max_values)`` - the merged file,
        how many rows survived, and the per-column extrema for the metadata.

    Raises:
        PermissionError: If the output file cannot be created.
    """
    if logger is None:
        logger = setup_logger()
    logger.info(f'Merging results of {len(infiles)} scans.')
    try:
        ii = 0
        while True:
            dirpath = join(dirname(infiles[0]), "results-{}.csv".format(ii))
            if not isfile(dirpath):
                break
            else:
                ii += 1
    except PermissionError:
        # cannot create directory
        mes = '[ERROR] Cannot create output file with scan result. Permission denied.'
        logger.critical(mes)
        raise PermissionError(mes)
    else:
        if len(suffix) > 0:
            outpath = join(dirname(infiles[0]), "results-{}-{}.csv".format(suffix, ii))
        else:
            outpath = join(dirname(infiles[0]), "results-{}.csv".format(ii))
        min_values = []
        max_values = []
        header = ''
        n_total = 0
        n_dropped = 0
        n_written = 0
        with open(outpath, 'a') as fout:
            with open(infiles[0], 'r') as fin:
                header = fin.readline()
                fout.write(header)
            for ff, fpath in enumerate(infiles):
                with open(fpath, 'r') as fin:
                    loaded_data = np.loadtxt(fin, float, skiprows=1, delimiter=',')
                    if len(loaded_data.shape) == 1:
                        loaded_data = np.reshape(loaded_data, (1, loaded_data.shape[0]))
                    n_total += len(loaded_data)

                    # A row whose likelihood came back NaN/inf carries the
                    # placeholder instead. Drop it BEFORE the min/max are taken,
                    # otherwise the placeholder becomes the recorded maximum.
                    bad = find_placeholder_rows(loaded_data)
                    if bad.any():
                        n_dropped += int(bad.sum())
                        loaded_data = loaded_data[~bad]

                    if len(loaded_data) == 0:
                        continue
                    min_values.append(np.amin(loaded_data, axis=0))
                    max_values.append(np.amax(loaded_data, axis=0))
                    np.savetxt(fout, loaded_data, fmt="%+010.8f", delimiter=',')
                    n_written += len(loaded_data)
                if not keep_files:
                    try:
                        os.remove(fpath)
                    except FileNotFoundError:
                        logger.error(f"File {basename(fpath)} cannot be deleted because it doesn't exist!.")

        if n_dropped:
            logger.warning(f'Dropped {n_dropped} of {n_total} rows whose likelihood was NaN or '
                           f'infinite (written as the +/-1e10 placeholder). {n_written} rows kept.')
        else:
            logger.info(f'{n_written} rows written, none carried a NaN placeholder.')

        if n_written == 0:
            names = [c.strip() for c in header.strip().split(',')][-8:]
            logger.critical(
                f'NO USABLE ROWS LEFT: all {n_total} sampled points had a NaN or infinite '
                f'likelihood and were dropped, so {basename(outpath)} contains only its header. '
                f'The affected columns are {names}. This usually means the scan explored yields '
                f'where the model is undefined - check the scan limits (a lower limit that is too '
                f'negative), the "low_lim_samples" probe, and the starting points. Per-column '
                f'min/max cannot be computed and are left empty in the metadata.')
            return outpath, [], []

        min_values = np.array(min_values)
        max_values = np.array(max_values)
        return outpath, np.amin(min_values, axis=0).tolist(), np.amax(max_values, axis=0).tolist()


def create_metadata(param_dict, bkg_yields, bkg_unc, obs_yields, lower_limits, upper_limits, minS_orig, logger):
    """Build the metadata dictionary saved alongside a results file.

    NOTE: the MCMC starting points are deliberately NOT stored. They are one
    arbitrary draw per chain, they are reproducible from the recorded seed and
    start_method, and for a wide scan they dominate the size of the file.
    """
    metadata = copy.deepcopy(param_dict)
    if param_dict['analysis'] in analysis_name_dict.keys():
        metadata['analysis_altname'] = analysis_name_dict[param_dict['analysis']]
    elif param_dict['analysis'].split('-')[0] in analysis_name_dict.keys():
        metadata['analysis_altname'] = analysis_name_dict[param_dict['analysis'].split('-')[0]]
    else:
        logger.warning(f"Could not find alternative name for {param_dict['analysis']}")
        metadata['analysis_altname'] = ''
    metadata['bkg_yields'] = bkg_yields
    metadata['bkg_unc'] = bkg_unc
    metadata['obs_yields'] = obs_yields
    metadata['lower_limits'] = list(lower_limits)
    metadata['upper_limits'] = list(upper_limits)
    metadata['initial_lower_limits'] = list(minS_orig)
    return metadata


def update_metadata(metadata, data_min, data_max, nLL_max):  
    """Record the per-column extrema of a finished scan in its metadata.

    The last 8 columns of a result row are the likelihood values and the rest
    are bin yields, so the extrema are split accordingly.

    Args:
        metadata (dict): Run metadata, updated in place.
        data_min (array-like): Per-column minimum over the merged results.
        data_max (array-like): Per-column maximum over the merged results.
        nLL_max (list or None): Four lists - expected, observed, asimov
            expected and asimov observed - of the per-scan maximal nLL.

    Returns:
        dict: The same ``metadata`` object.

    Raises:
        ValueError: If ``nLL_max`` is given but does not hold 4 entries.
    """
    metadata['x_min'] = data_min[:-8]
    metadata['y_min'] = data_min[-8:]
    metadata['x_max'] = data_max[:-8]
    metadata['y_max'] = data_max[-8:]
    metadata['nLL_exp_max'] = []
    metadata['nLL_obs_max'] = []
    metadata['nLLA_exp_max'] = []
    metadata['nLLA_obs_max'] = []
    if nLL_max is not None:
        if len(nLL_max) != 4:
            raise ValueError('nLL_max should be a list of 4 values!')
        metadata['nLL_exp_max'] = [float(em) if em is not None else None for em in nLL_max[0]]
        metadata['nLL_obs_max'] = [float(em) if em is not None else None for em in nLL_max[1]]
        metadata['nLLA_exp_max'] = [float(em) if em is not None else None for em in nLL_max[2]]
        metadata['nLLA_obs_max'] = [float(em) if em is not None else None for em in nLL_max[3]]
    return metadata

