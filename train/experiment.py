import numpy as np
import torch
import torch.nn as nn

import os
import time
from omegaconf import OmegaConf, open_dict

from base_experiment import BaseExperiment
from dataset import nLLsDataset
from preprocessing import (
    preprocess_features,
    preprocess_nLLs,
    undo_preprocess_nLLs,
)
from plots import plot_mixer
from logger import LOGGER
from mlflow_util import log_mlflow
from losses import LogCoshLoss, RelL1Loss, HeteroscedasticLoss, MAEToHetLoss

def log_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    LOGGER.info(
        f"[{tag}] RSS={mem_info.rss/1e9:.2f} GB, VMS={mem_info.vms/1e9:.2f} GB, "
        f"System available={psutil.virtual_memory().available/1e9:.2f} GB"
    )

def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        LOGGER.info(f"[{tag}] GPU allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

TYPE_TOKEN_DICT = {
    '1911.06660-leakage-10_': [0],
    '1911.06660-leakage-10__cleaned': [0],
    '1909.09226-leakage-10_': [0],
    '1908.08215-400k-fluct20_': [0],
    "1911.12606-EWKinos-1M-z4-nll400-delta200": [0],
    "1911.12606-EWKinos-1M-z4-nll400-delta200-obs60plus": [0],
    "1911.12606-EWKinos-1M-z4-nll400-delta200-obs40cut": [0],
    "1911.12606-EWKinos-1M-z4-nll400-delta200-exp60plus": [0],
    "1911.12606-sleptons-700k-fluct30%-nll300--delta300-z3": [0],   
    "1911.12606-sleptons-200k-fluct20_": [0],   
    '2106.01676-offshell-higgsino-300k-fluct20_': [0],
    '2106.01676-offshell-winobino-minus-300k-fluct20_': [0],
    '2106.01676-offshell-winobino-plus-fluct20_-300k': [0],
    '2106.01676-onshell-winobino-fluct25_-300k': [0],
    '2106.01676-offshell-TChiWZoff-jsons2': [0],
    '2106.01676-winobino-minus-jsons2-rehearsal': [0],
    '2106.01676-winobino-minus-tchiwzoff-retrain': [0],
    "1911.12606-EWKinos-1M-fluct100_new-trim": [0],
    "1911.12606-sleptons-scan201816": [0],
    "1911.12606-EWKinos-scan201816": [0],
    "1911.12606-EWKinos-158k-boundaryscan": [0],
    "1911.12606-sleptons-153k-boundaryscan": [0],
    "aag": [0, 0, 1, 1, 0],
    "aagg": [0, 0, 1, 1, 0, 0],
    "zg": [0, 0, 1, 2],
    "zgg": [0, 0, 1, 2, 2],
    "zggg": [0, 0, 1, 2, 2, 2],
    "zgggg": [0, 0, 1, 2, 2, 2, 2],
    "zggggg": [0, 0, 1, 2, 2, 2, 2, 2],
    "wz": [0, 0, 1, 2],
}
DATASET_TITLE_DICT = {
    '1911.06660-leakage-10_': r"1911.06660",
    '1911.06660-leakage-10__cleaned': r"1911.06660",
    "1911.12606-EWKinos-1M-z4-nll400-delta200": r"1911.12606 EWKinos",
    "1911.12606-EWKinos-1M-z4-nll400-delta200-obs60plus": r"1911.12606 EWKinos (obs $\Delta>60$)",
    "1911.12606-EWKinos-1M-z4-nll400-delta200-obs40cut": r"1911.12606 EWKinos (obs $\Delta\leq40$)",
    "1911.12606-EWKinos-1M-z4-nll400-delta200-exp60plus": r"1911.12606 EWKinos (exp $\Delta>60$)",
    "1911.12606-sleptons-700k-fluct30%-nll300--delta300-z3": r"1911.12606 Sleptons",
    '1911.12606-sleptons-200k-fluct20_': r"1911.12606 Sleptons 200k",
    "1911.12606-sleptons-scan201816": r"1911.12606 Sleptons (2018-16 scan)",
    "1911.12606-EWKinos-scan201816": r"1911.12606 EWKinos (2018-16 scan)",
    "1911.12606-EWKinos-158k-boundaryscan": r"1911.12606 EWKinos (boundary scan)",
    "1911.12606-sleptons-153k-boundaryscan": r"1911.12606 Sleptons (boundary scan)",
    '2106.01676-offshell-higgsino-300k-fluct20_': r"2106.01676 Offshell Higgsinos",
    '2106.01676-offshell-winobino-minus-300k-fluct20_': r"2106.01676 Winobino Minus",
    '2106.01676-offshell-TChiWZoff-jsons2': r"2106.01676 TChiWZoff",
    '2106.01676-winobino-minus-jsons2-rehearsal': r"2106.01676 Winobino Minus + TChiWZoff (rehearsal)",
    '2106.01676-winobino-minus-tchiwzoff-retrain': r"2106.01676 Winobino Minus + TChiWZoff",
    '2106.01676-offshell-winobino-plus-fluct20_-300k': r"2106.01676 Winobino Plus",
    '2106.01676-onshell-winobino-fluct25_-300k': r"2106.01676 Onshell Winobino",
    '1909.09226-leakage-10_': r"1909.09226",
    '1908.08215-400k-fluct20_': r"1908.08215",
    "aag": r"$gg\to\gamma\gamma g$",
    "aag_inv": r"$gg\to\gamma\gamma g$",
    "aag_inv_naiv": r"$gg\to\gamma\gamma g$",
    "aag_inv_all": r"$gg\to\gamma\gamma g$",
    "aagg": r"$gg\to\gamma\gamma gg$",
    "zg": r"$q\bar q\to Zg$",
    "zgg": r"$q\bar q\to Zgg$",
    "zgg_inv": r"$q\bar q\to Zgg$",
    "zgg_pinv": r"$q\bar q\to Zgg$",
    "zggg": r"$q\bar q\to Zggg$",
    "zgggg": r"$q\bar q\to Zgggg$",
    "zgggg_sorted": r"$q\bar q\to Zgggg$",
    "zggggg": r"$q\bar q \to Zggggg$",
    "wz": r"$q\bar q \to WZ$",
}
MODEL_TITLE_DICT = {
    "Transformer": "Tr",
    "MLP": "MLP",
    "MuMLP": "μP MLP",
    "FV_MLP": "FV MLP",
    "DSI": "DSI",
    "LGATr": "LGATr",
}


class nLLsExperiment(BaseExperiment):
    def init_physics(self):
        assert (
            not self.cfg.training.force_xformers
        ), "nLLs experiment assumes default torch attention"
        self.n_datasets = len(self.cfg.data.dataset)

        # create type_token list
        self.type_token = []
        for dataset in self.cfg.data.dataset:
            if self.cfg.data.include_permsym:
                self.type_token.append(TYPE_TOKEN_DICT[dataset])
            else:
                self.type_token.append(list(range(len(TYPE_TOKEN_DICT[dataset]))))

        token_size = max(
            [max([max(token) for token in self.type_token]) + 1, self.n_datasets]
        )
        OmegaConf.set_struct(self.cfg, True)
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        if modelname in ["GAP", "MLP", "DSI"]:
            assert len(self.cfg.data.dataset) == 1, (
                f"Architecture {modelname} can not handle several datasets "
                f"as specified in {self.cfg.data.dataset}"
            )

        with open_dict(self.cfg):
            if modelname == "LGATr":
                  self.cfg.model.net.in_s_channels = token_size
                  self.cfg.model.token_size = token_size
            self.cfg.model.net.type_token_list = TYPE_TOKEN_DICT[
                self.cfg.data.dataset[0]
            ]
            assert (
                len(np.unique(self.cfg.model.net.type_token_list))
                == max(self.cfg.model.net.type_token_list) + 1
            ), f"Invalid type_token_list={self.cfg.model.net.type_token_list}"

    def init_data(self):
        LOGGER.info(
            f"Working with dataset {self.cfg.data.dataset} "
            f"and type_token={self.type_token}"
        )

        # load all datasets and organize them in lists
        (
            self.features,
            self.nLLs,
            self.features_prepd,
            self.nLLs_prepd,
            self.prepd_mean,
            self.prepd_mean_features,
            self.prepd_std,
            self.prepd_std_features,
            self.prepd_nll_bounds,
            self.props,
        ) = ([], [], [], [], [], [], [], [], [], [])
        self.presplit = []
        # per-dataset mu=0 baselines, one per output; see init_data
        self.nLL_mu0 = []

        for dataset in self.cfg.data.dataset:
            # load data
            data_path = os.path.join(self.cfg.data.data_path, f"{dataset}.npy")
            data_path_test = os.path.join(self.cfg.data.data_path, f"{dataset}_test.npy")
            data_path_val = os.path.join(self.cfg.data.data_path, f"{dataset}_val.npy")

            print("cwd:", os.getcwd())
            print("configured path:", self.cfg.data.data_path)
            print("absolute path:", os.path.abspath(self.cfg.data.data_path))
            print("exists:", os.path.exists(self.cfg.data.data_path))

            assert os.path.exists(self.cfg.data.data_path), f"data_path {self.cfg.data.data_path} does not exist"
            assert os.path.exists(data_path), f"data_path {data_path} does not exist"
            if os.path.exists(data_path_val):
                data_train_raw = np.load(data_path, allow_pickle=True)
                data_val_raw = np.load(data_path_val, allow_pickle=True)
                data_test_raw = np.load(data_path_test, allow_pickle=True)
                data_raw = np.concatenate([data_train_raw, data_val_raw, data_test_raw], axis=0)
                # The split is FIXED by the files: rows stay in train|val|test order
                # and must not be shuffled across that boundary. Needed whenever the
                # train set oversamples a region (duplicated rows would otherwise leak
                # into val/test) or when val/test hold out whole physics points.
                presplit = (len(data_train_raw), len(data_val_raw), len(data_test_raw))
            else:
                data_raw = np.load(data_path, allow_pickle=True)
                presplit = None

            LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

            if presplit is None:
                # shuffle for reproducibility
                np.random.seed(1234)
                np.random.shuffle(data_raw)
            else:
                LOGGER.info(
                    f"Using the on-disk train/val/test split {presplit} "
                    f"(no shuffle: it would mix the splits)"
                )
                # shuffle *within* the train block only
                rng = np.random.default_rng(1234)
                data_raw[: presplit[0]] = data_raw[rng.permutation(presplit[0])]
            self.presplit.append(presplit)

            # bring data into correct shape
            features = data_raw[:, :-8]
            nLLs = data_raw[:, -8:] 

            # Keep the mu=0 baselines before they are subtracted away. The network
            # regresses the delta, but the *physical* quantity is the absolute nLL
            # at mu=1, = baseline + delta. A relative error taken against that
            # never divides by ~0, whereas the delta is exactly 0 on every
            # signal-free scan row. Baselines are constant per dataset (they depend
            # only on the analysis, not the point), so the median is exact for the
            # scans and within ~1e-6 relative for the older fluctuated sets.
            nLL_mu0 = np.median(nLLs[:, 0::2], axis=0)

            # process nLLs: subtract baseline, keep differences
            for i in range(4):
                print(f'nLLs mu0 {i} range: {min(nLLs[:, 2*i])} - {max(nLLs[:, 2*i])}')
                print(f'nLLs mu1 {i} range: {min(nLLs[:, 2*i+1])} - {max(nLLs[:, 2*i+1])}')
                nLLs[:, 2*i+1] -= nLLs[:, 2*i]
            nLLs = nLLs[:, 1::2]
            for i in range(4):
                print(f'nLLs {i} range: {min(nLLs[:, i])} - {max(nLLs[:, i])}')

            # optionally restrict to a subset of outputs
            target_indices = list(self.cfg.data.get("target_indices") or range(4))
            if target_indices != list(range(4)):
                nLLs = nLLs[:, target_indices]
                nLL_mu0 = nLL_mu0[target_indices]
            self.nLL_mu0.append(nLL_mu0)

            # ensure fvs if required
            if (
                "DSI" in self.cfg.model.net._target_
                or "FV_MLP" in self.cfg.model.net._target_
            ):
                assert self.cfg.data.incl_fvs, "DSI/FV_MLP model requires fvs"

            # Finetuning: optionally PIN the input/output normalization to a
            # pretrained model's stats (from its exported ONNX) instead of refitting
            # from the current data, so old data is decoded exactly as before and is
            # retained. Requires the same trafos as the pretrained model.
            fx_nLL_mean = fx_nLL_std = fx_feat_mean = fx_feat_std = None
            fixed_stats_onnx = self.cfg.data.get("fixed_stats_onnx")
            if fixed_stats_onnx:
                import onnx, json as _json
                _m = onnx.load(fixed_stats_onnx)
                _st = _json.loads({p.key: p.value for p in _m.metadata_props}["standardization"])
                fx_nLL_mean = np.asarray(_st["nLLs_mean"]).ravel()
                fx_nLL_std = np.asarray(_st["nLLs_std"]).ravel()
                fx_feat_mean = np.asarray(_st["features_mean"]).ravel()
                fx_feat_std = np.asarray(_st["features_std"]).ravel()
                if target_indices != list(range(4)):
                    fx_nLL_mean = fx_nLL_mean[target_indices]
                    fx_nLL_std = fx_nLL_std[target_indices]
                LOGGER.info(f"Pinning normalization to pretrained stats from {fixed_stats_onnx}")

            # Number of leading rows that _init_dataloader will use as the
            # training block. The standardisation statistics must be fitted on
            # those rows alone: fitting on the whole array leaks the column means
            # and variances of val/test into the normalisation the network is
            # trained with. Rows are already in train|val|test order here, either
            # from the shuffle above or from the on-disk presplit.
            n_data = len(features)
            if presplit is not None:
                n_fit = presplit[0]
            else:
                n_fit = int(n_data * self.cfg.data.train_test_val[0])
                if self.cfg.data.subsample is not None:
                    n_fit = min(int(self.cfg.data.subsample), n_fit)
            LOGGER.info(
                f"Fitting preprocessing statistics on the training block only "
                f"({n_fit} of {n_data} rows)"
            )

            # preprocess data
            LOGGER.info(f"Preprocessing nLLs using trafos={self.cfg.data.nLL_trafos}")
            nLLs_prepd, prepd_mean, prepd_std, prepd_nll_bounds = preprocess_nLLs(
                nLLs, trafos=self.cfg.data.nLL_trafos,
                fixed_mean=fx_nLL_mean, fixed_std=fx_nLL_std,
                n_fit=n_fit,
            )

            LOGGER.info(f"Preprocessing features using trafos={self.cfg.data.trafos}")
            print("#######################")
            print(self.cfg.data.trafos)
            features_prepd, prepd_mean_features, prepd_std_features = preprocess_features(
                features,
                self.type_token[0],
                trafos=self.cfg.data.trafos,
                incl_fvs=self.cfg.data.incl_fvs,
                mean=fx_feat_mean,
                std=fx_feat_std,
                n_fit=n_fit,
            )
            print("########################")
            print("prepd_mean_features:", prepd_mean_features)
            print("prepd_std_features:", prepd_std_features)

            # save number of features for later
            self.cfg.model.net.n_features = features_prepd.shape[-1]
            with open_dict(self.cfg):
                self.cfg.model.net.out_shape = nLLs.shape[-1]

            # collect everything
            self.features.append(features)
            self.nLLs.append(nLLs)
            self.features_prepd.append(features_prepd)
            self.nLLs_prepd.append(nLLs_prepd)
            self.prepd_mean.append(prepd_mean)
            self.prepd_mean_features.append(prepd_mean_features)
            self.prepd_std.append(prepd_std)
            self.prepd_std_features.append(prepd_std_features)
            self.prepd_nll_bounds.append(prepd_nll_bounds)


    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        # separate data into train, test, validation subsets
        train_sets, test_sets, val_sets = (
            {"features": [], "nLLs": []},
            {"features": [], "nLLs": []},
            {"features": [], "nLLs": []},
        )

        for idataset in range(self.n_datasets):
            n_data = self.features[idataset].shape[0]
            presplit = self.presplit[idataset] if getattr(self, "presplit", None) else None

            if presplit is not None:
                # split fixed on disk: <dataset>.npy | _val.npy | _test.npy, in that
                # order (see init_data). train_test_val/subsample do not apply.
                n_tr, n_va, _ = presplit
                train_idx = np.arange(0, n_tr)
                val_idx   = np.arange(n_tr, n_tr + n_va)
                test_idx  = np.arange(n_tr + n_va, n_data)
                n_train = n_tr
            else:
                # desired train size
                if self.cfg.data.subsample is None:
                    self.cfg.data.subsample = int(n_data * self.cfg.data.train_test_val[0])
                n_train = int(self.cfg.data.subsample) if self.cfg.data.subsample is not None else int(n_data * self.cfg.data.train_test_val[0])
                n_train = min(n_train, int(n_data * self.cfg.data.train_test_val[0]))  # cap

                # ratio val/train relative to config
                val_ratio = self.cfg.data.train_test_val[2] / self.cfg.data.train_test_val[0]
                n_val = max(int(n_train * val_ratio), 1)

                # pick indices
                train_idx = np.arange(0, n_train)
                val_idx   = np.arange(n_train, n_train + n_val)
                test_idx  = np.arange(n_train + n_val, n_data)  # remainder

            # slice preprocessed
            train_sets["features"].append(self.features_prepd[idataset][train_idx])
            train_sets["nLLs"].append(self.nLLs_prepd[idataset][train_idx])

            val_sets["features"].append(self.features_prepd[idataset][val_idx])
            val_sets["nLLs"].append(self.nLLs_prepd[idataset][val_idx])

            test_sets["features"].append(self.features_prepd[idataset][test_idx])
            test_sets["nLLs"].append(self.nLLs_prepd[idataset][test_idx])

            if self.cfg.data.no_props:
                self.props_val = self.props[idataset][self.split_test : self.split_val]

        # Per-output range of the TRAINING targets in preprocessed space. Used to
        # bound the model's mean outputs (models.mup_mlp.MuMLP.set_output_clamp)
        # so a runaway prediction cannot decode to an absurd nLL through the
        # exponential inverse. Test/val are deliberately excluded.
        _tr = np.concatenate([np.asarray(a) for a in train_sets["nLLs"]], axis=0)
        self.nLL_prepd_train_range = (_tr.min(axis=0), _tr.max(axis=0))

        # create dataloaders
        self.cfg.training.batchsize = int(min(self.cfg.training.batchsize, n_train / 2))
        self.train_loader = torch.utils.data.DataLoader(
            dataset=nLLsDataset(
                train_sets["features"], train_sets["nLLs"], dtype=self.dtype
            ),
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
            drop_last=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=nLLsDataset(
                test_sets["features"], test_sets["nLLs"], dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            drop_last=True,
        )

        self.val_loader = torch.utils.data.DataLoader(
            dataset=nLLsDataset(
                val_sets["features"], val_sets["nLLs"], dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            drop_last=True,
        )

        n_train = sum(len(x) for x in train_sets["features"])
        n_val   = sum(len(x) for x in val_sets["features"])
        n_test  = sum(len(x) for x in test_sets["features"])

        LOGGER.info(
            f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "
            f"train_samples={n_train}, val_samples={n_val}, test_samples={n_test}, "
            f"train_batches={len(self.train_loader)}, val_batches={len(self.val_loader)}, test_batches={len(self.test_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )
        
    def _result_extra(self) -> dict:
        """Metrics on the best model for DyHPO ranking.

        Emits both the preprocessed-space val MSE (``val_mse``, legacy) and a
        physical-space relative-error metric (``val_rel_err`` over all outputs,
        ``val_rel_err_obs`` for the observed output). The relative metric is
        |pred-truth|/(|truth|+eps) with a small numerical eps, so it rewards
        accuracy where |nLL| is small (the near-UL physics) and is comparable
        across different preprocessings — unlike preprocessed-space MSE.
        """
        eps = 1e-6
        try:
            self.model.eval()
            with torch.no_grad():
                val_results = self._evaluate_single(self.val_loader, "val")
            # column position of the observed output (orig index 1) after any
            # target_indices restriction
            ti = list(self.cfg.data.get("target_indices") or range(4))
            obs_pos = ti.index(1) if 1 in ti else None

            # Events with |truth| <= rel_min have no relative scale (delta-nLL == 0
            # means the signal is invisible), and |p-t|/(|t|+eps) would report
            # |p|/1e-6 for them — enough to dominate the metric the sweep selects
            # on. Judge those by absolute error (val_abs_err) instead.
            rel_min = float(self.cfg.evaluation.get("rel_min_truth", 1e-2))

            mse_values = []
            rel_mean, rel_med, rel_obs_mean, rel_obs_med = [], [], [], []
            abs_mean, abs_med = [], []
            nll_mean, nll_med, nll_obs_mean, nll_obs_med = [], [], [], []
            for idataset, split in enumerate(val_results.values()):
                if "preprocessed" in split and "mse" in split["preprocessed"]:
                    mse_values.append(split["preprocessed"]["mse"])
                raw = split.get("raw", {})
                t = np.asarray(raw.get("truth"))
                p = np.asarray(raw.get("prediction"))
                if t.size and p.size and t.shape == p.shape:
                    err = np.abs(p - t)
                    abs_mean.append(float(err.mean()))
                    abs_med.append(float(np.median(err)))
                    ok = np.abs(t) > rel_min
                    rel = np.where(ok, err / (np.abs(t) + eps), np.nan)
                    if ok.any():
                        rel_mean.append(float(np.nanmean(rel)))
                        rel_med.append(float(np.nanmedian(rel)))
                        if obs_pos is not None and rel.ndim == 2 and obs_pos < rel.shape[1]:
                            if ok[:, obs_pos].any():
                                rel_obs_mean.append(float(np.nanmean(rel[:, obs_pos])))
                                rel_obs_med.append(float(np.nanmedian(rel[:, obs_pos])))

                    # Same error, but relative to the ABSOLUTE nLL at mu=1
                    # (baseline + delta) rather than to the delta. The baseline is
                    # O(200-700), so the denominator is never near 0 and no row has
                    # to be masked out — unlike the delta metric above, which
                    # discards every signal-free scan row and so is blind to
                    # exactly the low-signal regime these scans add.
                    base = (self.nLL_mu0[idataset]
                            if idataset < len(self.nLL_mu0) else None)
                    if base is not None and np.ndim(t) == 2 and len(base) == t.shape[1]:
                        denom = np.abs(t + base)
                        if np.all(denom > eps):
                            rel_nll = err / denom
                            nll_mean.append(float(rel_nll.mean()))
                            nll_med.append(float(np.median(rel_nll)))
                            if obs_pos is not None and obs_pos < rel_nll.shape[1]:
                                nll_obs_mean.append(float(rel_nll[:, obs_pos].mean()))
                                nll_obs_med.append(float(np.median(rel_nll[:, obs_pos])))

            extra = {}
            if mse_values:
                extra["val_mse"] = float(np.mean(mse_values))
            # mean is spiky (dominated by near-zero-truth points at small eps);
            # median is the robust relative-error signal
            if rel_mean:
                extra["val_rel_err"] = float(np.mean(rel_mean))
                extra["val_rel_err_med"] = float(np.mean(rel_med))
            if rel_obs_mean:
                extra["val_rel_err_obs"] = float(np.mean(rel_obs_mean))
                extra["val_rel_err_obs_med"] = float(np.mean(rel_obs_med))
            if abs_mean:
                extra["val_abs_err"] = float(np.mean(abs_mean))
                extra["val_abs_err_med"] = float(np.mean(abs_med))
            # relative to the absolute nLL (baseline + delta): no masking, every
            # validation row contributes
            if nll_mean:
                extra["val_rel_err_nll"] = float(np.mean(nll_mean))
                extra["val_rel_err_nll_med"] = float(np.mean(nll_med))
            if nll_obs_mean:
                extra["val_rel_err_nll_obs"] = float(np.mean(nll_obs_mean))
                extra["val_rel_err_nll_obs_med"] = float(np.mean(nll_obs_med))
            return extra
        except Exception:
            pass
        return {}

    def evaluate(self):
        with torch.no_grad():
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results_train = self._evaluate_single(
                        self.train_loader, "train"
                    )
                    self.results_val = self._evaluate_single(self.val_loader, "val")
                    self.results_test = self._evaluate_single(self.test_loader, "test")

                # also evaluate without ema to see the effect
                self._evaluate_single(self.train_loader, "train_noema")
                self._evaluate_single(self.val_loader, "val_noema")
                self._evaluate_single(self.test_loader, "test_noema")

            else:
                self.results_train = self._evaluate_single(self.train_loader, "train")
                self.results_val = self._evaluate_single(self.val_loader, "val")
                self.results_test = self._evaluate_single(self.test_loader, "test")
            
            self.results = {}
            for dataset in self.results_test.keys():
                self.results[dataset] = {
                    "train": self.results_train[dataset],
                    "val": self.results_val[dataset],
                    "test": self.results_test[dataset],
                }
            
        return self.results

    def call_model_fn(self, x, idataset):
        return self.model(
            x.to(self.device),
            type_token=torch.tensor(
                [self.type_token[idataset]],
                dtype=torch.long,
                device=self.device,
            ),
            global_token=torch.tensor(
                [idataset],
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _evaluate_single(self, loader, title):
        # compute predictions
        # note: shuffle=True or False does not matter, because we take the predictions directly from the dataloader and not from the dataset
        nLLs_truth_prepd, nLLs_pred_prepd = [
            [] for _ in range(self.n_datasets)
        ], [[] for _ in range(self.n_datasets)]
        if self.cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
            nLLs_sigmas = [[] for _ in range(self.n_datasets)]
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        t0 = time.time()
        for data in loader:
            for idataset, data_onedataset in enumerate(data):
                x, y = data_onedataset
                #print('_evaluate_single: x shape:', x.shape, 'y shape:', y.shape)
                #x = x.unsqueeze(0)
                pred = self.model(
                    x.to(self.device),
                    type_token=torch.tensor(
                        [self.type_token[idataset]],
                        dtype=torch.long,
                        device=self.device,
                    ),
                    global_token=torch.tensor(
                        [idataset], dtype=torch.long, device=self.device
                    ),
                )
                #print(f'pred: {pred}')
                #print(f'pred shape: {pred.shape}')
                y_pred = pred
                #print(y)
                #print('shape:',y.shape)
                #print('######################################################')
                #print(y_pred)
                #print('pred shape:',y_pred.shape)

                nLLs_pred_prepd[idataset].append(y_pred[:, :4].cpu().float().numpy())
                if self.cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
                    nLLs_sigmas[idataset].append(y_pred[:, -4:].cpu().float().numpy())
                nLLs_truth_prepd[idataset].append(
                    y.cpu().float().numpy()
                )
        #print(nLLs_pred_prepd)
        #print('pred shape:',nLLs_pred_prepd.shape)
        #print('######################################################')
        #print(nLLs_truth_prepd)
        print('nLLs_pred_prepd length:',len(nLLs_pred_prepd[0]))
        print('nLLs_truth_prepd length:',len(nLLs_truth_prepd[0]))
        #print('truth shape:',nLLs_truth_prepd.shape)
        nLLs_pred_prepd = [
            np.array(individual) for individual in nLLs_pred_prepd
        ]
        nLLs_truth_prepd = [
            np.array(individual) for individual in nLLs_truth_prepd
        ]
        print('pred shape:',nLLs_pred_prepd[0].shape)
        print('truth shape:',nLLs_truth_prepd[0].shape)
        if self.cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
            nLLs_sigmas = [np.array(individual) for individual in nLLs_sigmas]
            print('sigmas shape:',nLLs_sigmas[0].shape)
            #print('sigmas:',nLLs_sigmas[0])
            
        dt = (
            (time.time() - t0)
            * 1e6
            / sum(arr.shape[0] for arr in nLLs_truth_prepd)
        )
        LOGGER.info(
            f"Evaluation time: {dt:.2f}s for 1M events "
            f"using batchsize {self.cfg.evaluation.batchsize}"
        )

        results = {}
        for idataset, dataset in enumerate(self.cfg.data.dataset):
            nLL_pred_prepd = nLLs_pred_prepd[idataset]
            nLL_truth_prepd = nLLs_truth_prepd[idataset]
            nLL_pred_prepd = nLL_pred_prepd.reshape(-1, nLL_pred_prepd.shape[-1])
            nLL_truth_prepd = nLL_truth_prepd.reshape(-1, nLL_truth_prepd.shape[-1])
            if self.cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
                nLL_sigmas = nLLs_sigmas[idataset]
                nLL_sigmas = nLL_sigmas.reshape(-1, nLL_sigmas.shape[-1])
                #print('nLL_sigmas:',nLL_sigmas)

            # compute metrics over preprocessed nLLs
            mse_prepd = np.mean((nLL_pred_prepd - nLL_truth_prepd) ** 2)
            LOGGER.info(f"MSE on {title} {dataset} dataset: {mse_prepd:.4e}")
            l1_prepd = np.mean(np.abs(nLL_pred_prepd - nLL_truth_prepd))
            LOGGER.info(f"L1 on {title} {dataset} dataset: {l1_prepd:.4e}")
            l1_rel_prepd = np.mean(np.abs(nLL_pred_prepd - nLL_truth_prepd)/np.maximum(np.abs(nLL_truth_prepd),1e-8))
            LOGGER.info(f"Relative L1 on {title} {dataset} dataset: {l1_rel_prepd:.4e}")


            # undo preprocessing
            nLL_truth = undo_preprocess_nLLs(
                nLL_truth_prepd,
                self.prepd_mean[idataset],
                self.prepd_std[idataset],
                trafos=self.cfg.data.nLL_trafos,
                nll_bounds=self.prepd_nll_bounds[idataset],
            )
            nLL_pred = undo_preprocess_nLLs(
                nLL_pred_prepd,
                self.prepd_mean[idataset],
                self.prepd_std[idataset],
                trafos=self.cfg.data.nLL_trafos,
                nll_bounds=self.prepd_nll_bounds[idataset],
            )
            if self.cfg.data.no_props:
                match title:
                    case "train":
                        nLL_pred /= self.props_train.flatten()
                        nLL_truth /= self.props_train.flatten()
                    case "val":
                        nLL_pred /= self.props_val.flatten()
                        nLL_truth /= self.props_val.flatten()
                    case "test":
                        nLL_pred /= self.props_test.flatten()
                        nLL_truth /= self.props_test.flatten()

            # compute metrics over actual nLLs
            mse = np.mean((nLL_truth - nLL_pred) ** 2)
            l1 = np.mean(np.abs(nLL_truth - nLL_pred))

            # Relative error is only meaningful where the truth is bounded away from
            # zero: a point with delta-nLL == 0 (no sensitivity to the signal) has no
            # relative scale at all, and near-zero truth makes the ratio explode. Mask
            # those out of every relative metric and report them via the absolute
            # error below instead. `rel_min_truth` = 0 restores the old behaviour.
            rel_min = float(self.cfg.evaluation.get("rel_min_truth", 1e-2))
            rel_ok = np.abs(nLL_truth) > rel_min
            denom = np.where(rel_ok, nLL_truth, np.nan)

            l1_rel = np.nanmean(np.abs(nLL_truth - nLL_pred) / np.abs(denom))

            delta = (nLL_truth - nLL_pred) / denom
            delta_abs = np.abs(delta)
            delta_abs_mean = np.nanmean(delta_abs, axis=0)
            print('delta_abs_mean:', delta_abs_mean)
            n_excl = (~rel_ok).sum(axis=0)
            if n_excl.any():
                LOGGER.info(
                    f"Relative metrics on {dataset} {title}: excluded per output "
                    f"{n_excl.tolist()} of {len(nLL_truth)} events with "
                    f"|truth| <= {rel_min} (no relative scale)"
                )

            delta_maxs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            delta_rates = []
            for delta_max in delta_maxs:
                # nanmean over the masked subset: rate among events that have a
                # relative scale, not diluted by the ones that don't
                rate = np.nanmean(
                    np.where(rel_ok, (delta > -delta_max) & (delta < delta_max), np.nan),
                    axis=0
                )
                delta_rates.append(rate)

            # absolute nLL error per output (physical units) — comparable across
            # different nLL preprocessings (relative metrics blow up for the
            # sign-crossing `obs` output, so track absolute error there too)
            abs_err = np.abs(nLL_truth - nLL_pred)
            abs_err_mean = abs_err.mean(axis=0)
            abs_within1 = (abs_err < 1.0).mean(axis=0)
            neg = nLL_truth < 0.0  # per-output mask of sign-crossing (obs) events

            # LOGGER.info(
            #     f"Mean absolute relative error on {dataset} {title} dataset, output {i}: {delta_abs_mean[i]:.4f}"
            # )
            for i in range(self.model.net.out_shape):
                rates_str = ", ".join([f"{rate[i]} ({delta_maxs[j]})" for j, rate in enumerate(delta_rates)])
                LOGGER.info(
                f"Mean absolute relative error on {dataset} {title} dataset, output {i}: {delta_abs_mean[i]}"
                )
                LOGGER.info(f"Delta rate output {i} [{title} {dataset}]: {rates_str}")
                n_neg = int(neg[:, i].sum())
                neg_abs = float(abs_err[neg[:, i], i].mean()) if n_neg else float("nan")
                LOGGER.info(
                    f"Abs nLL error output {i} [{title} {dataset}]: "
                    f"mean={abs_err_mean[i]:.4f}, frac|err|<1={abs_within1[i]:.4f}, "
                    f"neg-subset mean={neg_abs:.4f} (n={n_neg})"
                )

            #LOGGER.info(
            #    f"rate of events in delta interval on {dataset} {title} dataset:\t"
            #    f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}"
            #)

            # determine 1% largest nLLs
            delta_abs_mean_1percent = []
            for i in range(self.model.net.out_shape):
                scale_i = np.abs(nLL_truth[:, i])
                idx = np.argsort(scale_i)[-int(0.01 * len(scale_i)):]
                delta_abs_mean_1percent.append(np.nanmean(np.abs(delta[idx, i])))

            delta_abs_mean_1percent = np.array(delta_abs_mean_1percent)
            for i in range(self.model.net.out_shape):
                LOGGER.info(
                    f"Mean absolute relative error on 1% largest nLLs on {dataset} {title} dataset, output {i}: {delta_abs_mean_1percent[i]}"
                )

            if self.cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
                pull = (nLL_truth_prepd - nLL_pred_prepd) / nLL_sigmas
            # log to mlflow
            if self.cfg.use_mlflow:
                log_dict = {
                    f"eval.{title}.mse": mse_prepd,
                    f"eval.{title}.l1": l1_prepd,
                    f"eval.{title}.l1_rel": l1_rel_prepd,
                    f"eval.{title}.mse_raw": mse,
                    f"eval.{title}.l1_raw": l1,
                    f"eval.{title}.l1_rel_raw": l1_rel,
                    f"eval.{title}.delta_abs_mean": delta_abs_mean,
                    f"eval.{title}.delta_abs_mean_1percent": delta_abs_mean_1percent,
                }
                for key, value in log_dict.items():
                    log_mlflow(key, value)
            # print('###############################################')
            # print('nLL_pred shape', nLL_pred.shape)
            # print('nLL_truth shape', nLL_truth.shape)
            # print('###############################################')
            nLL = {
                "raw": {
                    "truth": nLL_truth,
                    "prediction": nLL_pred,
                    "mse": mse,
                    "l1": l1,
                    "l1_rel": l1_rel,
                },
                "preprocessed": {
                    "truth": nLL_truth_prepd,
                    "prediction": nLL_pred_prepd,
                    "mse": mse_prepd,
                    "l1": l1_prepd,
                    "l1_rel": l1_rel_prepd,
                },
            }
            if self.cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
                nLL["preprocessed"]["sigmas"] = nLL_sigmas
                nLL["preprocessed"]["pull"] = pull
            results[dataset] = nLL
        return results

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        dataset_titles = [
            DATASET_TITLE_DICT[dataset] for dataset in self.cfg.data.dataset
        ]
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = [f"{model_title}: {dataset_title}" for dataset_title in dataset_titles]
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict["results_test"] = self.results_test
            plot_dict["results_train"] = self.results_train
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        match self.cfg.training.loss:
            case "MSE":
                self.loss = torch.nn.MSELoss()
                LOGGER.info("Using MSE loss")
            case "L1":
                self.loss = torch.nn.L1Loss()
                LOGGER.info("Using L1 loss")
            case "LogCosh":
                self.loss = LogCoshLoss()
                LOGGER.info("Using LogCosh loss")
            case "RelL1":
                self.loss = RelL1Loss()
                LOGGER.info("Using Relative L1 loss")
            case "HETEROSC":
                self.loss = HeteroscedasticLoss()
                LOGGER.info("Using Heteroscedastic loss")
            case "MAE_TO_HETEROSC":
                self.loss = MAEToHetLoss()
                LOGGER.info("Using MAE-to-Heteroscedastic interpolated loss")
            case _:
                raise ValueError(f"Unknown loss function {self.cfg.training.loss}")
            
    def _init_regularization(self):
        self.regularization_lambda = self.cfg.training.regularization_lambda
        match self.cfg.training.regularization:
            case "L2":
                self.regularization = lambda model: sum(param.pow(2.0).sum() for param in model.parameters())
            case "L1":
                self.regularization = lambda model: sum(param.abs().sum() for param in model.parameters())
            case None:
                self.regularization = lambda model: 0.0
            case _:
                raise ValueError(
                    f"Unknown regularization function {self.cfg.training.regularization}"
                )

    def _step(self, data, step):
        self.current_step = step
        super()._step(data, step)

    def _batch_loss(self, data):
        # average over contributions from different datasets
        loss = 0.0
        mse = []
        if len(data) == 1:
            # print('single dataset batch')
            x, y = data[0]
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            attn_mask = None
            type_token = torch.tensor(
                self.type_token, dtype=torch.long, device=self.device
            )
            global_token = torch.tensor([0], dtype=torch.long, device=self.device)
        else:
            features_max = data[-1][0].shape[-2]
            assert features_max == max([d[0].shape[-2] for d in data])
            y = torch.stack([d[1] for d in data], dim=0)
            x = torch.zeros(len(data), *data[-1][0].shape, dtype=self.dtype)
            # carefully construct padding attention mask as float (and not bool!)
            # bool padding mask does not work, because a full row/column of zeros yields nan's in the softmax
            # this will hopefully be fixed soon, see https://github.com/pytorch/pytorch/issues/103749
            attn_mask = (
                torch.ones(
                    len(data),
                    1,
                    1,
                    1 + features_max,
                    1 + features_max,
                    dtype=self.dtype,
                )
                * torch.finfo(self.dtype).min
            )
            type_token = torch.zeros(
                len(data), features_max, dtype=torch.long, device=self.device
            )
            for i, d in enumerate(data):
                features_i = d[0].shape[-2]
                x[i, :, :features_i, :] = d[0]
                attn_mask[i, :, :, : (1 + features_i), : (1 + features_i)] = 0.0
                type_token[i, :features_i] = torch.tensor(
                    self.type_token[i], dtype=torch.long, device=self.device
                )
            global_token = torch.tensor(
                range(len(data)), dtype=torch.long, device=self.device
            )
        x, y = x.to(self.device), y.to(self.device)
        #print('_batch_loss: x shape:', x.shape, 'y shape:', y.shape)
        
        x=x[0]
        y_pred = self.model(
            x, type_token=type_token, global_token=global_token, attn_mask=attn_mask
        )
        #y_pred = pred[0]
        # print('###############################')
        # print('y_pred shape:', y_pred.shape)
        # print(y_pred)
        # print('###############################')
        out_shape = self.cfg.model.net.out_shape
        output_weights_cfg = self.cfg.training.get("output_weights")
        if output_weights_cfg is not None:
            output_weights = torch.tensor(
                list(output_weights_cfg), dtype=y_pred.dtype, device=y_pred.device
            )
        else:
            output_weights = None

        if self.cfg.training.loss in ("HETEROSC", "MAE_TO_HETEROSC"):
            sigma = y_pred[..., -out_shape:]
            y_pred = y_pred[..., :out_shape]
            if self.cfg.training.loss == "MAE_TO_HETEROSC":
                transition_steps = self.cfg.training.mae_to_het_transition_frac * self.cfg.training.iterations
                alpha = min(1.0, getattr(self, 'current_step', 0) / max(1, transition_steps))
            if output_weights is not None:
                # compute per-sample per-output loss, then weight and mean
                if self.cfg.training.loss == "MAE_TO_HETEROSC":
                    mae_per = torch.abs(y - y_pred)
                    het_per = (y - y_pred) ** 2 / (2 * sigma ** 2) + torch.log(sigma)
                    per_output = (1.0 - alpha) * mae_per + alpha * het_per
                else:
                    per_output = (y - y_pred) ** 2 / (2 * sigma ** 2) + torch.log(sigma)
                loss = (per_output * output_weights).mean()
            else:
                if self.cfg.training.loss == "MAE_TO_HETEROSC":
                    loss = self.loss(y_pred, y, sigma, alpha)
                else:
                    loss = self.loss(y_pred, y, sigma)
        else:
            if output_weights is not None:
                if self.cfg.training.loss == "L1":
                    per_output = torch.abs(y_pred - y)
                else:
                    per_output = (y_pred - y) ** 2
                loss = (per_output * output_weights).mean()
            else:
                loss = self.loss(y_pred, y)
        loss = loss + self.regularization_lambda*self.regularization(self.model)
        assert torch.isfinite(loss).all()

        return loss

    def _init_metrics(self):
        metrics = {f"{dataset}.mse": [] for dataset in self.cfg.data.dataset}
        return metrics
