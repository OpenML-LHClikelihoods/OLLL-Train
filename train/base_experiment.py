import numpy as np
import torch
import torch.nn as nn

import math 
import datetime
import pickle 
import os, time
import zipfile
import logging
import glob
import gzip
import copy
import shutil
import pandas as pd
import json
import onnx

from pathlib import Path
from omegaconf import OmegaConf, open_dict, errors
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
#import mlflow
from torch_ema import ExponentialMovingAverage
from wrappers import Log, Exp, Sqrt, Inverse, Standardization, Unstandardization, LogWithNegatives, ExpWithNegatives

from misc import get_device, flatten_dict
import logger
from logger import LOGGER, MEMORY_HANDLER, FORMATTER
from plots import plot_gradients#, plot_weights
#from mlflow_util import log_mlflow

from misc import cosine_warmup_scheduler

from lion_pytorch import Lion
import schedulefree

from IntrinsicDimDeep.get_dim import get_intrinsic_dim   

from mup import set_base_shapes, make_base_shapes
from mup import MuAdam, MuAdamW  # μP optimizers


# set to 'True' to debug autograd issues (slows down code)
torch.autograd.set_detect_anomaly(False)
MIN_STEP_SKIP = 1000


class BaseExperiment:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
    def __call__(self):
        # pass all exceptions to the logger
        try:
            self.run_mlflow()
        except errors.ConfigAttributeError:
            LOGGER.exception(
                "Tried to access key that is not specified in the config files"
            )
        except:
            LOGGER.exception("Exiting with error")

        # print buffered logger messages if failed
        if not logger.LOGGING_INITIALIZED:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            MEMORY_HANDLER.setTarget(stream_handler)
            MEMORY_HANDLER.close()

    def run_mlflow(self):
        experiment_id, run_name = self._init()
        git_hash = os.popen("git rev-parse HEAD").read().strip()
        LOGGER.info(
            f"### Starting experiment {self.cfg.exp_name}/{run_name} (mlflowid={experiment_id}) (jobid={self.cfg.jobid}) (git_hash={git_hash} ###"
        )
        if self.cfg.use_mlflow:
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                self.full_run()
        else:
            # dont use mlflow
            self.full_run()

    def full_run(self):
        # implement all ml boilerplate as private methods (_name)
        t0 = time.time()

        # save config
        LOGGER.debug(OmegaConf.to_yaml(self.cfg))
        self._save_config("config.yaml", to_mlflow=True)
        self._save_config(f"config_{self.cfg.run_idx}.yaml")

        self.init_physics()
        # self.init_geometric_algebra()
        self.init_data()
        self._init_dataloader()
        self.init_model()
        self._init_loss()
        self._init_regularization()

        if self.cfg.train:
            self._init_optimizer()
            self._init_scheduler()
            self.train()
            self._save_model()

        if self.cfg.evaluate:
            self.evaluate()

        if self.cfg.plot and self.cfg.save:
            self.plot()

        if self.device == torch.device("cuda"):
            max_used = torch.cuda.max_memory_allocated()
            max_total = torch.cuda.mem_get_info()[1]
            LOGGER.info(
                f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
            )

        self.compress_models()
        dt = time.time() - t0
        LOGGER.info(
            f"Finished experiment {self.cfg.exp_name}/{self.cfg.run_name} after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )
        # self._wrap_preprocessing()
        # LOGGER.info("onnx model with cfg saved successfully.")

    def init_model(self):
        # initialize model
        self.cfg.model.net.loss = self.cfg.training.loss
        print(self.cfg.model)
        
        self.model = instantiate(self.cfg.model)
        print(self.model)

        # Pin the soft-saturation knees to the training-target range. Identity
        # inside, logarithmic outside — see MuMLP._clamp_means. Traced into the
        # exported ONNX graph, so the behaviour travels with the model instead of
        # relying on the consumer.
        soft = getattr(self.model.net, "out_soft_bound", False)
        rng = getattr(self, "nLL_prepd_train_range", None)
        if soft and rng is not None:
            lo, hi = rng
            self.model.net.set_output_clamp(lo, hi)
            LOGGER.info(
                f"Output soft bound (preprocessed space): "
                f"lo={np.round(lo, 4).tolist()} hi={np.round(hi, 4).tolist()}"
            )
        elif soft:
            LOGGER.warning("out_soft_bound set but no training-target range available")
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.cfg.use_mlflow:
            log_mlflow("num_parameters", float(num_parameters), step=0)
        LOGGER.info(
            f"Instantiated model {type(self.model.net).__name__} with {num_parameters} learnable parameters"
        )

        if self.cfg.ema:
            LOGGER.info(f"Using EMA for validation and eval")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            )
        else:
            LOGGER.info(f"Not using EMA")
            self.ema = None

        # load existing model if specified
        if self.warm_start:
            model_path = os.path.join(
                self.warm_start_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                elif os.path.exists(model_path + ".gz"):
                    with gzip.open(model_path + ".gz", "rb") as f:
                        ckpt = torch.load(f, map_location=self.device, weights_only=False)
                    LOGGER.info(f"Loading model from {model_path}.gz")
                else:
                    raise FileNotFoundError(model_path)
                LOGGER.info(f"Loading model from {model_path}")
                self.model.load_state_dict(ckpt["model"])
                if self.ema is not None:
                    LOGGER.info(f"Loading EMA from {model_path}")
                    self.ema.load_state_dict(ckpt["ema"])
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load model from {model_path}(.gz), training model from scratch"
                )

        # === μP setup ===
        if self.cfg.model.net._target_ == "models.mup_mlp.MuMLP":
            self.bsh_path = os.path.join(self.cfg.run_dir, "base_shapes.bsh")

            if os.path.exists(self.bsh_path):
                LOGGER.info(f"Loading μP base shapes from {self.bsh_path}")
                # Critical: always reset infshapes after load
                # - warm start → rescale_params=False
                # - fresh init → rescale_params=True
                set_base_shapes(self.model, self.bsh_path, rescale_params=not self.warm_start)

            else:
                LOGGER.info("Defining base and delta models for μP setup")
                base_cfg = copy.deepcopy(self.cfg.model)
                base_cfg.net.hidden_channels = 49  # example base width
                base_model = instantiate(base_cfg)

                delta_cfg = copy.deepcopy(self.cfg.model)
                delta_cfg.net.hidden_channels = 128  # example delta width
                delta_model = instantiate(delta_cfg)

                make_base_shapes(base_model, delta_model, savefile=self.bsh_path)
                LOGGER.info(f"Saved μP base shapes to {self.bsh_path}")

                # rescale only on fresh init; on warm start (e.g. finetuning from a
                # pretrained model in another run_dir that lacks a local base_shapes.bsh)
                # the loaded weights must NOT be re-rescaled.
                set_base_shapes(self.model, self.bsh_path, rescale_params=not self.warm_start)

            LOGGER.info("Using μP-aware MLP")

        # === Move to device ===
        self.model.to(self.device, dtype=self.dtype)
        if self.ema is not None:
            self.ema.to(self.device)

    def _init(self):
        run_name = self._init_experiment()
        self._init_directory()

        if self.cfg.use_mlflow:
            experiment_id = self._init_mlflow()
        else:
            experiment_id = None

        # initialize environment
        self._init_logger()
        self._init_backend()

        return experiment_id, run_name

    def _init_experiment(self):
        self.warm_start = False if self.cfg.warm_start_idx is None else True
        # run_dir may be pre-set by DyHPO's run_trial.py as a Hydra override.
        external_run_dir = getattr(self.cfg, "run_dir", None)

        if not self.warm_start:
            run_idx = 0
            if external_run_dir is None:
                if self.cfg.run_name is None:
                    modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
                    now = datetime.datetime.now()
                    rnd_number = np.random.randint(low=0, high=9999)
                    run_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{modelname}_{rnd_number:04}"
                    self.cfg.run_name = run_name
                else:
                    run_name = self.cfg.run_name
                run_dir = os.path.join(
                    self.cfg.base_dir, "runs", self.cfg.exp_name, run_name
                )
                LOGGER.info(f"Creating new experiment {self.cfg.exp_name}/{run_name}")
            else:
                run_dir = external_run_dir
                run_name = os.path.basename(run_dir)
                LOGGER.info(f"Creating new run in {run_dir}")
        else:
            # warm start: run_idx follows from the previous checkpoint index
            run_idx = self.cfg.warm_start_idx + 1
            if external_run_dir is not None:
                run_dir = external_run_dir
            else:
                run_dir = os.path.join(
                    self.cfg.base_dir, "runs", self.cfg.exp_name, self.cfg.run_name
                )
            run_name = self.cfg.run_name if self.cfg.run_name else os.path.basename(run_dir)
            LOGGER.info(
                f"Warm-starting in {run_dir} for run {run_idx}"
            )

        with open_dict(self.cfg):
            self.cfg.run_idx = run_idx
            self.cfg.run_dir = run_dir
            if not self.warm_start:
                self.cfg.warm_start_idx = 0

            # only use mlflow if save=True
            self.cfg.use_mlflow = (
                False if self.cfg.save == False else self.cfg.use_mlflow
            )

        # warm_start_dir: where to load pretrained weights/optimizer/scheduler from.
        # Defaults to run_dir (existing DyHPO multi-fidelity behaviour).
        # Override with warm_start_dir config key to load from a separate pretrained run.
        _ws_dir_override = getattr(self.cfg, 'warm_start_dir', None)
        self.warm_start_dir = _ws_dir_override if _ws_dir_override else run_dir

        # set seed
        if self.cfg.seed is not None:
            LOGGER.info(f"Using seed {self.cfg.seed}")
            torch.random.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        return run_name

    def _init_mlflow(self):
        # mlflow tracking location
        # mlflow.start_run(if you )
        Path(self.cfg.mlflow.db).parent.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{Path(self.cfg.mlflow.db).parent.resolve()}")
        # mlflow.set_tracking_uri(f"sqlite:///{Path(self.cfg.mlflow.db).resolve()}")

        Path(self.cfg.mlflow.artifacts).mkdir(exist_ok=True)
        try:
            # artifacts not supported
            # mlflow call triggers alembic.runtime.migration logger to shout -> shut it down
            logging.disable(logging.WARNING)
            experiment_id = mlflow.create_experiment(
                self.cfg.exp_name,
                artifact_location=f"file:{Path(self.cfg.mlflow.artifacts)}",
            )
            logging.disable(logging.DEBUG)
            LOGGER.info(
                f"Created mlflow experiment {self.cfg.exp_name} with id {experiment_id}"
            )
        except mlflow.exceptions.MlflowException:
            LOGGER.info(f"Using existing mlflow experiment {self.cfg.exp_name}")
            logging.disable(logging.DEBUG)

        experiment = mlflow.set_experiment(self.cfg.exp_name)
        experiment_id = experiment.experiment_id

        # Check if the meta.yaml file exists
        meta_yaml_path = Path(self.cfg.mlflow.db).parent / experiment_id / 'meta.yaml'
        if not meta_yaml_path.exists():
            LOGGER.error(f"meta.yaml file does not exist at {meta_yaml_path}")
            raise mlflow.exceptions.MissingConfigException(f"Yaml file '{meta_yaml_path}' does not exist.")


        LOGGER.info(f"Set experiment {self.cfg.exp_name} with id {experiment_id}")
        return experiment_id

    def _init_directory(self):
        if not self.cfg.save:
            LOGGER.info(f"Running with save=False, i.e. no outputs will be saved")
            return

        # create experiment directory
        run_dir = Path(self.cfg.run_dir).resolve()
        if run_dir.exists() and not self.warm_start:
            raise ValueError(
                f"Experiment in directory {self.cfg.run_dir} alredy exists. Aborting."
            )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

        # save source
        if self.cfg.save_source:
            zip_name = os.path.join(self.cfg.run_dir, "source.zip")
            LOGGER.debug(f"Saving source to {zip_name}")
            zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
            path_experiment = os.path.join(self.cfg.base_dir, "experiments")
            for path in [path_experiment]:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, path))
            zipf.close()

    def _init_logger(self):
        # silence other loggers
        # (every app has a logger, eg hydra, torch, mlflow, matplotlib, fontTools...)
        # for name, other_logger in logging.root.manager.loggerDict.items():
        #     if not "lorentz-gatr" in name:
        #         other_logger.level = logging.WARNING

        if logger.LOGGING_INITIALIZED:
            LOGGER.info("Logger already initialized")
            return

        LOGGER.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)

        # init file_handler
        if self.cfg.save:
            file_handler = logging.FileHandler(
                Path(self.cfg.run_dir) / f"out_{self.cfg.run_idx}.log"
            )
            file_handler.setFormatter(FORMATTER)
            file_handler.setLevel(logging.DEBUG)
            LOGGER.addHandler(file_handler)

        # init stream_handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOGGER.level)
        stream_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(stream_handler)

        # flush memory to stream_handler
        # this allows to catch logs that were created before the logger was initialized
        MEMORY_HANDLER.setTarget(
            stream_handler
        )  # can only flush to one handler, choose stream_handler
        MEMORY_HANDLER.close()
        LOGGER.removeHandler(MEMORY_HANDLER)

        # add new handlers to logger
        LOGGER.propagate = False  # avoid duplicate log outputs

        logger.LOGGING_INITIALIZED = True
        LOGGER.debug("Logger initialized")

    def _init_backend(self):
        #self.device = get_device()
        LOGGER.info(f"Using device {self.device}")

        if (
            self.cfg.training.float16
            and self.device == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            self.dtype = torch.bfloat16
            LOGGER.debug("Using dtype bfloat16")
        elif self.cfg.training.float16:
            self.dtype = torch.float16
            LOGGER.debug(
                "Using dtype float16 (bfloat16 is not supported by environment)"
            )
        else:
            self.dtype = torch.float32
            LOGGER.debug("Using dtype float32")

        torch.backends.cuda.enable_flash_sdp(self.cfg.training.enable_flash_sdp)
        torch.backends.cuda.enable_math_sdp(self.cfg.training.enable_math_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(
            self.cfg.training.enable_mem_efficient_sdp
        )
        if self.cfg.training.force_xformers:
            LOGGER.debug("Forcing use of xformers' attention implementation")
            gatr.primitives.attention.FORCE_XFORMERS = True

    def _init_optimizer(self, param_groups=None):
        if param_groups is None:
            param_groups = [
                {"params": self.model.parameters(), "lr": self.cfg.training.lr}
            ]

        # Check if we should use μP optimizers
        use_mup = self.cfg.model.net._target_ == "models.mup_mlp.MuMLP"

        if self.cfg.training.optimizer == "Adam":
            if use_mup:
                self.optimizer = MuAdam(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                )

        elif self.cfg.training.optimizer == "AdamW":
            if use_mup:
                self.optimizer = MuAdamW(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=self.cfg.training.betas,
                    eps=self.cfg.training.eps,
                    weight_decay=self.cfg.training.weight_decay,
                )

        elif self.cfg.training.optimizer == "RAdam":
            # no MuRAdam implemented yet
            self.optimizer = torch.optim.RAdam(
                param_groups,
                betas=self.cfg.training.betas,
                eps=self.cfg.training.eps,
                weight_decay=self.cfg.training.weight_decay,
            )

        elif self.cfg.training.optimizer == "Lion":
            # Lion is not μP-aware (you'd need to implement a MuLion if required)
            self.optimizer = Lion(
                param_groups,
                betas=self.cfg.training.betas,
                weight_decay=self.cfg.training.weight_decay,
            )

        elif self.cfg.training.optimizer == "ScheduleFree":
            # also not μP-aware
            self.optimizer = schedulefree.AdamWScheduleFree(
                param_groups,
                betas=self.cfg.training.betas,
                weight_decay=self.cfg.training.weight_decay,
            )

        else:
            raise ValueError(f"Optimizer {self.cfg.training.optimizer} not implemented")

        LOGGER.debug(
            f"Using optimizer {self.cfg.training.optimizer}{' (μP)' if use_mup else ''} with lr={self.cfg.training.lr}"
        )

        # load existing optimizer if specified
        if self.warm_start:
            model_path = os.path.join(
                self.warm_start_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                elif os.path.exists(model_path + ".gz"):
                    with gzip.open(model_path + ".gz", "rb") as f:
                        ckpt = torch.load(f, map_location=self.device, weights_only=False)
                else:
                    raise FileNotFoundError(model_path)
                LOGGER.info(f"Loading optimizer from {model_path}")
                self.optimizer.load_state_dict(ckpt["optimizer"])
                # load_state_dict also restores the saved param_groups' lr — which for
                # a finetune warm-start is the pretrained run's *annealed* lr (~0 at the
                # end of a cosine schedule), freezing all updates. Restore the configured
                # lr so the finetune actually trains.
                for _pg in self.optimizer.param_groups:
                    _pg["lr"] = self.cfg.training.lr
                    _pg["initial_lr"] = self.cfg.training.lr
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load optimizer from {model_path}(.gz), starting from scratch"
                )


    def _init_scheduler(self):
        if self.cfg.training.scheduler is None:
            self.scheduler = None  # constant lr
            LOGGER.info("Using no scheduler")
        elif self.cfg.training.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.training.lr * self.cfg.training.onecycle_max_lr,
                pct_start=self.cfg.training.onecycle_pct_start,
                total_steps=int(
                    self.cfg.training.iterations * self.cfg.training.scheduler_scale
                ),
            )
            LOGGER.info('Using OneCycleLR scheduler')
        elif self.cfg.training.scheduler == "CosineAnnealingLR":
            T_max = int(self.cfg.training.iterations * self.cfg.training.scheduler_scale)
            warmup_frac = self.cfg.training.cosanneal_warmup_frac
            warmup_steps = (
                round(warmup_frac * T_max) if warmup_frac > 0
                else self.cfg.training.cosanneal_warmup_steps
            )
            self.scheduler = cosine_warmup_scheduler(
                self.optimizer,
                warmup_steps,
                T_max=T_max,
                eta_min=self.cfg.training.cosanneal_eta_min,
            )
            LOGGER.info('Using CosineAnnealingLR scheduler')
        elif self.cfg.training.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg.training.reduceplateau_factor,
                patience=self.cfg.training.reduceplateau_patience,
            )
            LOGGER.info('Using ReduceLROnPlateau scheduler')
        else:
            raise ValueError(
                f"Learning rate scheduler {self.cfg.training.scheduler} not implemented"
            )

        LOGGER.debug(f"Using learning rate scheduler {self.cfg.training.scheduler}")

        # load existing scheduler if specified
        if self.warm_start and self.scheduler is not None:
            model_path = os.path.join(
                self.warm_start_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt"
            )
            try:
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                elif os.path.exists(model_path + ".gz"):
                    with gzip.open(model_path + ".gz", "rb") as f:
                        ckpt = torch.load(f, map_location="cpu", weights_only=False)
                else:
                    raise FileNotFoundError(model_path)
                LOGGER.info(f"Loading scheduler from {model_path}")
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except FileNotFoundError:
                LOGGER.warning(
                    f"Cannot load scheduler from {model_path}(.gz), starting from scratch"
                )

    def train(self):
        # performance metrics
        self.train_lr, self.train_loss, self.val_loss, self.train_grad_norm = (
            [],
            [],
            [],
            [],
        )
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()

        # early stopping
        smallest_val_loss, smallest_val_loss_step = 1e10, 0
        patience = 0

        # main train loop
        LOGGER.info(
            f"Starting to train for {self.cfg.training.iterations} iterations "
            f"= {self.cfg.training.iterations / len(self.train_loader):.1f} epochs "
            f"on a dataset with {len(self.train_loader)} batches "
            f"using early stopping with patience {self.cfg.training.es_patience} "
            f"while validating every {self.cfg.training.validate_every_n_steps} iterations"
        )
        self.training_start_time = time.time()

        # recycle trainloader
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        results = []#{}
        #results['samplesize']=int(self.cfg.data.subsample)
        mses_prepd = []
        iters = []
        iters_evaluated = []
        iterator = iter(cycle(self.train_loader))
        for step in range(self.cfg.training.iterations):
            # training
            if step > 0:   # skip first step to get evaluation at initialization
                self.model.train()
                if self.cfg.training.optimizer == "ScheduleFree":
                    self.optimizer.train()
                data = next(iterator)
                self._step(data, step)
            
            if self.cfg.training.save_intermediate or (step+1) == self.cfg.training.iterations: ########## FIX LATER
                oom=math.floor(np.log10(self.cfg.training.iterations))
                mantissa=self.cfg.training.iterations/(10**oom)
                iters_to_eval=[math.floor(mantissa*10**(i/2)) for i in range(0,2*oom+1)]
                if ((step+1) in iters_to_eval) or (step == 0):
                    self._evals(results,smallest_val_loss_step,step)
            
                if self.cfg.training.save_gradients:
                    if ((step+1) in iters_to_eval) or (step == 0): #plot gradients
                        self._save_gradients(step)                    
                # if self.cfg.training.save_weights or (step == 0):
                #     if ((step+1) in iters_to_eval): #plot weights
                #         self._save_weights(step)
            
            # validation (and early stopping)        
            if ((step + 1) % self.cfg.training.validate_every_n_steps == 0) or (step < self.cfg.training.validate_every_n_steps and self.cfg.training.save_intermediate):
                val_loss = self._validate(step)
                if val_loss < smallest_val_loss:
                    #LOGGER.info(
                    #    f"Validation loss improved from {smallest_val_loss:.4f} to {val_loss:.4f} in iteration {step+1}"
                    #)
                    smallest_val_loss = val_loss
                    smallest_val_loss_step = step
                    patience = 0

                    # save best model
                    if self.cfg.training.es_load_best_model:
                        self._save_model(
                            filename=f"model_run{self.cfg.run_idx}_best.pt",#it{smallest_val_loss_step}.pt"
                            step=step,
                        )
                else:
                    if (step < self.cfg.training.validate_every_n_steps and self.cfg.training.save_intermediate):
                        patience += 1
                    else:
                        patience += self.cfg.training.validate_every_n_steps
                    if patience > self.cfg.training.es_patience:
                        LOGGER.info(
                            f"Early stopping in iteration {step} = epoch {step / len(self.train_loader):.1f}"
                        )
                        break  # early stopping

                if self.cfg.training.scheduler in ["ReduceLROnPlateau"]:
                    self.scheduler.step(val_loss)

            # output
            dt = time.time() - self.training_start_time
            if step in [0, 999]:
                dt_estimate = dt * self.cfg.training.iterations / (step + 1)
                LOGGER.info(
                    f"Finished iteration {step+1} after {dt:.2f}s, "
                    f"training time estimate: {dt_estimate/60:.2f}min "
                    f"= {dt_estimate/60**2:.2f}h"
                )

        dt = time.time() - self.training_start_time
        LOGGER.info(
            f"Finished training for {step} iterations = {step / len(self.train_loader):.1f} epochs "
            f"after {dt/60:.2f}min = {dt/60**2:.2f}h"
        )
        if self.cfg.use_mlflow:
            log_mlflow("iterations", step)
            log_mlflow("epochs", step / len(self.train_loader))
            log_mlflow("traintime", dt / 3600)

        # wrap up early stopping
        if (self.cfg.training.es_load_best_model) and (patience > self.cfg.training.es_patience):
            self._evals(results,smallest_val_loss_step,step)
            self._load_previous_best_model()
        elif (self.cfg.training.es_load_best_model):
            self._load_previous_best_model()

        # DyHPO: write result JSON so run_trial.py can report to the surrogate
        result_path = getattr(self.cfg.training, "result_path", None)
        if result_path:
            import json
            result = {"val_loss": float(smallest_val_loss)}
            result.update(self._result_extra())
            with open(result_path, "w") as f:
                json.dump(result, f)

    def _result_extra(self) -> dict:
        """Subclasses can override to add extra fields to the result JSON."""
        return {}
        
    def _evals(self,results,smallest_val_loss_step,step):
        LOGGER.info(
                f"### Evaluating model from iteration {smallest_val_loss_step+1} in iteration {step+1}###"
            )
        self._save_model(
            filename=f"model_run{self.cfg.run_idx}_current.pt",
            step=step,
        )
        if self.cfg.training.load_best_previous:
            try:
                self._load_previous_best_model()
            except FileNotFoundError:
                self._load_previous_current_model()
        res = self.evaluate()
        

        if self.cfg.training.get_ID:  ### ONLY WORKS WITH ONE DATASET, FIX LATER
            id_time = time.time()
            ID_mean, ID_std = get_intrinsic_dim(
                model = self.model,
                input_dataloader = self.test_loader,
                nsamples = min(1e3,self.cfg.data.subsample*self.cfg.training.train_test_val[1]),
                bs=self.cfg.training.batchsize,
                divs = 2,
                res = 3,
                call_model_fn=self.call_model_fn
            )
            id_time = time.time() - id_time
            LOGGER.info(
                f"Intrinsic Dimension estimation took {id_time:.2f}s"
            )
        for dataset, dataset_results in self.results.items():  # Loop through datasets
            for split, split_results in dataset_results.items():  # Loop through splits (train/val/test)
                for processing_type, metrics in split_results.items():  # Loop through raw/preprocessed
                    # Create directory for predictions if it doesn't exist
                    if self.cfg.training.save_preds_intermediate or (step+1) == self.cfg.training.iterations:
                        self._save_preds_intermediate(
                            step + 1, dataset, split, processing_type, metrics
                        )
                    # Append results to list
                    
                    results.append({
                        'run_id': self.cfg.run_idx,
                        'samplesize': int(self.cfg.data.subsample) if self.cfg.data.subsample is not None else -1,
                        'dataset': dataset,
                        'split': split,  # train/val/test
                        'processing_type': processing_type,  # raw/preprocessed
                        'mse': metrics.get('mse'),
                        'l1': metrics.get('l1'),
                        'l1_rel': metrics.get('l1_rel'),
                        'iter': step + 1,
                        'iter_evaluated': smallest_val_loss_step,
                        'ID': [ID_mean, ID_std] if self.cfg.training.get_ID else None,
                        'time': time.time() - self.training_start_time
                    })
        self._load_previous_current_model()
        LOGGER.info(
            f"Saving results to {self.cfg.run_dir}/results_intermediate.pkl"
        )
        with open(os.path.join(self.cfg.run_dir, "results_intermediate.pkl"), "wb") as f:
            pickle.dump(results, f)


    def _save_gradients(self, step):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        LOGGER.info(f"Plotting gradients to {plot_path}/gradients_{step+1}.pdf")
        plot_gradients(file=plot_path,model=self.model, iteration=step+1)

    def _save_weights(self, step):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        LOGGER.info(f"Plotting weights to {plot_path}/weights_{step+1}.pdf")
        plot_weights(file=plot_path,model=self.model, iteration=step+1)

    def _save_preds_intermediate(self, step, dataset, split, processing_type, metrics):
        os.makedirs(os.path.join(self.cfg.run_dir, "preds"), exist_ok=True)
                                    
        # Save predictions to file
        pred_path = os.path.join(self.cfg.run_dir, f"preds/{step+1}_{dataset}_{split}_{processing_type}_pred.npy")
        np.save(pred_path, metrics["prediction"])
        
        # For heteroscedastic case, also save sigmas if they exist
        if self.cfg.training.loss == "HETEROSC" and processing_type == "preprocessed":
            sigma_path = os.path.join(self.cfg.run_dir, f"preds/{step+1}_{dataset}_{split}_{processing_type}_sigmas.npy")
            np.save(sigma_path, metrics["sigmas"])
            pull_path = os.path.join(self.cfg.run_dir, f"preds/{step+1}_{dataset}_{split}_{processing_type}_pull.npy")
            np.save(pull_path, metrics["pull"])

    def _load_previous_model(self,step):
        model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"model_run{self.cfg.run_idx}_it{step}.pt",
            )
        try:
            state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["model"]
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.warning(
                f"Cannot load model (epoch {step}) from {model_path}"
            )  
        if self.cfg.model.net._target_ == "models.mup_mlp.MuMLP":
            set_base_shapes(self.model, self.bsh_path, rescale_params=False)

    def _load_previous_best_model(self):
        model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"model_run{self.cfg.run_idx}_best.pt",
            )
        try:
            state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["model"]
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.warning(
                f"Cannot load best model from {model_path}"
            )
        if self.cfg.model.net._target_ == "models.mup_mlp.MuMLP":
            set_base_shapes(self.model, self.bsh_path, rescale_params=False)
        
    def _load_previous_current_model(self):
        model_path = os.path.join(
                self.cfg.run_dir,
                "models",
                f"model_run{self.cfg.run_idx}_current.pt",
            )
        try:
            state_dict = torch.load(model_path, map_location=self.device,weights_only=False)["model"]
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            LOGGER.warning(
                f"Cannot load current model from {model_path}"
            )
        if self.cfg.model.net._target_ == "models.mup_mlp.MuMLP":
            set_base_shapes(self.model, self.bsh_path, rescale_params=False)
    
    def _step(self, data, step):
        # actual update step
        loss = self._batch_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.training.clip_grad_value is not None:
            # clip gradients at a certain value (this is dangerous!)
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.cfg.training.clip_grad_value,
            )
        # rescale gradients such that their norm matches a given number
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.clip_grad_norm,
                error_if_nonfinite=False,
            )
            .cpu()
            .item()
        )
        if step > MIN_STEP_SKIP and self.cfg.training.max_grad_norm is not None:
            if grad_norm > self.cfg.training.max_grad_norm:
                LOGGER.warning(
                    f"Skipping update, gradient norm {grad_norm} exceeds maximum {self.cfg.training.max_grad_norm}"
                )
                return

        self.optimizer.step()
        if self.ema is not None:
            self.ema.update()

        if self.cfg.training.scheduler in ["OneCycleLR", "CosineAnnealingLR"]:
            self.scheduler.step()

        # collect metrics
        self.train_loss.append(loss.item())
        self.train_lr.append(self.optimizer.param_groups[0]["lr"])
        self.train_grad_norm.append(grad_norm)

        # log to mlflow
        if (
            self.cfg.use_mlflow
            and self.cfg.training.log_every_n_steps != 0
            and step % self.cfg.training.log_every_n_steps == 0
        ):
            log_dict = {
                "loss": loss.item(),
                "lr": self.train_lr[-1],
                "time_per_step": (time.time() - self.training_start_time) / (step + 1),
                "grad_norm": grad_norm,
            }
            for key, values in log_dict.items():
                log_mlflow(f"train.{key}", values, step=step)

    def _validate(self, step):
        start_time_validate = time.time()
        losses = []
        metrics = self._init_metrics()

        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        with torch.no_grad():
            for data in self.val_loader:
                # use EMA for validation if available
                if self.ema is not None:
                    with self.ema.average_parameters():
                        loss = self._batch_loss(data)
                else:
                    loss = self._batch_loss(data)

                losses.append(loss.cpu().item())
        val_loss = np.mean(losses)
        #LOGGER.info(
        #    f"Validation loss after {step} iterations = {val_loss:.4f} "
        #    f"(mean over {len(losses)} batches)"
        #)
        if ((step + 1) % self.cfg.training.validate_every_n_steps == 0):
            self.val_loss.append(val_loss)
        if self.cfg.use_mlflow:
            log_mlflow("val.loss", val_loss, step=step)

        end_time_validate = time.time()
        #LOGGER.info(
        #    f"Validation took {end_time_validate - start_time_validate:.2f}s " ######################################################
        # )
        return val_loss

    def _save_config(self, filename, to_mlflow=False):
        # Save config
        if not self.cfg.save:
            return

        config_filename = Path(self.cfg.run_dir) / filename
        LOGGER.debug(f"Saving config at {config_filename}")
        with open(config_filename, "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

        if to_mlflow and self.cfg.use_mlflow:
            for key, value in flatten_dict(self.cfg).items():
                log_mlflow(key, value, kind="param")

    def _save_model(self, step='end', filename=None):
        if not self.cfg.save:
            return

        if filename is None:
            filename = f"model_run{self.cfg.run_idx}.pt"
        model_path = os.path.join(self.cfg.run_dir, "models", filename)
        LOGGER.debug(f"Saving model at {model_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "step": step,
            },
            model_path,
        )

    def compress_models(self):
        pt_files = glob.glob(os.path.join(self.cfg.run_dir , "models", "*.pt"))

        if not pt_files:
            print("No .pt files found.")
        else:
            for path in pt_files:
                gz_path = path + ".gz"

                if os.path.exists(gz_path):
                    print(f"Skipping {path} (already compressed)")
                    continue

                print(f"Compressing {path} -> {gz_path}")
                with open(path, 'rb') as f_in:
                    with gzip.open(gz_path, 'wb', compresslevel=9) as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(path)  # Remove original
                print(f"  Done, original removed.")

    def build_preprocess_from_cfg(self,  cfg, stats):
        layers = []

        for t in cfg.data.trafos.fvs_standardized:
            if t == "log":
                layers.append(Log())
            elif t == "exp":
                layers.append(Exp())
            elif t == "sqrt":
                layers.append(Sqrt())
            elif t == "inverse":
                layers.append(Inverse())
            elif t == "standardization":
                layers.append(Standardization(stats["mean_features"], stats["std_features"]))
            elif t == "log_w_negatives":
                layers.append(LogWithNegatives())
            else:
                raise ValueError(f"Transformation '{t}' not ONNX-exportable")

        return nn.Sequential(*layers)

    def build_postprocess_from_cfg(self, cfg, stats):
        layers = []

        for t in reversed(cfg.data.nLL_trafos):
            if t == "standardization":
                layers.append(
                    Unstandardization(
                        stats["mean_NLL"],
                        stats["std_NLL"],
                    )
                )

            elif t == "log_w_negatives":
                layers.append(ExpWithNegatives())
            elif t in {"log", "exp", "sqrt", "inverse"}:
                # inverse mapping
                inv = {
                    "log": Exp(),
                    "exp": Log(),
                    "sqrt": lambda x: x**2,
                    "inverse": Inverse(),
                }[t]
                layers.append(inv)
            else:
                raise ValueError(f"Inverse of '{t}' not supported")

        return nn.Sequential(*layers)

    
    
    def _wrap_preprocessing(self):
        class FullModel(nn.Module):
            def __init__(self, preprocess, core_model, postprocess=None):
                super().__init__()
                self.preprocess = preprocess
                self.core = core_model
                self.postprocess = postprocess

            def forward(self, x):
                x = self.preprocess(x)
                x = self.core(x,0,0)
                if self.postprocess is not None:
                    x = self.postprocess(x)
                return x

        stats = {
            "mean_NLL": self.prepd_mean,      # shape (n_outputs,)
            "mean_features": self.prepd_mean_features,      # shape (n_inputs,)
            "std_NLL": self.prepd_std,        # shape (n_outputs,)
            "std_features": self.prepd_std_features,        # shape (n_inputs,)
            # "shift": shift_value,    # scalar or vector
            # optional:
            # "sort_idx_E": ...
            # "sort_idx_pt": ...
        }
        pre = self.build_preprocess_from_cfg(self.cfg, stats)
        post = self.build_postprocess_from_cfg(self.cfg, stats)

        full_model = FullModel(pre, self.model, post)
        full_model = full_model.to("cpu")
        full_model.eval()

        model_path = os.path.join(self.cfg.run_dir, "models", 'model_full.onnx')
        # get one batch from the train loader
        batch = next(iter(self.train_loader))

        # batch is a list/tuple over datasets
        # take the first dataset
        x, y = batch[0]

        # take one example (batch dimension = 1)
        dummy_input = x[:1].to("cpu")


        torch.onnx.export(
            full_model,
            dummy_input,
            model_path,
            input_names=["SRs/CRs"],
            output_names=["nLLs_pred"],
            # dynamic_axes={
            #     "nLLs": {0: "batch"},
            #     "nLLs_pred": {0: "batch"},
            # },
            opset_version=17,
        )

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        cfg_json = json.dumps(cfg_dict)
        onnx_model = onnx.load(model_path)

        meta = onnx_model.metadata_props.add()
        meta.key = "cfg"
        meta.value = cfg_json

        onnx.save(onnx_model, model_path)




    def init_physics(self):
        raise NotImplementedError()

    def init_data(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def _init_dataloader(self):
        raise NotImplementedError()

    def _init_loss(self):
        raise NotImplementedError()
    
    def _init_regularization(self):
        raise NotImplementedError()

    def _batch_loss(self, data):
        raise NotImplementedError()

    def _init_metrics(self):
        raise NotImplementedError()