from typing import Sequence

import numpy as np
import torch

from .data_sampler import (
    EVALUATION_PROTOCOL,
    Poisson5DSampler,
    Poisson5DValidationDataLoader,
    Poisson5DValidationDataSet,
)
from .networks import *
from .physical_residual import *
from .run_test import *
from .simulation_paras import *
from ..trainer_basis import *


class Poisson5DTrainerBasis:
    def set_configs_type_dataloader(self):
        self.configs_handler.add_config_item("n_internal", mandatory=True, value_type=int, description="Num of samples in internal domain")
        self.configs_handler.add_config_item("n_boundary", mandatory=True, value_type=int, description="Num of samples for boundary conditions")
        self.configs_handler.add_config_item("n_validation", mandatory=False, default_value=N_VALIDATION, value_type=int, description="Num of validation points")
        self.configs_handler.add_config_item("x_start", mandatory=True, value_type=float, description="Start of each axis")
        self.configs_handler.add_config_item("x_end", mandatory=True, value_type=float, description="End of each axis")

    def generate_dataloader(self, train_dataset, validation_dataset):
        train_dataloader = Poisson5DSampler(
            n_internal=self.configs.n_internal,
            n_boundary=self.configs.n_boundary,
            device=self.configs.device,
            update_data=self.configs.update_training_data,
            seed=self.configs.random_seed,
            data_sampler=self.configs.data_sampler,
            x_start=self.configs.x_start,
            x_end=self.configs.x_end,
        )
        if validation_dataset is not None:
            vali_dataloader = Poisson5DValidationDataLoader(validation_dataset, device=self.configs.device)
        else:
            vali_dataloader = None
        return train_dataloader, vali_dataloader

    def validation_step(self, network, batched_data, idx_batch, num_batches, idx_epoch, num_epoch):
        prediction = network(self.validate_dataloader.xs)
        return torch.nn.functional.mse_loss(prediction, self.validate_dataloader.us)

    def get_internal_loss(self, network, idx_epoch: int):
        x_internal = self.train_dataloader.sample_internal()
        prediction = network(x_internal)
        residual = physical_residual(prediction, x_internal)
        internal_loss = torch.mean(residual**2) + 0.0 * prediction.sum()
        self.recorder.add_scalar("separated_loss/internal_loss", internal_loss.item(), idx_epoch)
        return internal_loss

    def get_boundary_loss(self, network, idx_epoch: int):
        x_boundary, value_boundary = self.train_dataloader.sample_boundary()
        boundary_loss = torch.mean((network(x_boundary) - value_boundary) ** 2)
        self.recorder.add_scalar("separated_loss/boundary_loss", boundary_loss.item(), idx_epoch)
        return boundary_loss

    def _loss_funcs(self):
        if self.configs.n_losses != 2:
            raise ValueError("poisson5d supports only 2-loss: PDE residual and Dirichlet boundary loss.")
        return [self.get_internal_loss, self.get_boundary_loss]


def training(
    epochs,
    trainer: Poisson5DTrainerBasis,
    random_seeds: Sequence,
    path_config_file,
    name,
    num_run=3,
    device="cuda:0",
    n_validation_point=N_VALIDATION,
    **kwargs,
):
    errors = []
    rels = []
    for i in range(num_run):
        set_random_seed(random_seeds[i])
        network = Poisson5DNet()
        validation_dataset = Poisson5DValidationDataSet(
            n_point=n_validation_point,
        )
        trainer.train_from_scratch(
            network=network,
            train_dataset=None,
            validation_dataset=validation_dataset,
            run_in_silence=True,
            path_config_file=path_config_file,
            name=name,
            epochs=epochs,
            warmup_epoch=min(100, int(epochs * 0.01)),
            random_seed=random_seeds[i],
            device=device,
            **kwargs,
        )
        mse_loss, _, _, _, relative_l2 = run_test(
            network,
            n_point=n_validation_point,
            device=device,
        )
        errors.append(mse_loss)
        rels.append(relative_l2)
        write_formal_metrics(
            trainer,
            "poisson5d",
            random_seeds[i],
            mse_loss,
            relative_l2,
            evaluation_protocol=EVALUATION_PROTOCOL,
            evaluation_points=len(validation_dataset.x),
        )
        print("mse_loss:%.5e, relative_l2:%.5e" % (mse_loss, relative_l2))
    print("mse_loss:%.5e±%.5e" % (np.mean(errors), np.std(errors)))
    print("relative_l2:%.5e±%.5e" % (np.mean(rels), np.std(rels)))
    return errors, np.mean(errors), np.std(errors)


def run_poisson5d(
    name,
    trainer,
    epochs=50000,
    num_run=3,
    save_path=None,
    config_file=None,
    update_training_data=True,
    **kwargs,
):
    run_training(
        run_name=name,
        equation_name="poisson5d",
        training_func=training,
        trainer=trainer,
        epochs=epochs,
        num_run=num_run,
        save_path=save_path,
        config_file=config_file,
        update_training_data=update_training_data,
        **kwargs,
    )
