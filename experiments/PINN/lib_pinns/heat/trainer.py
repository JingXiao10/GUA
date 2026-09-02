from typing import Sequence

import numpy as np
import torch

from .data_sampler import HeatSampler, HeatValidationDataLoader, HeatValidationDataSet
from .networks import *
from .physical_residual import *
from .run_test import *
from .simulation_paras import *
from ..trainer_basis import *


class HeatTrainerBasis:
    def set_configs_type_dataloader(self):
        self.configs_handler.add_config_item("n_internal", mandatory=True, value_type=int, description="Num of samples in internal domain")
        self.configs_handler.add_config_item("n_boundary", mandatory=True, value_type=int, description="Num of samples for boundary conditions")
        self.configs_handler.add_config_item("n_initial", mandatory=True, value_type=int, description="Num of samples for initial conditions")
        self.configs_handler.add_config_item("x_start", mandatory=True, value_type=float, description="Start of x axis")
        self.configs_handler.add_config_item("x_end", mandatory=True, value_type=float, description="End of x axis")
        self.configs_handler.add_config_item("y_start", mandatory=True, value_type=float, description="Start of y axis")
        self.configs_handler.add_config_item("y_end", mandatory=True, value_type=float, description="End of y axis")
        self.configs_handler.add_config_item("simulation_time", mandatory=True, value_type=float, description="Simulation time")
        self.configs_handler.add_config_item("heat_diffusivity_x", mandatory=False, default_value=HEAT_DIFFUSIVITY_X, value_type=float, description="HeatMS diffusivity in x direction.")
        self.configs_handler.add_config_item("heat_diffusivity_y", mandatory=False, default_value=HEAT_DIFFUSIVITY_Y, value_type=float, description="HeatMS diffusivity in y direction.")

    def generate_dataloader(self, train_dataset, validation_dataset):
        train_dataloader = HeatSampler(
            n_internal=self.configs.n_internal,
            n_initial=self.configs.n_initial,
            n_boundary=self.configs.n_boundary,
            device=self.configs.device,
            update_data=self.configs.update_training_data,
            seed=self.configs.random_seed,
            data_sampler=self.configs.data_sampler,
            x_start=self.configs.x_start,
            x_end=self.configs.x_end,
            y_start=self.configs.y_start,
            y_end=self.configs.y_end,
            simulation_time=self.configs.simulation_time,
        )
        if validation_dataset is not None:
            vali_dataloader = HeatValidationDataLoader(validation_dataset, device=self.configs.device)
        else:
            vali_dataloader = None
        return train_dataloader, vali_dataloader

    def validation_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        prediction = network(self.validate_dataloader.xs, self.validate_dataloader.ys, self.validate_dataloader.ts)
        return torch.nn.functional.mse_loss(prediction, self.validate_dataloader.values)

    def get_internal_loss(self, network, idx_epoch: int):
        x_internal, y_internal, t_internal = self.train_dataloader.sample_internal()
        prediction = network(x_internal, y_internal, t_internal)
        residual = physical_residual(
            prediction,
            x_internal,
            y_internal,
            t_internal,
            heat_diffusivity_x=self.configs.heat_diffusivity_x,
            heat_diffusivity_y=self.configs.heat_diffusivity_y,
        )
        internal_loss = torch.mean(residual**2) + 0.0 * prediction.sum()
        self.recorder.add_scalar("separated_loss/internal_loss", internal_loss.item(), idx_epoch)
        return internal_loss

    def get_bound_ini_loss(self, network, idx_epoch: int):
        x_bi, y_bi, t_bi, value_bi = self.train_dataloader.sample_initial_boundary()
        bound_ini_loss = torch.mean((network(x_bi, y_bi, t_bi) - value_bi) ** 2)
        self.recorder.add_scalar("separated_loss/boundary_initial_loss", bound_ini_loss.item(), idx_epoch)
        return bound_ini_loss

    def get_boundary_loss(self, network, idx_epoch: int):
        x_b, y_b, t_b, value_b = self.train_dataloader.sample_boundary()
        boundary_loss = torch.mean((network(x_b, y_b, t_b) - value_b) ** 2)
        self.recorder.add_scalar("separated_loss/boundary_loss", boundary_loss.item(), idx_epoch)
        return boundary_loss

    def get_initial_loss(self, network, idx_epoch: int):
        x_i, y_i, t_i, value_i = self.train_dataloader.sample_initial()
        initial_loss = torch.mean((network(x_i, y_i, t_i) - value_i) ** 2)
        self.recorder.add_scalar("separated_loss/initial_loss", initial_loss.item(), idx_epoch)
        return initial_loss

    def _loss_funcs(self):
        if self.configs.n_losses == 2:
            return [self.get_internal_loss, self.get_bound_ini_loss]
        if self.configs.n_losses == 3:
            return [self.get_internal_loss, self.get_boundary_loss, self.get_initial_loss]
        raise ValueError("n_losses should be 2 or 3")


def training(
    epochs,
    trainer: HeatTrainerBasis,
    random_seeds: Sequence,
    path_config_file,
    name,
    num_run=3,
    device="cuda:0",
    **kwargs,
):
    validation_dataset = HeatValidationDataSet()
    errors = []
    relative_errors = []
    for i in range(num_run):
        set_random_seed(random_seeds[i])
        network = HeatNet()
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
        network.eval()
        network.to(device)
        mse_loss, _, _, _, relative_l2 = run_test(network, device=device)
        errors.append(mse_loss)
        relative_errors.append(relative_l2)
        write_formal_metrics(trainer,"heat",random_seeds[i],mse_loss,relative_l2)
        print("mse_loss:%.5e, relative_l2:%.5e" % (mse_loss, relative_l2))
    print("mse_loss:%.5e±%.5e" % (np.mean(errors), np.std(errors)))
    print("relative_l2:%.5e±%.5e" % (np.mean(relative_errors), np.std(relative_errors)))
    return errors, np.mean(errors), np.std(errors)


def run_heat(
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
        equation_name="heat",
        training_func=training,
        trainer=trainer,
        epochs=epochs,
        num_run=num_run,
        save_path=save_path,
        config_file=config_file,
        update_training_data=update_training_data,
        **kwargs,
    )
