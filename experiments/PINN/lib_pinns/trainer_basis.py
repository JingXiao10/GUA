import torch
import math
import os
import json
from .foxutils.trainerX import *
from .network_initialization import *
from .helpers import *

from conflictfree.grad_operator import *
from conflictfree.utils import (
    correct_optimizer_state_from_direction,
    get_adam_v_hat_vector,
    get_optimizer_proposal,
    project,
    task_gradient_conflict_stats,
)

DEFAULT_RANDOM_SEEDS = (0, 1, 2, 3, 4)


def write_formal_metrics(
    trainer,
    equation,
    seed,
    mse,
    relative_l2,
    evaluation_seed=None,
    evaluation_protocol=None,
    evaluation_points=None,
):
    values={"mse":float(mse),"relative_l2":float(relative_l2)}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Final evaluation metrics must be finite.")
    payload={
        "schema_version":1,
        "equation":str(equation),
        "seed":int(seed),
        **values,
        "diagnostics_enabled":bool(trainer.configs.record_update_conflict),
    }
    if evaluation_seed is not None:
        payload["evaluation_seed"]=int(evaluation_seed)
    if evaluation_protocol is not None:
        payload["evaluation_protocol"]=str(evaluation_protocol)
    if evaluation_points is not None:
        payload["evaluation_points"]=int(evaluation_points)
    training_elapsed=float(getattr(trainer,"training_loop_elapsed_seconds",math.nan))
    if not math.isfinite(training_elapsed) or training_elapsed<=0:
        raise RuntimeError("A positive training-loop runtime was not recorded.")
    payload["training_loop_elapsed_seconds"]=training_elapsed
    if torch.cuda.is_available() and str(trainer.configs.device).startswith("cuda"):
        device=torch.device(trainer.configs.device)
        payload["peak_cuda_allocated_mib"]=torch.cuda.max_memory_allocated(device)/(1024**2)
        payload["peak_cuda_reserved_mib"]=torch.cuda.max_memory_reserved(device)/(1024**2)
    total=int(getattr(trainer,"_conflict_total_steps",0))
    if payload["diagnostics_enabled"]:
        if total<=0:
            raise RuntimeError("Conflict diagnostics were enabled but no steps were recorded.")
        optimizer_steps=max(1,int(getattr(trainer,"_after_optimizer_valid_steps",total)))
        payload["conflict"]={
            "r_g":trainer._task_gradient_conflict_steps/total,
            "r_g_pair":trainer._task_gradient_conflicting_pair_fraction_sum/total,
            "r_a":trainer._before_optimizer_conflict_steps/total,
            "r_u":trainer._after_optimizer_conflict_steps/optimizer_steps,
        }
        if trainer.configs.optimizer_correction!="none":
            payload["conflict"]["r_p"]=trainer._after_correction_conflict_steps/total
            all_count=int(trainer._correction_all_count)
            projected_count=int(trainer._correction_projected_count)
            if all_count!=total:
                raise RuntimeError("Correction diagnostics do not cover every recorded step.")
            payload["correction"]={
                "projected_rate":projected_count/all_count,
                "distance_tol":float(trainer.configs.correction_distance_tol),
                "all":{
                    key:value/all_count
                    for key,value in trainer._correction_all_sums.items()
                },
                "projected":{
                    key:(value/projected_count if projected_count else None)
                    for key,value in trainer._correction_projected_sums.items()
                },
            }
    path=os.path.join(trainer.project_path,"formal_metrics.json")
    temporary=path+".tmp"
    with open(temporary,"w",encoding="utf-8") as handle:
        json.dump(payload,handle,indent=2,sort_keys=True)
        handle.write("\n")
    os.replace(temporary,path)

def get_cosine_constant_lambda(initial_lr,final_lr,epochs,warmup_epoch,constant_start_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the cosine schedule.

    Args:
        initial_lr (float): The initial learning rate.
        final_lr (float): The final learning rate.
        epochs (int): The total number of epochs.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    cos_f=get_cosine_lambda(initial_lr,final_lr,constant_start_epoch,warmup_epoch)
    def cosine_constant_lambda(idx_epoch):
        if idx_epoch<constant_start_epoch:
            return cos_f(idx_epoch)
        else:
            return final_lr/initial_lr
    return cosine_constant_lambda

class TrainerBasis(Trainer):
    def __init__(self) -> None:
        super().__init__()
        
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("update_training_data",mandatory=False,default_value=True,
                                             value_type=bool,description="Weather to update bound_ini samples during training.")
        self.configs_handler.add_config_item("data_sampler",default_value="latin_hypercube",value_type=str,
                                             description="Sampler for training bound_ini.",option=["latin_hypercube","monte_carlo"])
        self.configs_handler.add_config_item("network_initialization",default_value="xavier",value_type=str,
                                             description="Initialization method for the network.",option=["xavier","kaiming"])
        self.configs_handler.add_config_item("n_losses",default_value=2,value_type=int,
                                             description="Number of loss terms.")
        self.configs_handler.add_config_item("constant_start_epoch",default_value=100000,value_type=int,
                                             description="Epoch to start constant learning rate.")
        self.configs_handler.add_config_item("lr_scheduler",default_value="cosine",value_type=str,description="Learning rate scheduler for training",option=["cosine","linear","constant","cosine_constant"])
    
    def get_lr_scheduler(self,optimizer):
        """
        Get the learning rate scheduler based on the configuration.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
        
        Raises:
            ValueError: If the learning rate scheduler is not supported.
        """
        if self.configs.lr_scheduler=="cosine":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_cosine_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="linear":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_linear_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_constant_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="cosine_constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_cosine_constant_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch,constant_start_epoch=self.configs.constant_start_epoch))
        else:
            raise ValueError("Learning rate scheduler '{}' not supported".format(self.configs.lr_scheduler))
                
    def event_before_training(self, network):
        if self.configs.network_initialization=="xavier":
            network.apply(xavier_init_weights)
    
    def _loss_funcs(self):
        raise NotImplementedError("The method _loss_funcs should be implemented in the bound_iniclass.")          

    def get_losses(self,network,idx_epoch:int):
        return torch.stack([loss_f(network,idx_epoch) for loss_f in self._loss_funcs()])
        
    def get_gradients(self,network,idx_epoch:int,return_loss=False):
        loss_func=self._loss_funcs()
        grads=[]
        if return_loss:
            losses=[]
        for loss_f in loss_func:
            loss_i=loss_f(network,idx_epoch)
            self.optimizer.zero_grad()
            loss_i.backward()
            grads.append(get_gradient_vector(network,none_grad_mode="zero"))
            if return_loss:
                losses.append(loss_i)
        if return_loss:
            return torch.stack(grads,dim=0),torch.stack(losses)
        else:
            return torch.stack(grads,dim=0)

    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        raise NotImplementedError("The method train_step should be implemented in the bound_iniclass.")

class GradVecTrainerBasis(TrainerBasis):
    
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="Adam",value_type=str,description="Optimizer for training.",option=["Adam"])
        self.configs_handler.add_config_item("record_update_conflict",mandatory=False,default_value=True,
                                             value_type=bool,description="Whether to record conflicts between task gradients and actual optimizer updates.")
        self.configs_handler.add_config_item("conflict_cos_tol",mandatory=False,default_value=1e-6,
                                             value_type=float,description="Negative cosine tolerance used to define a geometric conflict.")
        self.configs_handler.add_config_item("correction_distance_tol",mandatory=False,default_value=1e-8,
                                             value_type=float,description="Relative correction threshold used to count a projected step.")
        self.configs_handler.add_config_item("conflict_log_epoch_frequency",mandatory=False,default_value=100,
                                             value_type=int,description="Frequency for logging optimizer update conflict statistics.")
        self.configs_handler.add_config_item("optimizer_correction",mandatory=False,default_value="none",
                                             value_type=str,option=["none","gua"],
                                             description="Whether to apply GUA projection.")
        self.configs_handler.add_config_item("optimizer_projection_metric",mandatory=False,default_value="adam",
                                             value_type=str,option=["euclidean","adam"],
                                             description="Metric used by GUA projection.")
        self.configs_handler.add_config_item("optimizer_state_correction",mandatory=False,default_value=False,
                                             value_type=bool,description="Whether to align Adam first- and second-moment state.")
        self.configs_handler.add_config_item("optimizer_state_correction_rho_m",mandatory=False,default_value=0.0,
                                             value_type=float,description="Fixed Adam first-moment alignment strength.")
        self.configs_handler.add_config_item("optimizer_state_correction_rho_v",mandatory=False,default_value=0.0,
                                             value_type=float,description="Fixed Adam second-moment alignment strength.")

    def _optimizer_state_correction_rhos(self):
        return (
            float(self.configs.optimizer_state_correction_rho_m),
            float(self.configs.optimizer_state_correction_rho_v),
        )
    
    def initialize_gradient_operator(self):
        raise NotImplementedError
    
    
    def event_before_training(self, network):
        super().event_before_training(network)
        self.initialize_gradient_operator()
        self._conflict_total_steps=0
        self._task_gradient_conflict_steps=0
        self._task_gradient_conflicting_pair_fraction_sum=0.0
        self._task_gradient_min_cos_sum=0.0
        self._before_optimizer_conflict_steps=0
        self._after_optimizer_conflict_steps=0
        self._after_correction_conflict_steps=0
        self._before_optimizer_min_cos_sum=0.0
        self._after_optimizer_min_cos_sum=0.0
        self._after_optimizer_valid_steps=0
        self._after_correction_min_cos_sum=0.0
        self._correction_all_count=0
        self._correction_projected_count=0
        self._correction_all_sums={"cos_p_u":0.0,"relative":0.0,"norm_ratio":0.0}
        self._correction_projected_sums={"cos_p_u":0.0,"relative":0.0,"norm_ratio":0.0}
    
    def _get_conflict_stats(
            self,
            update_gradient:torch.Tensor,
            grads:torch.Tensor,
            conflict_tol:float=None,
            use_cosine_tolerance:bool=True):
        conflict_tol=float(self.configs.conflict_cos_tol) if conflict_tol is None else conflict_tol
        dots=grads@update_gradient
        update_norm=update_gradient.norm()
        grad_norms=grads.norm(dim=1)
        cosines=dots/(grad_norms*update_norm+1e-12)
        conflict_values=cosines if use_cosine_tolerance else dots
        return bool(torch.any(conflict_values < -conflict_tol).item()),torch.min(dots).item(),torch.min(cosines).item()











    def _record_conflict_stats(
            self,
            before_optimizer_direction:torch.Tensor,
            after_optimizer_direction:torch.Tensor,
            grads:torch.Tensor,
            idx_epoch:int):
        task_has_conflict,task_min_cos,task_pair_fraction=task_gradient_conflict_stats(
            grads,
            conflict_tol=float(self.configs.conflict_cos_tol),
        )
        before_has_conflict,before_min_dot,before_min_cos=self._get_conflict_stats(before_optimizer_direction,grads)
        after_stats_valid=after_optimizer_direction is not None
        if after_stats_valid:
            after_has_conflict,after_min_dot,after_min_cos=self._get_conflict_stats(after_optimizer_direction,grads)
        self._conflict_total_steps+=1
        self._task_gradient_conflict_steps+=int(task_has_conflict)
        self._task_gradient_conflicting_pair_fraction_sum+=task_pair_fraction
        self._task_gradient_min_cos_sum+=task_min_cos
        self._before_optimizer_conflict_steps+=int(before_has_conflict)
        self._before_optimizer_min_cos_sum+=before_min_cos
        if after_stats_valid:
            self._after_optimizer_conflict_steps+=int(after_has_conflict)
            self._after_optimizer_min_cos_sum+=after_min_cos
            self._after_optimizer_valid_steps+=1
        before_conflict_rate=self._before_optimizer_conflict_steps/self._conflict_total_steps
        after_conflict_rate=self._after_optimizer_conflict_steps/max(1,self._after_optimizer_valid_steps)
        task_conflict_rate=self._task_gradient_conflict_steps/self._conflict_total_steps
        task_pair_fraction=self._task_gradient_conflicting_pair_fraction_sum/self._conflict_total_steps
        self.recorder.add_scalar("conflict/r_g",task_conflict_rate,idx_epoch)
        self.recorder.add_scalar("conflict/r_g_pair",task_pair_fraction,idx_epoch)
        self.recorder.add_scalar("conflict/task_gradient_min_pair_cos",task_min_cos,idx_epoch)
        self.recorder.add_scalar("conflict/r_a",before_conflict_rate,idx_epoch)
        self.recorder.add_scalar("conflict/r_u",after_conflict_rate,idx_epoch)
        self.recorder.add_scalar("conflict/before_optimizer_min_dot",before_min_dot,idx_epoch)
        self.recorder.add_scalar("conflict/before_optimizer_min_cos",before_min_cos,idx_epoch)
        if after_stats_valid:
            self.recorder.add_scalar("conflict/after_optimizer_min_dot",after_min_dot,idx_epoch)
            self.recorder.add_scalar("conflict/after_optimizer_min_cos",after_min_cos,idx_epoch)
    def _record_correction_conflict_stats(
            self,
            before_correction_direction:torch.Tensor,
            after_correction_direction:torch.Tensor,
            grads:torch.Tensor,
            idx_epoch:int):
        after_has_conflict,after_min_dot,after_min_cos=self._get_conflict_stats(
            after_correction_direction,
            grads,
            conflict_tol=float(self.configs.conflict_cos_tol),
            use_cosine_tolerance=True,
        )
        self._after_correction_conflict_steps+=int(after_has_conflict)
        self._after_correction_min_cos_sum+=after_min_cos
        after_conflict_rate=self._after_correction_conflict_steps/self._conflict_total_steps
        self.recorder.add_scalar("conflict/r_p",after_conflict_rate,idx_epoch)
        self.recorder.add_scalar("conflict/after_correction_min_dot",after_min_dot,idx_epoch)
        self.recorder.add_scalar("conflict/after_correction_min_cos",after_min_cos,idx_epoch)
        norm_u=torch.linalg.vector_norm(before_correction_direction)
        norm_p=torch.linalg.vector_norm(after_correction_direction)
        relative=float((torch.linalg.vector_norm(after_correction_direction-before_correction_direction)/(norm_u+1e-12)).item())
        norm_ratio=float((norm_p/(norm_u+1e-12)).item())
        if relative<=float(self.configs.correction_distance_tol):
            cosine=1.0
            relative=0.0
            norm_ratio=1.0
        else:
            cosine=float((torch.dot(before_correction_direction,after_correction_direction)/(norm_u*norm_p).clamp_min(1e-30)).item())
        values={"cos_p_u":cosine,"relative":relative,"norm_ratio":norm_ratio}
        self._correction_all_count+=1
        for key,value in values.items():
            self._correction_all_sums[key]+=value
        if relative>float(self.configs.correction_distance_tol):
            self._correction_projected_count+=1
            for key,value in values.items():
                self._correction_projected_sums[key]+=value
        for key,value in values.items():
            self.recorder.add_scalar("optimizer_correction/{}_all".format(key),value,idx_epoch)











    def _project_optimizer_direction(self, optimizer_direction:torch.Tensor, grads:torch.Tensor, network) -> torch.Tensor:
        metric=self.configs.optimizer_projection_metric
        if metric=="euclidean":
            return project(optimizer_direction,grads,metric=metric)
        if metric=="adam":
            v_hat=get_adam_v_hat_vector(network,self.optimizer)
            eps=self.optimizer.param_groups[0].get("eps",1e-8)
            return project(
                optimizer_direction,
                grads,
                metric=metric,
                v_hat=v_hat,
                eps=eps,
            )
        raise ValueError("Unsupported optimizer_projection_metric '{}'.".format(metric))











    def _optimizer_direction_has_geometric_conflict(
        self,
        optimizer_direction:torch.Tensor,
        grads:torch.Tensor,
        conflict_tol:float=None,
    ) -> bool:
        conflict_tol=float(self.configs.conflict_cos_tol) if conflict_tol is None else conflict_tol
        dots=grads@optimizer_direction
        update_norm=optimizer_direction.norm()
        grad_norms=grads.norm(dim=1)
        cosines=dots/(grad_norms*update_norm+1e-12)
        if not torch.isfinite(cosines).all():
            return True
        return bool(torch.any(cosines < -conflict_tol).item())

    def _record_no_conflict_correction_skip(self, idx_epoch:int):
        self.recorder.add_scalar("optimizer_correction/skipped_no_conflict",1,idx_epoch)

    def _apply_optimizer_correction(
        self,
        correction_method:str,
        optimizer_direction:torch.Tensor,
        grads:torch.Tensor,
        network,
        idx_epoch:int,
    ) -> torch.Tensor:
        if correction_method=="none":
            return optimizer_direction
        if not self._optimizer_direction_has_geometric_conflict(optimizer_direction,grads):
            self._record_no_conflict_correction_skip(idx_epoch)
            return optimizer_direction
        if correction_method=="gua":
            return self._project_optimizer_direction(optimizer_direction,grads,network)
        raise ValueError("Unsupported optimizer_correction '{}'.".format(correction_method))
    
    def event_after_training_epoch(self, network, idx_epoch):
        if not self.configs.record_update_conflict or self._conflict_total_steps==0:
            return
        if idx_epoch%self.configs.conflict_log_epoch_frequency!=0 and idx_epoch!=self.configs.epochs:
            return
        task_conflict_rate=self._task_gradient_conflict_steps/self._conflict_total_steps
        task_pair_fraction=self._task_gradient_conflicting_pair_fraction_sum/self._conflict_total_steps
        before_conflict_rate=self._before_optimizer_conflict_steps/self._conflict_total_steps
        before_min_cos_average=self._before_optimizer_min_cos_sum/self._conflict_total_steps
        after_optimizer_valid_steps=max(1,getattr(self,"_after_optimizer_valid_steps",self._conflict_total_steps))
        after_conflict_rate=self._after_optimizer_conflict_steps/after_optimizer_valid_steps
        after_min_cos_average=self._after_optimizer_min_cos_sum/after_optimizer_valid_steps
        task_min_cos_average=self._task_gradient_min_cos_sum/self._conflict_total_steps
        info="[Conflict] epoch:{} steps:{} r_g:{:.2%} r_g_pair:{:.2%} r_a:{:.2%} r_u:{:.2%} task_min_pair_cos_avg:{:.4f} a_min_cos_avg:{:.4f} u_min_cos_avg:{:.4f}".format(
                idx_epoch,
                self._conflict_total_steps,
                task_conflict_rate,
                task_pair_fraction,
                before_conflict_rate,
                after_conflict_rate,
                task_min_cos_average,
                before_min_cos_average,
                after_min_cos_average,
        )
        if self.configs.optimizer_correction!="none":
            after_correction_rate=self._after_correction_conflict_steps/self._conflict_total_steps
            after_correction_min_cos_avg=self._after_correction_min_cos_sum/self._conflict_total_steps
            info+=" r_p:{:.2%} p_min_cos_avg:{:.4f}".format(
                after_correction_rate,
                after_correction_min_cos_avg,
            )
            all_count=max(1,self._correction_all_count)
            projected_count=max(1,self._correction_projected_count)
            info+=" projected_rate:{:.2%} cos_p_u_all:{:.6f} correction_rel_all:{:.6g} norm_ratio_all:{:.6g} cos_p_u_projected:{:.6f} correction_rel_projected:{:.6g} norm_ratio_projected:{:.6g}".format(
                self._correction_projected_count/all_count,
                self._correction_all_sums["cos_p_u"]/all_count,
                self._correction_all_sums["relative"]/all_count,
                self._correction_all_sums["norm_ratio"]/all_count,
                self._correction_projected_sums["cos_p_u"]/projected_count,
                self._correction_projected_sums["relative"]/projected_count,
                self._correction_projected_sums["norm_ratio"]/projected_count,
            )
        self.logger.info(info)

    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        grads,losses=self.get_gradients(network,idx_epoch,return_loss=True)
        operator_gradient=self.operator.calculate_gradient(grads,losses)
        if not torch.isfinite(grads).all() or not torch.isfinite(losses).all():
            raise FloatingPointError("Task gradients and losses must be finite before the optimizer update.")
        if not torch.isfinite(operator_gradient).all():
            raise FloatingPointError("Constructed gradient is non-finite.")
        apply_gradient_vector_para_based(network,operator_gradient)
        optimizer_correction=self.configs.optimizer_correction
        need_optimizer_trace=(
            self.configs.record_update_conflict
            or optimizer_correction!="none"
            or self.configs.optimizer_state_correction
        )
        if not need_optimizer_trace:
            self.optimizer.step()
        else:
            para_before=get_para_vector(network).clone()
            self.optimizer.step()
            lr=self.optimizer.param_groups[0]["lr"]
            if lr<=0:
                # LambdaLR intentionally starts warmup at zero. There is no
                # applied parameter update to diagnose on this step.
                total_loss=torch.sum(losses)
                return total_loss
            optimizer_direction=get_optimizer_proposal(self.optimizer, para_before)
            if optimizer_direction is None or not torch.isfinite(optimizer_direction).all():
                raise FloatingPointError("Optimizer returned an empty or non-finite proposal.")
            corrected_optimizer_direction=optimizer_direction
            if optimizer_correction!="none":
                corrected_optimizer_direction=self._apply_optimizer_correction(
                    correction_method=optimizer_correction,
                    optimizer_direction=optimizer_direction,
                    grads=grads,
                    network=network,
                    idx_epoch=idx_epoch,
                )
                if not torch.isfinite(corrected_optimizer_direction).all():
                    raise FloatingPointError("GUA returned a non-finite update.")
                apply_para_vector(network,para_before-lr*corrected_optimizer_direction)
            correction_relative = float(
                (
                    torch.linalg.vector_norm(corrected_optimizer_direction - optimizer_direction)
                    / (torch.linalg.vector_norm(optimizer_direction) + 1e-12)
                ).item()
            )
            correction_was_applied = (
                optimizer_correction != "none"
                and correction_relative > float(self.configs.correction_distance_tol)
            )
            if self.configs.optimizer_state_correction and correction_was_applied:
                rho_m,rho_v=self._optimizer_state_correction_rhos()
                correct_optimizer_state_from_direction(
                    network=network,
                    optimizer=self.optimizer,
                    target_direction=corrected_optimizer_direction,
                    rho_m=rho_m,
                    rho_v=rho_v,
                )
            if self.configs.record_update_conflict:
                self._record_conflict_stats(
                    operator_gradient,
                    optimizer_direction,
                    grads,
                    idx_epoch,
                )
                if optimizer_correction!="none":
                    self._record_correction_conflict_stats(
                        optimizer_direction,
                        corrected_optimizer_direction,
                        grads,
                        idx_epoch,
                    )
        total_loss=torch.sum(losses)
        return total_loss
  
    def back_propagate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        pass









def get_gradvec_trainer(sub_trainer,operator):
    class GradVecTrainer(GradVecTrainerBasis):
        def initialize_gradient_operator(self):
            self.operator=operator
    class Trainer(sub_trainer,GradVecTrainer):
        pass
    return Trainer()









def run_training(
    run_name,
    equation_name,
    training_func,
    trainer,
    epochs=100000,
    num_run=3,
    save_path=None,
    config_file=None,
    update_training_data=True,
    **kwargs
):

    set_random_seed(21339)
    seed_offset = int(kwargs.pop("seed_offset", 0) or 0)
    explicit_seed = kwargs.pop("random_seed", None)
    if explicit_seed is not None:
        if num_run != 1:
            raise ValueError("An explicit random_seed requires num_run=1")
        random_seeds=[int(explicit_seed)]
    else:
        random_seeds=list(DEFAULT_RANDOM_SEEDS)
    if seed_offset < 0:
        raise ValueError("seed_offset must be non-negative")
    if explicit_seed is None:
        random_seeds = random_seeds[seed_offset:]
    if len(random_seeds)<num_run:
        raise ValueError("Not enough random seeds for the requested runs")
    
    print(f"Running {equation_name}:{run_name}")
    if save_path is None:
        save_path="./PINN_trained/"+equation_name+"/"
    if config_file is None:
        config_file=package_path()+f"training_configs/{equation_name}.yaml"
        
    training_func(
        epochs=epochs,
        trainer=trainer,
        path_config_file=config_file,
        save_path=save_path,
        update_training_data=update_training_data,
        name=run_name,
        num_run=num_run,
        random_seeds=random_seeds,
        **kwargs
    )
