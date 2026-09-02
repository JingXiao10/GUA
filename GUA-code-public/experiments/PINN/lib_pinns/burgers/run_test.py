import torch
from .simulation_paras import *
from .exact_solution import load_burgers_reference

def run_test(network,n_t=N_T,n_x=N_X,x_test=X_TEST,t_test=T_TEST,device="cuda"):
    simulation_data=load_burgers_reference(xs=x_test,ts=t_test)
    network.to(device)
    network.eval()
    with torch.no_grad():
        xs_test_torch=torch.from_numpy(x_test).float().to(device).unsqueeze(0).repeat(n_t,1)
        ts_test_torch=torch.from_numpy(t_test).float().to(device).unsqueeze(1).repeat(1,n_x)
        prediction=network(xs_test_torch,ts_test_torch)
        prediction=prediction.detach().cpu().numpy()
    mse=(simulation_data-prediction)**2
    mse_value=np.mean(mse)
    relative_l2=np.linalg.norm(prediction.reshape(-1)-simulation_data.reshape(-1))/np.linalg.norm(simulation_data.reshape(-1))
    return float(mse_value),mse,prediction,simulation_data,float(relative_l2)


def validation(network,n_t=N_T,n_x=N_X,x_test=X_TEST,t_test=T_TEST,device="cuda"):
    simulation_data=load_burgers_reference(xs=x_test,ts=t_test)
    network.eval()
    with torch.no_grad():
        xs_test_torch=torch.from_numpy(x_test).float().to(device).unsqueeze(0).repeat(n_t,1)
        ts_test_torch=torch.from_numpy(t_test).float().to(device).unsqueeze(1).repeat(1,n_x)
        prediction=network(xs_test_torch,ts_test_torch)
        prediction=prediction
        mse=((torch.tensor(simulation_data,device=device)-prediction)**2).mean()
    return mse
