import torch
from .simulation_paras import *
from .data_sampler import KovasznayValidationDataSet

def run_test(network,x_start:float=X_START,
                 x_end=X_END,y_start=Y_START,y_end=Y_END,n_point=1001,device="cuda:0",return_ground_truth=False):
    network.eval()
    network.to(device)
    dataset=KovasznayValidationDataSet(x_start=x_start,x_end=x_end,y_start=y_start,y_end=y_end,n_point=n_point)
    with torch.no_grad():
        prediction=network(torch.from_numpy(dataset.x).float().to(device),torch.from_numpy(dataset.y).float().to(device))
        prediction=prediction.detach().cpu().numpy()
    mse=(dataset.uvp-prediction)**2
    mse_value=np.mean(mse)
    relative_l2=np.linalg.norm(prediction.reshape(-1)-dataset.uvp.reshape(-1))/np.linalg.norm(dataset.uvp.reshape(-1))
    if return_ground_truth:
        return float(mse_value),mse,prediction,dataset.uvp,float(relative_l2)
    else:
        return float(mse_value),mse,prediction,float(relative_l2)
