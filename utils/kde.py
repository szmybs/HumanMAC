import csv
import time
import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

from abc import ABC
from typing import Optional, List
import math
from torch import Tensor

from thop import profile
from thop import clever_format

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros



def kde(y, y_pred):
    bs, sp, ts, ns, d = y_pred.shape
    kde_ll = torch.zeros((bs, ts, ns), device=y_pred.device)

    for b in range(bs):
        for t in range(ts):
            for n in range(ns):
                try:
                    kernel = GaussianKDE(y_pred[b, :, t, n, :])
                except BaseException:
                    print("b: %d - t: %d - n: %d" % (b, t, n))
                    continue
                # pred_prob = kernel(y_pred[:, b, t, :, n])
                gt_prob = kernel(y[b, :, t, n, :])
                kde_ll[b, t, n] = gt_prob
    # mean_kde_ll = torch.mean(kde_ll)
    mean_kde_ll = torch.mean(torch.mean(kde_ll, dim=-1), dim=0)[None]
    return mean_kde_ll

  
class DynamicBufferModule(ABC, torch.nn.Module):
    """Torch module that allows loading variables from the state dict even in the case of shape mismatch."""
    
    def get_tensor_attribute(self, attribute_name: str) -> Tensor:
        """Get attribute of the tensor given the name.
        Args:
            attribute_name (str): Name of the tensor
        Raises:
            ValueError: `attribute_name` is not a torch Tensor
        Returns:
            Tensor: Tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, Tensor):
            return attribute
        raise ValueError(f"Attribute with name '{attribute_name}' is not a torch Tensor")

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """Resizes the local buffers to match those stored in the state dict.
        Overrides method from parent class.
        Args:
          state_dict (dict): State dictionary containing weights
          prefix (str): Prefix of the weight file.
          *args:
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix) :].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        attribute = self.get_tensor_attribute(param)
                        attribute.resize_(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args)
        

class GaussianKDE(DynamicBufferModule):
    """Gaussian Kernel Density Estimation.
    Args:
        dataset (Optional[Tensor], optional): Dataset on which to fit the KDE model. Defaults to None.
    """

    def __init__(self, dataset: Optional[Tensor] = None):
        super().__init__()

        self.register_buffer("bw_transform", Tensor())
        self.register_buffer("dataset", Tensor())
        self.register_buffer("norm", Tensor())
        
        if dataset is not None:
            self.fit(dataset)
        
        
    def forward(self, features: Tensor) -> Tensor:
        """Get the KDE estimates from the feature map.
        Args:
          features (Tensor): Feature map extracted from the CNN
        Returns: KDE Estimates
        """
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0]).to(features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = self.log_norm - (embedding / 2)
            estimate[i] = torch.mean(embedding)
        return estimate


    def fit(self, dataset: Tensor) -> None:
        """Fit a KDE model to the input dataset.
        Args:
          dataset (Tensor): Input dataset.
        Returns:
            None
        """        
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2
        
        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)
        
        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm
        self.log_norm = torch.log(self.norm)
        return


    @staticmethod
    def cov(tensor: Tensor) -> Tensor:
        """Calculate the unbiased covariance matrix.
        Args:
            tensor (Tensor): Input tensor from which covariance matrix is computed.
        Returns:
            Output covariance matrix.
        """
        mean = torch.mean(tensor, dim=1, keepdim=True)
        cov = torch.matmul(tensor - mean, (tensor - mean).T) / (tensor.size(1) - 1)
        return cov
    

def compute_kde(diffusion, multimodal_dict, model, logger, cfg):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def get_prediction(data, model_select):
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        # traj_np = data[:50, :, 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)
        
        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    num_samples = multimodal_dict['num_samples']
    
    K = 1000
    nll_list = []
    batch_size = 100
    iters = math.ceil(num_samples / batch_size)    
    for it in range(iters):
        if (it+1)*batch_size < num_samples:
            gt_group_it = gt_group[it*batch_size:(it+1)*batch_size]
            data_group_it = data_group[it*batch_size:(it+1)*batch_size]
        else:
            gt_group_it = gt_group[it*batch_size:]
            data_group_it = data_group[it*batch_size:]
                    
        pred = []
        for i in tqdm(range(0, K), position=0):
            # It generates a prediction for all samples in the test set
            # So we need loop for K times
            pred_i_nd = get_prediction(data_group_it, model)   # (1, 5168, 125, 48)
            pred.append(pred_i_nd)
        
        pred = np.concatenate(pred, axis=0) # pred [1000, 5187, 125, 48] in h36m
        pred = pred[:, :, cfg.t_his:, :] 

        try:
            gt_group = torch.from_numpy(gt_group_it).to('cuda')
            pred = torch.from_numpy(pred).to('cuda')
        except:
            pass     

        pred = torch.swapaxes(pred, 0, 1)
        pred = torch.reshape(pred, shape=(pred.shape[0], pred.shape[1], pred.shape[2], -1, 3))
        
        gt_group = torch.reshape(gt_group, shape=(gt_group.shape[0], gt_group.shape[1], -1, 3))
        gt_group = gt_group[:, None, ...]

        for idx in range(batch_size):
            kde_ll = kde(gt_group[idx:idx+1], pred[idx:idx+1])
            nll_list.append(kde_ll)
                
    kde_ll = torch.cat(nll_list, dim=0).mean(dim=0)
    kde_ll_np = kde_ll.to('cpu').numpy()
    print(kde_ll_np) 
    
    
    '''
    K = 1000
    pred = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group, model)   # (1, 5168, 125, 48)
        pred.append(pred_i_nd)
    
    pred = np.concatenate(pred, axis=0) # pred [1000, 5187, 125, 48] in h36m
    pred = pred[:, :, cfg.t_his:, :]
    
    try:
        gt_group = torch.from_numpy(gt_group).to('cuda')
        pred = torch.from_numpy(pred).to('cuda')
    except:
        pass
    
    pred = torch.swapaxes(pred, 0, 1)
    pred = torch.reshape(pred, shape=(pred.shape[0], pred.shape[1], pred.shape[2], -1, 3))
    
    gt_group = torch.reshape(gt_group, shape=(gt_group.shape[0], gt_group.shape[1], -1, 3))
    gt_group = gt_group[:, None, ...]

    
    nll_list = []
    for idx in range(num_samples):
        kde_ll = kde(gt_group[idx:idx+1], pred[idx:idx+1])
        nll_list.append(kde_ll)
            
    kde_ll = torch.cat(nll_list, dim=0).mean(dim=0)
    kde_ll_np = kde_ll.to('cpu').numpy()
    print(kde_ll_np)
    '''            