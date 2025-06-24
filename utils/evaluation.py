import csv
import time
import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

from FID.fid_classifier import classifier_fid_factory
from FID.fid import fid

from thop import profile
from thop import clever_format

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


class OneHumanMAC(torch.nn.Module):
    def __init__(self, diffusion) -> None:
        super().__init__()
        self.diffusion = diffusion
    
    def forward(self, traj, cfg, model_select):
        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = self.diffusion.sample_ddim(model_select, traj_dct, traj_dct_cond, mode_dict)
        return sampled_motion
    


def compute_stats(diffusion, multimodal_dict, model, logger, cfg):
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
        
        one_humanmac = OneHumanMAC(diffusion)
        macs, params = profile(one_humanmac, inputs=(traj, cfg, model_select))
        macs, params = clever_format([macs, params], "%.3f")
        print("flops = ", macs * 2) 
        print("params = ", params)

        print('warm up ... \n')
        for _ in range(20):
            start = time.time()
            outputs = one_humanmac(traj, cfg, model_select)
            torch.cuda.synchronize()
            end = time.time()
            print('Time:{}ms'.format((end-start)*1000))
            torch.cuda.empty_cache()

        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
            outputs = one_humanmac(traj, cfg, model_select)
        print(prof.key_averages().table(sort_by="cuda_time_total"))  
        

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    stats_meter = {x: {y: AverageMeter() for y in ['HumanMAC']} for x in stats_names}

    # K = 50
    K = 5
    pred = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group, model)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            # pred [50, 5187, 125, 48] in h36m
            pred = pred[:, :, cfg.t_his:, :]
            # Use GPU to accelerate        
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass
            
            '''
            _pred =  torch.swapaxes(pred, -2, -1)
            _gt = torch.swapaxes(gt_group, -2, -1)
            # _gt = gt_group

            classifier = classifier_fid_factory(_pred.device)
            pred_list = []
            for i in range(_pred.shape[0]):
                pred_activations = classifier.get_fid_features(motion_sequence=_pred[i]).cpu().data.numpy()
                pred_list.append(pred_activations)
            pred_list = np.concatenate(pred_list, 0)    
        
            gt_activations = classifier.get_fid_features(motion_sequence=_gt)
            gt_list = gt_activations.repeat(K, 1, 1).cpu().data.numpy()
            
            pred_list = np.reshape(pred_list, newshape=(-1, pred_list.shape[-1]))
            gt_list = np.reshape(gt_list, newshape=(-1, gt_list.shape[-1]))
                
            results_fid = fid(pred_list, gt_list)
            print(results_fid)
            '''
            
            # pred [50, 5187, 100, 48]
            for j in range(0, num_samples):
                apd, ade, fde, mmade, mmfde = compute_all_metrics(pred[:, j, :, :],
                                                                        gt_group[j][np.newaxis, ...],
                                                                        traj_gt_arr[j])
                stats_meter['APD']['HumanMAC'].update(apd)
                stats_meter['ADE']['HumanMAC'].update(ade)
                stats_meter['FDE']['HumanMAC'].update(fde)
                stats_meter['MMADE']['HumanMAC'].update(mmade)
                stats_meter['MMFDE']['HumanMAC'].update(mmfde)
            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                logger.info(str_stats)
            pred = []

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['HumanMAC'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['HumanMAC'] = new_meter['HumanMAC'].cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['HumanMAC']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)
