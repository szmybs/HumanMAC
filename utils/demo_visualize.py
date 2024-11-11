import os
import numpy as np
import torch
from torch import tensor
from utils import *
from utils.pose_gen import pose_generator, pose_generator_v2
from utils.visualization import render_animation
from utils.script import sample_preprocessing
from utils.vis_pose import plt_row, plt_row_independent_save, plt_row_mixtures
from utils.vis_skeleton import VisSkeleton


def demo_visualize(mode, cfg, model, diffusion, dataset):
    """
    script for drawing gifs in different modes
    """
    if cfg.dataset != 'h36m' and mode != 'pred':
        raise NotImplementedError(f"sorry, {mode} is currently only available in h36m setting.")
    if mode == 'switch':
        for i in range(0, cfg.vis_switch_num):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg, mode='switch')
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col,
                             output=os.path.join(cfg.gif_dir, f'switch_{i}.gif'), mode=mode)

    elif mode == 'pred':
        action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        for i in range(0, len(action_list)):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode='pred', action=action_list[i], nrow=cfg.vis_row)
            suffix = action_list[i]
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'pred_{suffix}.gif'), mode=mode)

    elif mode == 'control':
        # draw part-body controllable results
        fix_name = ['right_leg', 'left_leg', 'torso', 'left_arm', 'right_arm', 'fix_lower', 'fix_upper']
        for i in range(0, 7):
            mode_fix = 'fix' + '_' + str(i)
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode=mode_fix, nrow=cfg.vis_row)
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, fix_name[i] + '.gif'), mode=mode, fix_index=i)
    elif mode == 'zero_shot':
        amass_data = np.squeeze(np.load('./data/amass_retargeted.npy'))
        for i in range(0, 15):
            pose_gen = pose_generator(amass_data, model, diffusion, cfg, mode=mode, nrow=cfg.vis_row)
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'zero_shot_{str(i)}.gif'), mode=mode)
    else:
        raise



def demo_visualize_v2(mode, cfg, model, diffusion, dataset, action):
    """
    script for drawing gifs in different modes
    """
    if cfg.dataset != 'h36m' and mode != 'pred':
        raise NotImplementedError(f"sorry, {mode} is currently only available in h36m setting.")

    total_num = 0
    vis_skeleton = VisSkeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                        16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])  
    removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
    vis_skeleton.remove_joints(removed_joints)
    vis_skeleton.adjust_connection_manually(([11, 8], [14, 8]))
    
    if action != 'all':
        save_subdir = cfg.action
    else:
        save_subdir = action
    save_dir = os.path.join(os.getcwd(), 'output/imgs/Human36M', save_subdir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("save_dir:" + str(save_dir))   

    data_set = dataset['test']
    data_gen = data_set.iter_generator(step=25)
    for i, data in enumerate(data_gen):
        # gt
        gt = data[0].copy()
        gt[:, :1, :] = 0
        data[:, :, :1, :] = 0
        
        gt = np.expand_dims(gt, axis=0)
        traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])  
        traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)      

        mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, cfg, mode=mode)
        sampled_motion = diffusion.sample_ddim(model, traj_dct, traj_dct_mod, mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = post_process(traj_est, cfg)
        
        gt_vis = gt[:, cfg.t_his:]
        gt_vis = gt_vis[:, [19, 39, 59, 79, 99]]
        traj_vis = traj_est[:, cfg.t_his:]
        traj_vis = traj_vis[:, [19, 39, 59, 79, 99]]
        traj_vis = traj_vis[None].swapaxes(1, 2)

        for j in range(traj_vis.shape[0]):
            mixtures_lists = []
            for p in range(traj_vis.shape[1]):
                mixtures_lists.append([])
                for q in range(traj_vis.shape[2]):
                    mixtures_lists[p].append(traj_vis[j, p, q])
            
            plt_row_mixtures(
                skeleton = vis_skeleton,
                pose = mixtures_lists,
                type = "3D",
                lcolor = "#3498db", rcolor = "#e74c3c",
                # view = (0, -180, -90),
                view = (-90, -180, -90),
                titles = None,
                add_labels = False, 
                only_pose = True,
                save_dir = save_dir, 
                save_name = 'MAC_' + str(total_num) + '_mix'
            )

            poses = [gt_vis[j,k] for k in range(gt_vis.shape[1])]
            plt_row_mixtures(
                skeleton = vis_skeleton,
                pose = poses,
                type = "3D",
                lcolor = "#3498db", rcolor = "#e74c3c",
                # view = (0, -180, -90),
                view = (-90, -180, -90),
                titles = None,
                add_labels = False, 
                only_pose = True,
                save_dir = save_dir, 
                save_name = 'MAC_' + str(total_num)
            )
            total_num += 1
