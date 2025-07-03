import argparse
import sys

from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize, demo_visualize_v2
from utils.script import *

sys.path.append(os.getcwd())
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats, compute_stats_for_CMD
from utils.kde import compute_kde

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        default='h36m', help='h36m or humaneva')
    parser.add_argument('--mode', default='CMD', help='train / eval / pred / switch/ control/ zero_shot / kde/ vis /CMD')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--save_gif_interval', type=int, default=10)
    parser.add_argument('--save_metrics_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/h36m_ckpt.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=10)
    parser.add_argument('--vis_row', type=int, default=3)
    parser.add_argument('--action', type=str, default='directions')
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    if args.action != 'all':
        dataset, dataset_multi_test = dataset_split(cfg, [args.action])
    else:
        dataset, dataset_multi_test = dataset_split(cfg)

    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    """model"""
    model, diffusion = create_model_and_diffusion(cfg)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        # prepare full evaluation dataset
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            multimodal_dict=multimodal_dict,
            logger=logger,
            tb_logger=tb_logger)
        trainer.loop()

    elif args.mode == 'eval':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        # prepare full evaluation dataset
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats(diffusion, multimodal_dict, model, logger, cfg)
    
    elif args.mode == 'kde':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        multimodal_dict = get_gt(logger, dataset_multi_test, args, cfg)
        compute_kde(diffusion, multimodal_dict, model, logger, cfg)
    
    elif args.mode == 'vis':        
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize_v2(args.mode, cfg, model, diffusion, dataset, args.action)

    elif args.mode == 'CMD':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        if cfg.dataset == 'amass':
            multimodal_dict = get_multimodal_gt_full(logger, dataset['test'], args, cfg)
        else:
            multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats_for_CMD(diffusion, multimodal_dict, model, logger, cfg)
        
    else:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, diffusion, dataset)
