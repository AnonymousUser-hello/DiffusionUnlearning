import os
import gc
import sys
import time
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from hugf_diffusers.ddpm import DDPMPipeline_uncond
from hugf_diffusers.ddim import DDIMPipeline_uncond
from mu_uncond import train, train_AG, eval_uncond
from hugf_diffusers.unet import UnconditionalUNet

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger,
    count_parameters,
)
from data.load_data_uncond import load_data
from utils.mia_1 import compute_mia
from utils.mia import MIAClassifier, train_mia, test_mia
from utils.metrics import WeightDistance, calculate_wasserstein_distance
import copy
# from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Conditional Diffusion Models', add_help=False)
    parser.add_argument('--output_dir', type=str, default='./logs/DDIM/uncond')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed_precision', type=str, default='no', choices=["no", "fp16", "bf16"])
    parser.add_argument('--alg', type=str, default='DDIM')
    # Data
    parser.add_argument('--data_path', type=str, default='/data/datasets')
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--image_resolution', type=int, default=64)
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    # Train
    parser.add_argument('--train_state', default=False, action='store_true')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--pretrain_ckp', type=str, default='ori/weights/ckp_999.pth')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument('--train_T', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--use_ema', default=False, action='store_true')
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    # Eval
    parser.add_argument('--eva_state', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--eval_samples', type=int, default=50000)
    parser.add_argument('--inference_T', type=int, default=100)
    parser.add_argument('--save_model_epoch', type=int, default=100)
    parser.add_argument('--save_image_epoch', type=int, default=200)
    parser.add_argument('--eval_epoch', type=int, default=-1)
    parser.add_argument('--gold_epoch', type=int, default=1499)
    # Unlearn
    parser.add_argument('--unl_method', default='ori', type=str,
                        help='ori(Unscrubbed), base(Retrain), AG (bi-level with amibiguous labels)')
    parser.add_argument('--unl_ratio', default=0.1, type=float)
    parser.add_argument('--unl_cls', default='2+8', type=str)
    parser.add_argument('--unl_eps', type=int, default=2)
    parser.add_argument('--K_steps', default=1, type=int, help='K steps for LL')
    parser.add_argument('--S_steps', default=1, type=int, help='S steps for UL')
    parser.add_argument('--loss_thr', default=0.05, type=float)
    parser.add_argument('--eta_bome', type=float, default=0.01)
    parser.add_argument('--zta_bome', type=float, default=1e-4, help='lr')
    parser.add_argument('--partial_rem', default=False, action='store_true')
    parser.add_argument('--rem_num', type=int, default=2048)
    parser.add_argument('--lambda_bome', type=float, default=-1)
    #
    parser.add_argument('--MIA', default=False, action='store_true')
    parser.add_argument('--MIA_epochs', type=int, default=10)
    parser.add_argument('--MIA_lr', type=float, default=1e-3)
    parser.add_argument('--MIA_t', type=int, default=-1)
    parser.add_argument('--WD', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args_parser()
    save_path_root = args.output_dir + '/' + args.dataset + '/' + str(args.unl_ratio)
    save_path = save_path_root + '/' + args.unl_method
    if args.unl_method == 'AG':
        save_path = save_path + '/E' + str(args.unl_eps) + '_S' + str(args.S_steps) + '_K' + str(args.K_steps) + '_eta' + str(args.eta_bome) + '_lambda' + str(args.lambda_bome)
    os.makedirs(save_path, exist_ok=True)
    sys.stdout = Logger(save_path + '/log.csv' , sys.stdout)
    print(args)
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Dataset
    trainloader_all, trainloader_rem, trainloader_unl, \
        num_examples, trainset_rem, testset_rem, trainset_unl, testset_unl = load_data(args)

    # Model
    if args.image_resolution == 32:
        model = UnconditionalUNet(
            sample_size=args.image_resolution,  # image resolution of input/output samples, must be a multiple of 2**(len(block_out_channels)-1)
            in_channels=args.image_channel,  # the number of input channels
            out_channels=args.image_channel,  # the number of output channels
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "UpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
            ),
            block_out_channels=(128, 256, 256, 256),  # the number of output channels for each UNet block
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            norm_eps=1e-6,  # norm eps for batch norm
            downsample_padding=0,  # padding for downsampling layers
            dropout=0.1,
            flip_sin_to_cos=False,
            freq_shift=1,
            attention_head_dim=None,
        )
    elif args.image_resolution == 64:
        model = UnconditionalUNet(
            sample_size=args.image_resolution,  # image resolution of input/output samples, must be a multiple of 2**(len(block_out_channels)-1)
            in_channels=args.image_channel,  # the number of input channels
            out_channels=args.image_channel,  # the number of output channels
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(128, 256, 256, 256, 512),  # the number of output channels for each UNet block
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            norm_eps=1e-6,  # norm eps for batch norm
            downsample_padding=0,  # padding for downsampling layers
            dropout=0.1,
            flip_sin_to_cos=False,
            freq_shift=1,
            attention_head_dim=None,
        )
    else:
        model = UnconditionalUNet(
            sample_size=args.image_resolution,
            in_channels=args.image_channel,
            out_channels=args.image_channel,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(128, 128, 256, 256, 512, 512),
            layers_per_block=2,
            norm_eps=1e-6,
            downsample_padding=0,
            dropout=0.1,
            flip_sin_to_cos=False,
            freq_shift=1,
            attention_head_dim=None,
        )
    print(f"Number of parameters: {count_parameters(model)//1e6:.2f}M")
    if args.pretrained:
        loaded_dict = torch.load(save_path_root + '/' + args.pretrain_ckp)
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(loaded_dict['model'])
        else:
            model.load_state_dict(loaded_dict['model'])
        print('Loaded pretrained model.')

    # Optimizer and scheduler
    if args.unl_method == 'neggrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, maximize=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(trainloader_all) * args.num_epochs if args.unl_method == 'ori' else len(trainloader_rem) * args.num_epochs
    if args.lr_warmup_steps > 0 and args.lr_scheduler == "constant":
        args.lr_scheduler = "constant_with_warmup"
    if args.unl_method == 'AG':
        lr_scheduler = None
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps= total_steps,
        )
    if args.resume:
        # resume_path = os.path.join(save_path, f'/weights/ckp_{args.resume_epoch}.pth')
        resume_ckp = torch.load(save_path + f'/weights/ckp_{args.resume_epoch}.pth')
        model.load_state_dict(resume_ckp['model'])
        args.resume_epoch = resume_ckp['epoch']
        optimizer.load_state_dict(resume_ckp['optimizer'])
        lr_scheduler.load_state_dict(resume_ckp['scheduler'])
        print(f'Resum from epoch {args.resume_epoch}.')

    if args.alg == 'DDPM':
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.train_T)
    elif args.alg == 'DDIM':
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.train_T, beta_schedule="linear")
    else:
        raise NotImplementedError

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=save_path,
    )
    device = accelerator.device
    if accelerator.is_main_process:
        accelerator.init_trackers("train")
    model, optimizer, lr_scheduler, \
        train_a, train_r, train_u = accelerator.prepare(
            model, optimizer, lr_scheduler, trainloader_all, trainloader_rem, trainloader_unl)

    # EMA Model
    if args.use_ema:
        ema_model = EMAModel(model.parameters(), inv_gamma=args.ema_inv_gamma, power=args.ema_power, decay=args.ema_max_decay)
        ema_model.to(accelerator.device)
        if args.resume:
            resume_ckp = torch.load(save_path + f'/weights/ckp_{args.resume_epoch}.pth')
            ema_model.load_state_dict(resume_ckp['ema'])
    else:
        None

    # Train
    num_update_steps_per_epoch = np.ceil(len(trainloader_all) / args.gradient_accumulation_steps) if args.unl_method == 'ori' else np.ceil(len(trainloader_rem) / args.gradient_accumulation_steps)
    states_file = args.output_dir + '/' + args.dataset + '/GT/'
    st = time.time()
    best_epoch = 0
    if args.train_state:
        print('Start training...')
        # Unscrubbed model
        if args.unl_method == 'ori':
            headstr = 'Unscrubbed'
            train_num = num_examples["trainset_all"]
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_a,
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch)
        # Train from scratch
        elif args.unl_method == 'base':
            headstr = 'Retrain'
            train_num = num_examples["trainset_rem"]
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_r,
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch)
        # Unlearn using Ambiguous Labels
        elif args.unl_method == 'AG':
            headstr = 'AG_BOME'
            train_num = args.rem_num if (args.partial_rem and (args.rem_num < num_examples["trainset_rem"])) else num_examples["trainset_rem"]
            unl_num = num_examples["trainset_unl"]
            total_unlbatch = unl_num // args.train_batch_size + 1
            train_time, FIDs, ISs, best_epoch = train_AG(args, model, noise_scheduler, optimizer, train_r,
                                                         headstr, save_path, device, train_num, train_u, total_unlbatch, accelerator,
                                                         states_file)
        # Finetune
        elif args.unl_method == 'finetune':
            headstr = 'Finetune'
            train_num = num_examples["trainset_rem"]
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_r,
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch)
            best_epoch = args.num_epochs - 1
        # NegGrad
        elif args.unl_method == 'neggrad':
            headstr = 'NegGrad'
            train_num = num_examples["trainset_unl"]
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_u,
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch)
            best_epoch = args.num_epochs - 1
        # Blindspot
        elif args.unl_method == 'blindspot':
            raise NotImplementedError
        else:
            raise ValueError('unl_method should be ori, base or AG')

        print('Train time / Total time: {:.2f}/{:.2f} min'.format(train_time/60, (time.time() - st)/60))

    print('Start evaluating...')
    # eval_epoch = args.eval_epoch if args.eval_epoch > -1 else best_epoch
    eval_epoch = args.eval_epoch if args.eval_epoch > -1 else args.num_epochs - 1
    unet = accelerator.unwrap_model(model)
    if args.use_ema:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())
    if args.alg == 'DDPM':
        pipeline = DDPMPipeline_uncond(unet=unet, scheduler=noise_scheduler)
    elif args.alg == 'DDIM':
        pipeline = DDIMPipeline_uncond(unet=unet, scheduler=noise_scheduler)
    else:
        raise NotImplementedError
    if not args.train_state:
        unet.load_state_dict(torch.load(save_path + f'/weights/ckp_{eval_epoch}.pth')['model'])
        FIDs, ISs = eval_uncond(args, eval_epoch, pipeline, save_path, states_file, device, InS=True)
    print(f'uncon_FID: {FIDs}')
    if args.use_ema:
        ema_model.restore(unet.parameters())

    torch.cuda.empty_cache()
    gc.collect()
    if args.eva_state:
        unet = accelerator.unwrap_model(model)
        unet.load_state_dict(torch.load(save_path + f'/weights/ckp_{eval_epoch}.pth')['model'])

        # MIA over the unlearning dataset
        if args.MIA:
            print('Start MIA...')
            # load the target model
            unet.eval()
            testloader_unl = DataLoader(testset_unl, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            mia_score = compute_mia(unet, trainloader_unl, testloader_unl, noise_scheduler, device,
                                    args.unl_method, save_path)

            torch.cuda.empty_cache()
            gc.collect()
            attack_model = MIAClassifier().to(device)
            train_mia(args, attack_model, unet, trainset_rem, testset_rem, noise_scheduler, device)
            test_mia(args, attack_model, unet, trainset_unl, testset_unl, noise_scheduler, device, save_path, args.unl_method)
            torch.cuda.empty_cache()
            gc.collect()

        # compute the distances
        if args.WD:
            # gold model
            gold_model = copy.deepcopy(unet).to(device)
            gold_model.load_state_dict(torch.load(save_path_root + f'/base/weights/ckp_{args.gold_epoch}.pth')['model'])
            gold_model.eval()
            # target model
            # unet.load_state_dict(torch.load(save_path + f'/weights/ckp_{eval_epoch}.pth')['model'])
            unet.eval()
            # compute the distances
            weight_distance = WeightDistance(gold_model, unet)
            wasserstein_distance = calculate_wasserstein_distance(unet, gold_model, trainloader_unl, noise_scheduler, device)
            print('Weight distance: {:.6f}, Wassertein distance: {:.6f}'.format(weight_distance, wasserstein_distance))

            torch.cuda.empty_cache()
            gc.collect()


    print('Total time: {:.2f} min'.format((time.time() - st) / 60))
    print('Done!')
    print()

