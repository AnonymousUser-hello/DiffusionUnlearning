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
from hugf_diffusers.ddpm import DDPMPipeline_cond
from hugf_diffusers.ddim import DDIMPipeline_cond
from mu_cond import train, train_AG, blind_spot, eval_cond, eval_uncond, one_level, train_AG_max
from hugf_diffusers.cond_unet import ConditionalUNet

from utils.util import (
    system_startup,
    set_random_seed,
    set_deterministic,
    Logger,
    count_parameters,
)
from data.load_data import load_data
from utils.mia_1 import compute_mia
from utils.mia import MIAClassifier, train_mia, test_mia
from classifier import predict, data_transform, getUnlDevNum
from utils.resnetc import resnet20
from utils.metrics import WeightDistance, calculate_wasserstein_distance
import copy
# from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Conditional Diffusion Models', add_help=False)
    parser.add_argument('--output_dir', type=str, default='./logs/DDIM/cond')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed_precision', type=str, default='no', choices=["no", "fp16", "bf16"])
    parser.add_argument('--alg', type=str, default='DDIM')
    # Data
    parser.add_argument('--data_path', type=str, default='/data/datasets')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--image_resolution', type=int, default=32)
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
    parser.add_argument('--w', type=float, default=0.1)
    # Eval
    parser.add_argument('--eva_state', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--eval_samples', type=int, default=50000)
    parser.add_argument('--inference_T', type=int, default=100)
    parser.add_argument('--save_model_epoch', type=int, default=100)
    parser.add_argument('--save_image_epoch', type=int, default=200)
    parser.add_argument('--eval_epoch', type=int, default=-1)
    parser.add_argument('--gold_epoch', type=int, default=1999)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    # Unlearn
    parser.add_argument('--unl_method', default='ori', type=str,
                        help='ori(Unscrubbed), base(Retrain), AG (bi-level with amibiguous labels), onelevel, AGmax')
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
    parser.add_argument('--blind_epochs', type=int, default=50)
    parser.add_argument('--blind_lr', type=float, default=2e-4)
    #
    parser.add_argument('--classifier', default=False, action='store_true')
    parser.add_argument('--MIA', default=False, action='store_true')
    parser.add_argument('--MIA_epochs', type=int, default=20)
    parser.add_argument('--MIA_lr', type=float, default=1e-5)
    parser.add_argument('--MIA_t', type=int, default=-1)
    parser.add_argument('--WD', default=False, action='store_true')
    parser.add_argument('--RT', default=False, action='store_true')
    parser.add_argument('--OOD', default=False, action='store_true')
    parser.add_argument('--KL', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args_parser()
    save_path_root = args.output_dir + '/' + args.dataset + '/' + str(args.unl_ratio)
    save_path = save_path_root + '/' + args.unl_method
    if args.unl_method == 'AG':
        save_path = save_path + '/' + str(args.rem_num) + '/E' + str(args.unl_eps) + '_S' + str(args.S_steps) + '_K' + str(args.K_steps) + '_eta' + str(args.eta_bome) + '_lambda' + str(args.lambda_bome)
    os.makedirs(save_path, exist_ok=True)
    sys.stdout = Logger(save_path + '/log.csv' , sys.stdout)
    print(args)
    device, setup = system_startup()
    set_random_seed(args.seed)
    set_deterministic()

    # Dataset
    num_classes, trainloader_all, trainloader_rem, trainloader_unl, \
        num_examples, trainset_rem, testset_rem, trainset_unl, testset_unl = load_data(args)

    # Model
    if args.image_resolution == 32:
        model = ConditionalUNet(
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
            class_embed_type="labelembed",  # the type of class token, add a class token for classifier-free guidance
            num_class_embeds=num_classes,  # the number of class tokens for classification guidance,
            dropout_prob=args.dropout_prob,  # dropout probability for including classifier-free guidance
            dropout=0.1,
            flip_sin_to_cos=False,
            freq_shift=1,
            attention_head_dim=None,
        )
    elif args.image_resolution == 64:
        model = ConditionalUNet(
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
            class_embed_type="labelembed",  # the type of class token, add a class token for classifier-free guidance
            num_class_embeds=num_classes,  # the number of class tokens for classification guidance,
            dropout_prob=args.dropout_prob,  # dropout probability for including classifier-free guidance
            dropout=0.1,
            flip_sin_to_cos=False,
            freq_shift=1,
            attention_head_dim=None,
        )
    else:
        model = ConditionalUNet(
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
            class_embed_type="labelembed",
            num_class_embeds=num_classes,
            dropout_prob=args.dropout_prob,
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
        if args.dataset == 'CIFAR10':
            noise_scheduler = DDIMScheduler(num_train_timesteps=args.train_T, beta_schedule="linear")
        else:
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
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch, num_classes)
        # Train from scratch
        elif args.unl_method == 'base':
            headstr = 'Retrain'
            train_num = num_examples["trainset_rem"]
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_r,
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch, num_classes)
        # Unlearn using Ambiguous Labels
        elif args.unl_method == 'AG':
            headstr = 'AG_BOME'
            train_num = args.rem_num if (args.partial_rem and (args.rem_num < num_examples["trainset_rem"])) else num_examples["trainset_rem"]
            unl_num = num_examples["trainset_unl"]
            total_unlbatch = unl_num // args.train_batch_size + 1
            train_time, FIDs, ISs, best_epoch = train_AG(args, model, noise_scheduler, optimizer, train_r,
                                                         headstr, save_path, device, train_num, train_u, total_unlbatch, accelerator,
                                                         states_file, num_classes)
        # Finetune
        elif args.unl_method == 'finetune':
            headstr = 'Finetune'
            train_num = num_examples["trainset_rem"]
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_r,
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch, num_classes)
            best_epoch = args.num_epochs - 1
        # NegGrad
        elif args.unl_method == 'neggrad':
            headstr = 'NegGrad'
            train_num = num_examples["trainset_unl"]
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_u,
                                                      headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch, num_classes)
            best_epoch = args.num_epochs - 1
        # Blindspot
        elif args.unl_method == 'blindspot':
            headstr = 'BlindSpot'
            train_num = num_examples["trainset_unl"]
            train_time, FIDs, ISs, best_epoch = blind_spot(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_a, train_r,
                                                           headstr, save_path, states_file, device, num_examples, accelerator,
                                                           num_update_steps_per_epoch, num_classes, save_path_root)
        # One-level
        elif args.unl_method == 'onelevel':
            headstr = 'Onelevel'
            train_num = args.rem_num if (args.partial_rem and (args.rem_num < num_examples["trainset_rem"])) else num_examples["trainset_rem"]
            unl_num = num_examples["trainset_unl"]
            total_unlbatch = unl_num // args.train_batch_size + 1
            train_time, FIDs, ISs, best_epoch = one_level(args, model, noise_scheduler, optimizer, train_r,
                                                          headstr, save_path, device, train_num, train_u, total_unlbatch, accelerator,
                                                          states_file, num_classes)
        elif args.unl_method == 'AGmax':
            headstr = 'AG_BOME_max'
            train_num = args.rem_num if (args.partial_rem and (args.rem_num < num_examples["trainset_rem"])) else num_examples["trainset_rem"]
            unl_num = num_examples["trainset_unl"]
            total_unlbatch = unl_num // args.train_batch_size + 1
            train_time, FIDs, ISs, best_epoch = train_AG_max(args, model, noise_scheduler, optimizer, train_r,
                                                             headstr, save_path, device, train_num, train_u, total_unlbatch, accelerator,
                                                             states_file, num_classes)
        # Finetune
        else:
            raise ValueError('unl_method should be ori, base or AG')

        print('Train time / Total time: {:.2f}/{:.2f} min'.format(train_time/60, (time.time() - st)/60))

        # eval_epoch = args.eval_epoch if args.eval_epoch > -1 else best_epoch
        eval_epoch = args.eval_epoch if args.eval_epoch > -1 else args.num_epochs - 1
        unet = accelerator.unwrap_model(model)
        # unet.load_state_dict(torch.load(save_path + f'/weights/ckp_{eval_epoch}.pth')['model'])
        if args.use_ema:
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
        if args.alg == 'DDPM':
            pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
        elif args.alg == 'DDIM':
            pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
        else:
            raise NotImplementedError
        if not args.train_state:
            FIDs, ISs = eval_cond(args, eval_epoch, pipeline, save_path, states_file, num_classes, device, InS=True)
        print(f'mFID: {np.mean(FIDs)}, FIDs: {FIDs}')
        # print(f'mIS: {np.mean(ISs)}, ISs: {ISs}')
        FIDs, ISs = eval_uncond(args, eval_epoch, pipeline, save_path, states_file, num_classes, device, InS=True)
        print(f'uncon_FID: {FIDs}')
        if args.use_ema:
            ema_model.restore(unet.parameters())

    torch.cuda.empty_cache()
    gc.collect()
    if args.eva_state:
        print('Start evaluating...')
        eval_epoch = args.eval_epoch if args.eval_epoch > -1 else args.num_epochs - 1
        unet = accelerator.unwrap_model(model)
        unet.load_state_dict(torch.load(save_path + f'/weights/ckp_{eval_epoch}.pth')['model'])

        # MIA over the unlearning dataset
        if args.MIA:
            print('Start MIA...')
            # load the target model
            unet.eval()
            testloader_unl = DataLoader(testset_unl, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            mia_score = compute_mia(unet, trainloader_unl, testloader_unl, noise_scheduler, device,
                                    args.unl_method, save_path, args.MIA_t)
            # testloader_rem = DataLoader(testset_rem, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            # trainloader_rem = DataLoader(trainset_rem, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
            # _ = compute_mia(unet, trainloader_rem, testloader_rem, noise_scheduler, device,
            #                 args.unl_method, save_path, args.MIA_t)
            torch.cuda.empty_cache()
            gc.collect()
            # attack_model = MIAClassifier().to(device)
            # train_mia(args, attack_model, unet, trainset_rem, testset_rem, noise_scheduler, device)
            # test_mia(args, attack_model, unet, trainset_unl, testset_unl, noise_scheduler, device, save_path, args.unl_method)
            # torch.cuda.empty_cache()
            # gc.collect()

        # load images generated by the diffusion model with the target model
        if args.classifier:
            num_per_cls = int(np.ceil(args.eval_samples / num_classes)) # 50000 / 10 = 5000
            num_sampling_rounds = int(np.ceil(num_per_cls / args.eval_batch_size)) # 5000 / 1024 = 5
            correct = 0
            total = 0
            # for i in range(num_classes):
            unl_clses = getUnlDevNum(args.unl_cls)
            for i in unl_clses:
                for j in range(num_sampling_rounds):
                    test_dir = os.path.join(save_path, f'samples/{str(i)}/{eval_epoch:04d}')
                    imgs = Image.open(test_dir + '/'+str(j)+'.png').convert('RGB') # C, H, W
                    imgs = data_transform(imgs).to(device)
                    # print('imgs shape', imgs.shape)

                    imgs_list = []
                    gap_pixel = 2
                    img_pixel = [args.image_resolution, args.image_resolution]
                    hirozontal_num = (imgs.size(2)-gap_pixel) // (img_pixel[1]+gap_pixel)
                    vertical_num = (imgs.size(1)-gap_pixel) // (img_pixel[0]+gap_pixel)
                    # print(f'hirozontal_num {hirozontal_num}, vertical_num {vertical_num}')
                    for row in range(2, imgs.size(1)-gap_pixel, img_pixel[0]+gap_pixel):
                        for col in range(2, imgs.size(2)-gap_pixel, img_pixel[1]+gap_pixel):
                            imgs_list.append(imgs[:, row:row+img_pixel[0], col:col+img_pixel[1]].unsqueeze(0))

                    torch.cuda.empty_cache()
                    gc.collect()
                    imgs = torch.cat(imgs_list, dim=0).to(device)
                    # print('cat imgs shape', imgs.shape)
                    # classifier the target model w.r.t the generted images
                    with torch.no_grad():
                        classifier = resnet20(num_classes=num_classes)
                        classifier = classifier.to(**setup)
                        classifier.load_state_dict(torch.load(args.output_dir + '/' + args.dataset + '/classifier/classifier.pth')['model'])
                        classifier.eval()
                        # correct_, total_ = predict(classifier, imgs, args.unl_cls)
                        correct_, total_ = predict(classifier, imgs, i)
                    torch.cuda.empty_cache()
                    gc.collect()
                    correct += correct_
                    total += total_
            print('unl_cls: {}/{} generated images have been classified as the label in unlearned data with acc {:.2f}%'.format(int(correct), int(total), (correct/total) * 100.))

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

        # generate out of distribution samples
        if args.OOD:
            if args.alg == 'DDPM':
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError
            # unet.load_state_dict(torch.load(save_path + f'/weights/ckp_{eval_epoch}.pth')['model'])
            # ood_cls = num_classes + 3 # not able to use (need to extend the label embedding)
            ood_cls = num_classes + 1
            for i in range(num_classes, ood_cls): # num_classes is the unconditional cls
                labels = torch.full((args.eval_batch_size,), i, dtype=torch.long).to(device)
                print("OOD labels", labels[0].item())
                images = pipeline(
                    num_classes, args.w, labels=labels,
                    batch_size = args.eval_batch_size,
                    num_inference_steps=args.inference_T,
                    generator=torch.Generator(device=device).manual_seed(args.seed+i)
                ).images
                gc.collect()
                torch.cuda.empty_cache()
                # save the images
                test_dir = os.path.join(save_path, f'samples/{str(i)}/{eval_epoch:04d}')
                os.makedirs(test_dir, exist_ok=True)
                save_image(images, f'{test_dir}/ood.png')
                gc.collect()
                torch.cuda.empty_cache()

        # relearn the target model
        if args.RT:
            model.load_state_dict(torch.load(save_path + f'/weights/ckp_{eval_epoch}.pth')['model'])
            # TODO: add reference_fid for early stop
            total_steps = len(trainloader_unl) * args.num_epochs
            num_update_steps_per_epoch = np.ceil(total_steps / args.gradient_accumulation_steps)
            train_time, FIDs, ISs, best_epoch = train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_u,
                                                      headstr, save_path, states_file, device,
                                                      num_examples['trainset_unl'], accelerator,
                                                      num_update_steps_per_epoch, num_classes)
            print(f'Relearn time: {train_time} and best epoch: {best_epoch}')
            print(f'mFID: {np.mean(FIDs)}, FIDs: {FIDs}')
            print(f'mIS: {np.mean(ISs)}, ISs: {ISs}')

            torch.cuda.empty_cache()
            gc.collect()

        if args.KL:
            print('Compute KL distance between the model output and the guassian noise.')
            unet.eval()

            import matplotlib.pyplot as plt
            def kl_divergence(mu1, var1, mu2, var2):
                term1 = 0.5 * (np.log(var2 / var1) + (var1 / var2))
                term2 = 0.5 * (((mu1 - mu2) ** 2) / var2 - 1)
                return term1 + term2

            with torch.no_grad():
                # data
                # dataloaders = {'train_rem': trainloader_rem, 'train_unl': trainloader_unl}
                dataloaders = {'$D_r$': trainloader_rem, '$D_f$': trainloader_unl}

                # model, cifar10
                unet.load_state_dict(torch.load(save_path_root + f'/AG/8192/E2_S2_K2_eta0.5_lambda0.1/weights/ckp_1.pth')['model'])
                unet.eval()
                torch.cuda.empty_cache()
                gc.collect()
                ori_unet = copy.deepcopy(unet).to(device)
                ori_unet.load_state_dict(torch.load(save_path_root + f'/ori/weights/ckp_1999.pth')['model'])
                ori_unet.eval()
                torch.cuda.empty_cache()
                gc.collect()
                base_unet = copy.deepcopy(unet).to(device)
                base_unet.load_state_dict(torch.load(save_path_root + f'/base/weights/ckp_1999.pth')['model'])
                base_unet.eval()
                torch.cuda.empty_cache()
                gc.collect()
                # finetune_unet = copy.deepcopy(unet).to(device)
                # finetune_unet.load_state_dict(torch.load(save_path_root + f'/finetune/weights/ckp_99.pth')['model'])
                # finetune_unet.eval()
                # torch.cuda.empty_cache()
                # gc.collect()
                # neggrad_unet = copy.deepcopy(unet).to(device)
                # neggrad_unet.load_state_dict(torch.load(save_path_root + f'/neggrad/weights/ckp_9.pth')['model'])
                # neggrad_unet.eval()
                # torch.cuda.empty_cache()
                # gc.collect()
                # blindspot_unet = copy.deepcopy(unet).to(device)
                # blindspot_unet.load_state_dict(torch.load(save_path_root + f'/blindspot/weights/ckp_199.pth')['model'])
                # blindspot_unet.eval()
                # torch.cuda.empty_cache()
                # gc.collect()
                # models = [ori_unet, unet, base_unet, finetune_unet, neggrad_unet, blindspot_unet]
                # models_name = ['Unscrubbed', 'Ours', 'Retrain', 'Finetune', 'Neggrad', 'Blindspot']
                models = [ori_unet, unet, base_unet]
                models_name = ['Unscrubbed', 'Ours', 'Retrain']
                ranges = [[0., 0.015], [0.015, 0.6]]

                # # model, utkface
                # unet.load_state_dict(torch.load(save_path_root + f'/AG/8192/E1_S2_K2_eta0.5_lambda0.1/weights/ckp_0.pth')['model'])
                # unet.eval()
                # torch.cuda.empty_cache()
                # gc.collect()
                # ori_unet = copy.deepcopy(unet).to(device)
                # ori_unet.load_state_dict(torch.load(save_path_root + f'/ori/weights/ckp_499.pth')['model'])
                # ori_unet.eval()
                # torch.cuda.empty_cache()
                # gc.collect()
                # base_unet = copy.deepcopy(unet).to(device)
                # base_unet.load_state_dict(torch.load(save_path_root + f'/base/weights/ckp_499.pth')['model'])
                # base_unet.eval()
                # torch.cuda.empty_cache()
                # gc.collect()
                # models = [ori_unet, unet, base_unet]
                # models_name = ['Unscrubbed', 'Ours', 'Retrain']
                # ranges = [[0., 0.001], [0.35, 0.45]]

                fi = 0
                for key in dataloaders:
                    print(f'Compute KL distance on {key}.')

                    KL_list = [[] for _ in range(len(models))]
                    cnt = 0
                    for i, (clean_images, labels) in enumerate(dataloaders[key]):
                        clean_images, labels = clean_images.to(device), labels.to(device)

                        # Sample noise to add to the images
                        noise = torch.randn(clean_images.shape).to(clean_images.device) # B, C, H, W
                        bs = clean_images.shape[0]

                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
                        ).long()
                        if args.MIA_t != -1:
                            timesteps = torch.ones_like(timesteps) * args.MIA_t
                        # Add noise to the clean images according to the noise magnitude at each timestep
                        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                        # Predict the noise residual
                        for j in range(len(models)):
                            noise_pred = models[j](noisy_images, timesteps, labels, return_dict=False)[0]
                            torch.cuda.empty_cache()
                            gc.collect()

                            # Calculate the KL divergence between the predicted noise and the Gaussian noise
                            noise_flat = noise.view(bs, -1)
                            noise_pred_flat = noise_pred.view(bs, -1)
                            mu1 = torch.mean(noise_flat, dim=1)
                            var1 = torch.var(noise_flat, dim=1)
                            mu2 = torch.mean(noise_pred_flat, dim=1)
                            var2 = torch.var(noise_pred_flat, dim=1)
                            kl_div = 0.5 * (-1.0 + torch.log(var2) - torch.log(var1) + torch.exp(torch.log(var1) - torch.log(var2)) + ((mu1 - mu2) ** 2) * torch.exp(-torch.log(var2))) # B
                            # kl_div = kl_divergence(mu1, var1, mu2, var2)

                            for kl in kl_div:
                                KL_list[j].append(kl.cpu().numpy())

                        cnt += bs
                        # if cnt > 10000:
                        #     break

                        del clean_images, labels, noise, noisy_images, noise_pred, kl_div
                        torch.cuda.empty_cache()
                        gc.collect()

                    rows = zip(KL_list[0], KL_list[1], KL_list[2])
                    import csv
                    files = ['kl_rem.csv', 'kl_unl.csv']
                    with open(files[fi], 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(models_name)
                        for row in rows:
                            writer.writerow(row)
                    fi += 1

                    print(f'Number of processed images: {cnt}')
                    kl_means = []
                    if key != '$D_f$':
                        fig = plt.figure(figsize=(8, 4))
                        for j in range(len(models)):
                            KL_array = np.array(KL_list[j])
                            KL_mean = np.mean(KL_array)
                            print(f'Model {models_name[j]} | mean {KL_mean}.')
                            # print(f'KL distance {KL_array}')
                            kl_means.append(KL_mean)

                            torch.cuda.empty_cache()
                            gc.collect()
                            if j < 2:
                                plt.hist(KL_array, density=True, bins=50, alpha=0.5, label=models_name[j])

                        ax = plt.gca()
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        plt.legend(frameon=False, fontsize=16)
                        plt.ylabel("Frequency", fontsize=16)
                        plt.yscale("log")
                        plt.xlabel("KL distance", fontsize=16)
                        plt.title(f"Mean of KL distances: {kl_means[0]:.4f} vs. {kl_means[1]:.4f} on {key}.", fontsize=16)

                    else: # shared y-axis, and log scale
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey='all')
                        fig.subplots_adjust(wspace=0.05)

                        KL_array = np.array(KL_list[0])
                        KL_mean = np.mean(KL_array)
                        print(f'Model {models_name[0]} | mean {KL_mean}.')
                        kl_means.append(KL_mean)
                        torch.cuda.empty_cache()
                        gc.collect()
                        ax1.hist(KL_array, density=True, bins=50, alpha=0.5, label=models_name[0])
                        ax2.hist(KL_array, density=True, bins=50, alpha=0.5, label=models_name[0])
                        ax1.set_yscale("log")

                        KL_array = np.array(KL_list[1])
                        KL_mean = np.mean(KL_array)
                        print(f'Model {models_name[1]} | mean {KL_mean}.')
                        kl_means.append(KL_mean)
                        torch.cuda.empty_cache()
                        gc.collect()
                        ax1.hist(KL_array, density=True, bins=50, alpha=0.5, label=models_name[1])
                        ax2.hist(KL_array, density=True, bins=50, alpha=0.5, label=models_name[1])
                        ax2.legend(frameon=False, fontsize=14)
                        ax2.set_yscale("log")

                        ax1.set_xlim(ranges[0])
                        ax2.set_xlim(ranges[1])
                        ax1.spines["top"].set_visible(False)
                        ax1.spines["right"].set_visible(False)
                        ax2.spines["top"].set_visible(False)
                        ax2.spines["left"].set_visible(False)
                        ax2.spines["right"].set_visible(False)
                        ax2.axes.get_yaxis().set_visible(False)
                        ax2.tick_params(axis='y',length=0)
                        d = 0.05
                        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,linestyle="none",color='k', mec='k', mew=1, clip_on=False)
                        ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
                        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
                        plt.suptitle(f"Mean of KL distances: {kl_means[0]:.4f} vs. {kl_means[1]:.4f} on {key}.", fontsize=16)
                        fig.add_subplot(111, frameon=False)
                        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
                        plt.xlabel("KL distance", fontsize=16)
                        plt.ylabel("Frequency", fontsize=16)
                        plt.yscale("log")

                        for j in range(len(models)):
                            KL_array = np.array(KL_list[j])
                            KL_mean = np.mean(KL_array)
                            print(f'Model {models_name[j]} | mean {KL_mean}.')

                    plt.tight_layout()
                    plt.savefig(f"{save_path_root}/kl_T{args.MIA_t+1}_{key}.png", dpi=300, bbox_inches="tight")

    torch.cuda.empty_cache()
    gc.collect()
    if (not args.train_state) and (not args.eva_state):
        eval_epoch = args.eval_epoch
        unet = accelerator.unwrap_model(model)
        if args.alg == 'DDPM':
            pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
        elif args.alg == 'DDIM':
            pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
        else:
            raise NotImplementedError
        FIDs, ISs = eval_cond(args, eval_epoch, pipeline, save_path, states_file, num_classes, device, InS=True)
        print(f'mFID: {np.mean(FIDs)}, FIDs: {FIDs}')
        FIDs, ISs = eval_uncond(args, eval_epoch, pipeline, save_path, states_file, num_classes, device, InS=True)
        print(f'uncon_FID: {FIDs}')

    print('Total time: {:.2f} min'.format((time.time() - st) / 60))
    print('Done!')
    print()

