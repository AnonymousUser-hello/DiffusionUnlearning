import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from timm.utils import AverageMeter
from tqdm.auto import tqdm
from hugf_diffusers.ddpm import DDPMPipeline_uncond
from hugf_diffusers.ddim import DDIMPipeline_uncond

import os
import gc
import time
import numpy as np
import pickle
import copy
from dnnlib.util import open_url
from utils.metrics import calculate_frechet_distance, calculate_inception_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from utils.fid_evaluation_uncond import FIDEvaluation


def eval_uncond(args, epoch, pipeline, save_path, states_file, device, InS=False):
    # Sample some images from random noise (this is the backward diffusion process).

    ## load inception model
    fid_score = FIDEvaluation(
        args.eval_batch_size,
        pipeline,
        args.inference_T,
        channels=3,
        stats_dir=states_file,
        device=device,
        num_fid_samples=args.eval_samples,
        inception_block_idx=2048,
        seed=args.seed,
        save_path=save_path,
        epoch=epoch,
    )

    FIDs = fid_score.fid_score()
    return FIDs, None



def train(args, model, ema_model, noise_scheduler, optimizer, lr_scheduler, trainloader,
          headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch):

    train_time = 0.
    start_epoch = 0
    best_epoch = 0
    best_fid = np.inf
    train_loss = AverageMeter()
    if args.resume:
        start_epoch = args.resume_epoch + 1
    global_step = (0+num_update_steps_per_epoch) * args.resume_epoch

    for epoch in range(start_epoch, args.num_epochs):
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"{headstr}-Epoch {epoch}")

        train_st = time.time()
        for step, (gt_imgs) in enumerate(trainloader):
            clean_images = gt_imgs.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # EMA update
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

            assert accelerator.sync_gradients, "Sync failed"
            train_loss.update(loss.detach().item(), bs)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)

        train_time += time.time() - train_st
        print('{} | Epoch: {}\tLoss: {:.4f}\tTrainImg: {}\tLR: {:.6f}'.format(
            headstr, epoch, train_loss.avg, train_num, lr_scheduler.get_last_lr()[0]))
        accelerator.log({"loss": train_loss.avg, "lr": lr_scheduler.get_last_lr()[0]}, step=epoch)

        progress_bar.close()
        accelerator.wait_for_everyone()

        # After each epoch, optionally sample some demo images with eval() and save the model
        if ((epoch + 1) % args.save_model_epoch == 0 or epoch == args.num_epochs - 1) and accelerator.is_main_process:
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

            pipeline.save_pretrained(save_path)
            save_ckp = os.path.join(save_path, 'weights')
            os.makedirs(save_ckp, exist_ok=True)
            states = {
                'epoch': epoch,
                'model': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'ema': ema_model.state_dict() if args.use_ema else None,
            }
            torch.save(states, os.path.join(save_ckp, f'ckp_{epoch}.pth'))

            if args.use_ema:
                ema_model.restore(unet.parameters())

        if ((epoch + 1) % args.save_image_epoch == 0 or epoch == args.num_epochs - 1) and accelerator.is_main_process:
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

            InS = True if epoch == args.num_epochs - 1 else False
            FIDs, ISs = eval_uncond(args, epoch, pipeline, save_path, states_file, device, InS=InS)
            print(f'mFID: {FIDs}')
            if best_fid > FIDs:
                best_fid = FIDs
                best_epoch = epoch

            if args.use_ema:
                ema_model.restore(unet.parameters())

        accelerator.wait_for_everyone()

    accelerator.end_training()

    return train_time, FIDs, ISs, best_epoch


def get_param(net):
    new_param = []
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            new_param.append(param.clone())
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return new_param


def set_param(net, old_param):
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            param.copy_(old_param[j])
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return net


def train_AG(args, model, noise_scheduler, optimizer, trainloader_rem,
             headstr, save_path, device, train_num, trainloader_unl, total_unlbatch, accelerator, states_file):
    # # if args.lambda_bome > 0:
    # save_path = save_path + '/E' + str(args.unl_eps) + '_S' + str(args.S_steps) + '_K' + str(args.K_steps) + '_eta' + str(args.eta_bome) + '_lambda' + str(args.lambda_bome)
    # os.makedirs(save_path, exist_ok=True)

    train_time = 0.
    for epoch in range(args.unl_eps):
        train_st = time.time()
        for s_step in range(args.S_steps):
            param_i = get_param(model) # get \theta_i
            # T-steps: get \theta_i^K, train over the data via the ambiguous labels, line 1 in Algorithm 1 (BOME!)
            for j in range(args.K_steps):
                unl_losses = AverageMeter()
                for batch_idx, (unl_imgs) in enumerate(trainloader_unl):
                    x_0 = unl_imgs.to(device)

                    # Sample normal noise to add to the images
                    noise = torch.rand_like(x_0).to(device)
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (x_0.shape[0],), device=x_0.device
                    ).long()
                    # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)

                    with accelerator.accumulate(model):
                        # Predict the noise residual
                        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                        unl_loss = F.mse_loss(noise_pred, noise)
                        accelerator.backward(unl_loss)
                        optimizer.step()
                        optimizer.zero_grad()
                    unl_losses.update(unl_loss, x_0.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()
                    del x_0, noise, timesteps
                    del noise_pred, noisy_images, unl_loss

            model = set_param(model, param_i) # keep \theta_i for f and q
            # Update \theta_{i+1}
            num = 0
            sign = 0
            train_losses = AverageMeter()
            dr_losses = AverageMeter()
            q_losses = AverageMeter()
            for batch_idx, (gt_imgs) in enumerate(trainloader_rem):
                images = gt_imgs.to(device)
                num += gt_imgs.size(0)
                # print(f'remain {labels[0:10]}')

                # [1] Get loss over the remaining data
                noise = torch.randn(images.shape).to(device) # Gaussian noise
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss_dr = F.mse_loss(noise_pred, noise)
                torch.cuda.empty_cache()
                gc.collect()
                del images, noise
                del noise_pred, noisy_images, timesteps

                # [2] Set q^(\theta)
                for j, (unl_imgs) in enumerate(trainloader_unl):
                    if (j + sign * total_unlbatch) == batch_idx:
                        x_0 = unl_imgs.to(device)
                        # print(f'unlearn {y_t[0:10]}')

                        noise = torch.rand_like(x_0).to(device)
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (x_0.shape[0],), device=x_0.device
                        ).long()
                        noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)

                        with accelerator.accumulate(model):
                            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                            loss_du = F.mse_loss(noise_pred, noise)
                        if j == total_unlbatch - 1:
                            sign += 1
                        break
                    elif (j + sign * total_unlbatch) < batch_idx:
                        continue
                    torch.cuda.empty_cache()
                    gc.collect()
                    del x_0, noise, timesteps
                    del noise_pred, noisy_images
                loss_q = loss_du - unl_losses.avg.detach()  # line 2 in Algorithm 1 (BOME!)
                # if loss_q < args.loss_thr:
                #     continue

                # [3] Get lambda_bome
                if args.lambda_bome < 0:
                    optimizer.zero_grad()
                    # compute the gradient of the loss_q w.r.t. the model parameters using gpu
                    q_grads = torch.autograd.grad(loss_q, model.parameters(), retain_graph=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    q_grad_vector = torch.stack(list(map(lambda q_grad: torch.cat(list(map(lambda grad: grad.contiguous().view(-1), q_grad))), [q_grads])))
                    torch.cuda.empty_cache()
                    gc.collect()
                    q_grad_norm = torch.linalg.norm(q_grad_vector, 2)
                    torch.cuda.empty_cache()
                    gc.collect()
                    if q_grad_norm == 0:
                        lambda_bome = 0.
                    else:
                        # compute the gradient of the loss_dr w.r.t. the model parameters
                        optimizer.zero_grad()
                        dr_grads = torch.autograd.grad(loss_dr, model.parameters(), retain_graph=True)
                        torch.cuda.empty_cache()
                        gc.collect()
                        # similarity between dr_grads_vector and q_grad_vector
                        # compute the inner product of the gradient of dr_grads and q_grads
                        dr_grads_vector = torch.stack(list(map(lambda dr_grad: torch.cat(list(map(lambda grad: grad.contiguous().view(-1), dr_grad))), [dr_grads])))
                        dr_grad_norm = torch.linalg.norm(dr_grads_vector, 2)
                        torch.cuda.empty_cache()
                        gc.collect()
                        inner_product = torch.sum(dr_grads_vector * q_grad_vector)
                        tmp = inner_product / ( dr_grad_norm * q_grad_norm + 1e-8)
                        # tmp = inner_product / (q_grad_norm + 1e-8) # original verison in BOME!
                        # compute the lambda_bome
                        lambda_bome = (args.eta_bome - tmp).detach() if args.eta_bome > tmp else 0.
                        print(f'lambda_bome {lambda_bome}, tmp {tmp}')
                        torch.cuda.empty_cache()
                        gc.collect()
                        del dr_grads, dr_grads_vector, tmp
                    del q_grads, q_grad_vector, q_grad_norm
                else:
                    lambda_bome = args.lambda_bome
                    print(f'lambda_bome {lambda_bome}')

                # [4] Update the model parameters # line 3 in Algorithm 1 (BOME!)
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss = loss_dr + lambda_bome * loss_q
                    accelerator.backward(loss)
                    optimizer.step()
                train_losses.update(loss.detach().item(), gt_imgs.size(0))
                dr_losses.update(loss_dr.detach().item())
                q_losses.update(loss_q.detach().item())
                print(f'batch_idx: {batch_idx}, unl_loss: {unl_losses.avg.detach()}, loss_du: {loss_du}, loss_q: {loss_q}, loss_dr: {loss_dr}, loss: {loss}')
                torch.cuda.empty_cache()
                gc.collect()

                if (num > args.rem_num-1) and args.partial_rem:
                    break

        train_time += time.time() - train_st
        print('{} | Epoch: {}\tLoss: {:.4f}\tTrainImg: {}'.format(headstr, epoch, train_losses.avg, train_num))

        if ((epoch + 1) % args.save_model_epoch == 0 or epoch == args.unl_eps - 1) and accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            save_ckp = os.path.join(save_path, 'weights')
            os.makedirs(save_ckp, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': unet.state_dict(),
            }, os.path.join(save_ckp, f'ckp_{epoch}.pth'))

        if ((epoch + 1) % args.save_image_epoch == 0 or epoch == args.unl_eps - 1) and accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            if args.alg == 'DDPM':
                pipeline = DDPMPipeline_uncond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_uncond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError

            InS = True if epoch == args.num_epochs - 1 else False
            FIDs, ISs = eval_uncond(args, epoch, pipeline, save_path, states_file, device, InS=InS)
            print(f'mFID: {FIDs}')
    return train_time, FIDs, ISs, args.unl_eps-1




