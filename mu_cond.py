import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from timm.utils import AverageMeter
from tqdm.auto import tqdm
from hugf_diffusers.ddpm import DDPMPipeline_cond
from hugf_diffusers.ddim import DDIMPipeline_cond

import os
import gc
import time
import numpy as np
import pickle
import copy
import torch_utils
from dnnlib.util import open_url
from utils.metrics import calculate_frechet_distance, calculate_inception_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from utils.fid_evaluation import FIDEvaluation


def myeval(args, epoch, pipeline, save_path, dataloader, n_cls, device, InS=False):
    # Sample some images from random noise (this is the backward diffusion process).
    num_per_cls = int(np.ceil(args.eval_samples / n_cls))
    num_sampling_rounds = int(np.ceil(num_per_cls / args.eval_batch_size))
    FIDs = []
    ISs = []

    fid_ = FrechetInceptionDistance(normalize=True).to(device)
    is_ = InceptionScore(normalize=True).to(device)
    for i in range(n_cls): # per class
        if args.dataset == 'CelebA' and (i == 0 or i == 18):
            FIDs.append(0.)
            ISs.append(0.)
            continue

        fids_ = 0.
        iss_ = 0.
        for j in range(num_sampling_rounds): # per round
            labels = torch.full((args.eval_batch_size,), i, dtype=torch.long, device=device) # cls i
            # print("conditional labels", labels[0].item())

            images = pipeline(
                n_cls, args.w, img_resol=args.image_size,
                labels=labels, batch_size = args.eval_batch_size, num_inference_steps=args.inference_T,
                generator=torch.Generator(device=device).manual_seed(args.seed+i*num_sampling_rounds+j)
            ).images
            gc.collect()
            torch.cuda.empty_cache()

            # save the images
            test_dir = os.path.join(save_path, f'samples/{str(i)}/{epoch:04d}')
            os.makedirs(test_dir, exist_ok=True)
            save_image(images, f'{test_dir}/{str(j)}.png')
            gc.collect()
            torch.cuda.empty_cache()

            # compute FID and IS
            # images = next(iter(dataloader))[0].to(device)
            fid_.update(images, real=False)
            is_.update(images)

            total_processed = 0
            for real_batch in dataloader:
                realImgs, real_labels = real_batch[0].to(device), real_batch[1].to(device)
                idx = (real_labels == i)
                if idx.sum() == 0:
                    continue
                real_imgs = realImgs[idx]
                total_processed += real_imgs.shape[0]
                if total_processed < images.shape[0]:
                    fid_.update(real_imgs, real=True)
                else:
                    fid_.update(real_imgs[:images.shape[0]-total_processed], real=True)
                    break
            gc.collect()
            torch.cuda.empty_cache()

        fids_ = float(fid_.compute())
        iss_ = float(is_.compute()[0])
        fid_.reset()
        is_.reset()
        print(f'FID: {fids_}, IS: {iss_}')

        FIDs.append(fids_)
        ISs.append(iss_)
    return FIDs, ISs


def eval_cond_(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=False):
    # Sample some images from random noise (this is the backward diffusion process).

    ## load inception model
    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        fid_model = pickle.load(f).to(device)
        fid_model.eval()

    num_per_cls = int(np.ceil(args.eval_samples / n_cls))
    num_sampling_rounds = int(np.ceil(num_per_cls / args.eval_batch_size))
    FIDs = []
    ISs = []
    for i in range(n_cls): # per class

        if args.dataset == 'CelebA' and (i == 0 or i == 18):
            FIDs.append(0.)
            ISs.append(0.)
            continue

        pred_arr = []
        prob_arr = []
        for j in range(num_sampling_rounds): # per round
            labels = torch.full((args.eval_batch_size,), i, dtype=torch.long, device=device) # cls i
            # print("conditional labels", labels[0].item())

            images = pipeline(
                n_cls, args.w, labels=labels,
                batch_size = args.eval_batch_size,
                num_inference_steps=args.inference_T,
                generator=torch.Generator(device=device).manual_seed(args.seed+i*num_sampling_rounds+j)
            ).images
            gc.collect()
            torch.cuda.empty_cache()

            # save the images
            test_dir = os.path.join(save_path, f'samples/{str(i)}/{epoch:04d}')
            os.makedirs(test_dir, exist_ok=True)
            save_image(images, f'{test_dir}/{str(j)}.png')
            gc.collect()
            torch.cuda.empty_cache()

            # compute FID and IS
            inp = (images * 255.).to(torch.uint8)
            pred = fid_model(inp, return_features=True) # [batch_size, 2048]
            pred_arr.append(pred.cpu().numpy())
            gc.collect()
            torch.cuda.empty_cache()
            if InS:
                prob = fid_model(inp, return_features=False) # [batch_size, 1008]
                prob_arr.append(prob.cpu().numpy())
            gc.collect()
            torch.cuda.empty_cache()

        pred_arr = np.concatenate(pred_arr, axis=0)
        pred_arr = pred_arr[:args.eval_samples]
        pred_mu = np.mean(pred_arr, axis=0)
        pred_sigma = np.cov(pred_arr, rowvar=False)
        ## load gt_mu and gt_sigma
        GT = np.load(states_file+str(i)+'.npz')
        gt_mu = GT['mu']
        gt_sigma = GT['sigma']
        FID = calculate_frechet_distance(gt_mu, gt_sigma, pred_mu, pred_sigma)

        if InS:
            prob_arr = np.concatenate(prob_arr, axis=0)
            prob_arr = prob_arr[:args.eval_samples]
            IS, _ = calculate_inception_score(prob_arr)
        else:
            IS = 0.

        FIDs.append(FID)
        ISs.append(IS)
    return FIDs, ISs


def eval_uncond(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=False):
    # Sample some images from random noise (this is the backward diffusion process).

    ## load inception model
    fid_score = FIDEvaluation(
        args.eval_batch_size,
        pipeline,
        args.inference_T,
        n_cls,
        args.w,
        channels=3,
        stats_dir=states_file,
        device=device,
        num_fid_samples=args.eval_samples,
        inception_block_idx=2048,
        seed=args.seed,
        save_path=save_path,
        epoch=epoch,
    )

    FIDs = fid_score.fid_score_uncond()
    return FIDs, None


def eval_cond(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=False):
    # Sample some images from random noise (this is the backward diffusion process).

    ## load inception model
    fid_score = FIDEvaluation(
        args.eval_batch_size,
        pipeline,
        args.inference_T,
        n_cls,
        args.w,
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
          headstr, save_path, states_file, device, train_num, accelerator, num_update_steps_per_epoch, n_cls):

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
        # model.train()
        for step, (gt_imgs, gt_labs) in enumerate(trainloader):
            clean_images = gt_imgs.to(device)
            labels = gt_labs.to(device)

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
                noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]
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
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
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
            # unet.eval()
            if args.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())
            if args.alg == 'DDPM':
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError

            InS = True if epoch == args.num_epochs - 1 else False
            FIDs, ISs = eval_cond(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=InS)
            mFID = np.mean(FIDs)
            print(f'mFID: {mFID}')
            if best_fid > mFID:
                best_fid = mFID
                best_epoch = epoch

            if args.use_ema:
                ema_model.restore(unet.parameters())
            # unet.train()

        accelerator.wait_for_everyone()

    accelerator.end_training()

    return train_time, FIDs, ISs, best_epoch
    # return train_time, None, None, 0


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
             headstr, save_path, device, train_num, trainloader_unl, total_unlbatch, accelerator, states_file, n_cls):
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
                for batch_idx, (unl_imgs, unl_labs) in enumerate(trainloader_unl):
                    x_0 = unl_imgs.to(device)
                    y_t = unl_labs.to(device)

                    # # Sample normal noise to add to the images
                    noise = torch.rand_like(x_0).to(device)
                    # # or sample noise with mean 0.5 and var 1.0 to the images
                    # noise = (torch.randn(x_0.shape) * 1.0 + 0.5).to(device)
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (x_0.shape[0],), device=x_0.device
                    ).long()
                    # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)

                    with accelerator.accumulate(model):
                        # Predict the noise residual
                        noise_pred = model(noisy_images, timesteps, y_t, return_dict=False)[0]
                        unl_loss = F.mse_loss(noise_pred, noise)
                        accelerator.backward(unl_loss)
                        optimizer.step()
                        optimizer.zero_grad()
                    unl_losses.update(unl_loss, x_0.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()
                    del x_0, y_t, noise, timesteps
                    del noise_pred, noisy_images, unl_loss

            model = set_param(model, param_i) # keep \theta_i for f and q
            # Update \theta_{i+1}
            num = 0
            sign = 0
            train_losses = AverageMeter()
            dr_losses = AverageMeter()
            q_losses = AverageMeter()
            for batch_idx, (gt_imgs, gt_labs) in enumerate(trainloader_rem):
                images = gt_imgs.to(device)
                labels = gt_labs.to(device)
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
                    noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]
                    loss_dr = F.mse_loss(noise_pred, noise)
                torch.cuda.empty_cache()
                gc.collect()
                del images, labels, noise
                del noise_pred, noisy_images, timesteps

                # [2] Set q^(\theta)
                for j, (unl_imgs, unl_labs) in enumerate(trainloader_unl):
                    if (j + sign * total_unlbatch) == batch_idx:
                        x_0 = unl_imgs.to(device)
                        y_t = unl_labs.to(device)
                        # print(f'unlearn {y_t[0:10]}')

                        noise = torch.rand_like(x_0).to(device)
                        # noise = (torch.randn(x_0.shape) * 1.0 + 0.5).to(device)
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (x_0.shape[0],), device=x_0.device
                        ).long()
                        noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)

                        with accelerator.accumulate(model):
                            noise_pred = model(noisy_images, timesteps, y_t, return_dict=False)[0]
                            loss_du = F.mse_loss(noise_pred, noise)
                        if j == total_unlbatch - 1:
                            sign += 1
                        break
                    elif (j + sign * total_unlbatch) < batch_idx:
                        continue
                    torch.cuda.empty_cache()
                    gc.collect()
                    del x_0, y_t, noise, timesteps
                    del noise_pred, noisy_images
                loss_q = loss_du - unl_losses.avg.detach()  # line 2 in Algorithm 1 (BOME!)
                # if loss_q < args.loss_thr:
                #     continue

                # [3] Get lambda_bome
                if args.lambda_bome < 0:
                    optimizer.zero_grad()
                    # params = []
                    # for name, param in model.named_parameters():
                    #     # print(name, param.requires_grad, param.size())
                    #     if param.requires_grad:
                    #         params.append(param)
                    #     torch.cuda.empty_cache()
                    #     gc.collect()
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
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError

            InS = True if epoch == args.num_epochs - 1 else False
            FIDs, ISs = eval_cond(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=InS)
            mFID = np.mean(FIDs)
            print(f'mFID: {mFID}')
    return train_time, FIDs, ISs, args.unl_eps-1



# Blindspot
def getUnlDevNum(unl_dev):
    if unl_dev == '':
        return []
    unl_dev_list = []
    for dev in unl_dev.split('+'):
        unl_dev_list.append(int(dev))
    return unl_dev_list


def attention(x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()


def forget_loss(model_output, model_activations, proxy_output, proxy_activations, mask, AT_beta = 50):

    # loss = F.l1_loss(model_output[mask==1], proxy_output[mask==1])
    loss = F.mse_loss(model_output[mask==1], proxy_output[mask==1])
    if AT_beta > 0:
        at_loss = 0
        for i in range(len(proxy_activations)):
            at_loss = at_loss + AT_beta * attention_diff(model_activations[i][mask==1], proxy_activations[i][mask==1])
    else:
        at_loss = 0

    # print(f'loss: {loss}, at_loss: {at_loss}, 10*at_loss: {10. * at_loss}')
    total_loss = loss + at_loss

    return total_loss


def blind_spot(args, proxy_model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_a, train_r,
               headstr, save_path, states_file, device, num_examples, accelerator,
               num_update_steps_per_epoch, n_cls, save_path_root):
    train_time = 0.
    unl_clses = getUnlDevNum(args.unl_cls)

    # get the proxy model
    # proxy_time = train(args, proxy_model, ema_model, noise_scheduler, optimizer, lr_scheduler, train_r,
    #                    headstr, save_path, states_file, device, num_examples["trainset_rem"], accelerator,
    #                    num_update_steps_per_epoch, n_cls)[0]
    # train_time += proxy_time
    proxy_model.load_state_dict(torch.load(save_path + '/weights/ckp_99.pth')['model'])
    proxy_model = accelerator.unwrap_model(proxy_model)
    if args.use_ema:
        ema_model.store(proxy_model.parameters())
        ema_model.copy_to(proxy_model.parameters())
    proxy_model.eval()

    # get the unlearned model
    origin_model = copy.deepcopy(proxy_model).to(device)
    origin_model.load_state_dict(torch.load(save_path_root + '/ori/weights/ckp_1999.pth')['model'])
    optimizer = torch.optim.Adam(origin_model.parameters(), lr = args.blind_lr)
    origin_model.train()

    train_loss = AverageMeter()
    global_step = 0
    FIDs, ISs = None, None
    num_update_steps_per_epoch = np.ceil(len(train_a) / args.gradient_accumulation_steps)
    for epoch in range(args.blind_epochs):
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"{headstr}-blind_Epoch {epoch+args.num_epochs}")

        train_st = time.time()
        for step, (gt_imgs, gt_labs) in enumerate(train_a):
            clean_images = gt_imgs.to(device)
            labels = gt_labs.to(device)

            # Assign mask 1 to unlearned images, mask 0 to remaining images
            mask = torch.zeros_like(labels).to(device)
            for unl_cls in unl_clses:
                mask = mask + (labels == unl_cls)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # # Predict the noise residual
            # noise_pred = origin_model(noisy_images, timesteps, labels, return_dict=False)[0]
            # with torch.no_grad():
            #     proxy_pred = proxy_model(noisy_images, timesteps, labels, return_dict=False)[0]
            noise_pred, noise_acts = origin_model(noisy_images, timesteps, labels, return_dict=False, sign=True)
            with torch.no_grad():
                proxy_pred, proxy_acts = proxy_model(noisy_images, timesteps, labels, return_dict=False, sign=True)

            remain_loss = 0.
            if mask.sum() < bs:
                remain_loss = F.mse_loss(noise_pred[mask==0], noise[mask==0])
            proxy_loss = 0.
            if mask.sum() > 0:
                # proxy_loss = forget_loss(noise_pred, None, proxy_pred, None, mask, 0)
                proxy_loss = forget_loss(noise_pred, noise_acts, proxy_pred, proxy_acts, mask, 10.)

            coeff = mask.sum()/bs
            loss = coeff*proxy_loss + (1-coeff)*remain_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            train_loss.update(loss.detach().item(), bs)
            # print(f'loss: {loss}, proxy_loss: {proxy_loss}, remain_loss: {remain_loss}, coeff: {coeff}')
            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr'], "step": global_step}
            progress_bar.set_postfix(**logs)

        train_time += time.time() - train_st
        print('{} | Epoch: {}\tLoss: {:.4f}\tTrainImg: {}\tLR: {:.6f}'.format(
              headstr, epoch+args.num_epochs, train_loss.avg, num_examples["trainset_all"], optimizer.param_groups[0]['lr']))
        accelerator.log({"loss": train_loss.avg, "lr": optimizer.param_groups[0]['lr']}, step=epoch)

        # After each epoch, optionally sample some demo images with eval() and save the model
        if ((epoch + 1) % args.save_model_epoch == 0 or epoch == args.blind_epochs - 1) and accelerator.is_main_process:
            unet = accelerator.unwrap_model(origin_model)
            if args.alg == 'DDPM':
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError
            pipeline.save_pretrained(save_path)
            save_ckp = os.path.join(save_path, 'weights')
            os.makedirs(save_ckp, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_ckp, f'ckp_{epoch+args.num_epochs}.pth'))

        if ((epoch + 1) % args.save_image_epoch == 0 or epoch == args.blind_epochs - 1) and accelerator.is_main_process:
            unet = accelerator.unwrap_model(origin_model)
            if args.alg == 'DDPM':
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError
            InS = True if epoch == args.blind_epochs - 1 else False
            FIDs, ISs = eval_cond(args, epoch+args.num_epochs, pipeline, save_path, states_file, n_cls, device, InS=InS)
            mFID = np.mean(FIDs)
            print(f'mFID: {mFID}')

    return train_time, FIDs, ISs, args.blind_epochs - 1 + args.num_epochs




def one_level(args, model, noise_scheduler, optimizer, trainloader_rem,
             headstr, save_path, device, train_num, trainloader_unl, total_unlbatch, accelerator, states_file, n_cls):

    train_time = 0.
    for epoch in range(args.unl_eps):
        train_st = time.time()
        num = 0
        sign = 0
        train_losses = AverageMeter()
        for batch_idx, (gt_imgs, gt_labs) in enumerate(trainloader_rem):
            images = gt_imgs.to(device)
            labels = gt_labs.to(device)
            num += gt_imgs.size(0)

            noise = torch.randn(images.shape).to(device) # Gaussian noise
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]
                loss_dr = F.mse_loss(noise_pred, noise)
            torch.cuda.empty_cache()
            gc.collect()
            del images, labels, noise
            del noise_pred, noisy_images, timesteps

            for j, (unl_imgs, unl_labs) in enumerate(trainloader_unl):
                if (j + sign * total_unlbatch) == batch_idx:
                    x_0 = unl_imgs.to(device)
                    y_t = unl_labs.to(device)
                    # print(f'unlearn {y_t[0:10]}')

                    noise = torch.randn(x_0.shape).to(device) # Gaussian noise
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (x_0.shape[0],), device=x_0.device
                    ).long()
                    noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)

                    with accelerator.accumulate(model):
                        noise_pred = model(noisy_images, timesteps, y_t, return_dict=False)[0]
                        loss_du = F.mse_loss(noise_pred, noise)
                    if j == total_unlbatch - 1:
                        sign += 1
                    break
                elif (j + sign * total_unlbatch) < batch_idx:
                    continue
                torch.cuda.empty_cache()
                gc.collect()
                del x_0, y_t, noise, timesteps
                del noise_pred, noisy_images

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = 10 * loss_dr - args.lambda_bome * loss_du
                accelerator.backward(loss)
                optimizer.step()
            train_losses.update(loss.detach().item(), gt_imgs.size(0))

            print(f'batch_idx: {batch_idx}, loss_du: {loss_du}, loss_dr: {loss_dr}, loss: {loss}')
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
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError

            InS = True if epoch == args.num_epochs - 1 else False
            FIDs, ISs = eval_cond(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=InS)
            mFID = np.mean(FIDs)
            print(f'mFID: {mFID}')
    return train_time, FIDs, ISs, args.unl_eps-1



def train_AG_max(args, model, noise_scheduler, optimizer, trainloader_rem,
                 headstr, save_path, device, train_num, trainloader_unl, total_unlbatch, accelerator, states_file, n_cls):
    train_time = 0.
    optimizer_u = torch.optim.Adam(model.parameters(), lr=1e-4, maximize=True)
    for epoch in range(args.unl_eps):
        train_st = time.time()
        for s_step in range(args.S_steps):
            param_i = get_param(model) # get \theta_i
            # T-steps: get \theta_i^K, train over the data via the ambiguous labels, line 1 in Algorithm 1 (BOME!)
            for j in range(args.K_steps):
                unl_losses = AverageMeter()
                for batch_idx, (unl_imgs, unl_labs) in enumerate(trainloader_unl):
                    x_0 = unl_imgs.to(device)
                    y_t = unl_labs.to(device)

                    noise = torch.randn(x_0.shape).to(device) # Gaussian noise
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (x_0.shape[0],), device=x_0.device
                    ).long()
                    # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)

                    with accelerator.accumulate(model):
                        # Predict the noise residual
                        noise_pred = model(noisy_images, timesteps, y_t, return_dict=False)[0]
                        unl_loss = F.mse_loss(noise_pred, noise)
                        accelerator.backward(unl_loss)
                        optimizer_u.step()
                        optimizer_u.zero_grad()
                    unl_losses.update(unl_loss, x_0.size(0))
                    torch.cuda.empty_cache()
                    gc.collect()
                    del x_0, y_t, noise, timesteps
                    del noise_pred, noisy_images, unl_loss

            model = set_param(model, param_i) # keep \theta_i for f and q
            # Update \theta_{i+1}
            num = 0
            sign = 0
            train_losses = AverageMeter()
            dr_losses = AverageMeter()
            q_losses = AverageMeter()
            for batch_idx, (gt_imgs, gt_labs) in enumerate(trainloader_rem):
                images = gt_imgs.to(device)
                labels = gt_labs.to(device)
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
                    noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]
                    loss_dr = F.mse_loss(noise_pred, noise)
                torch.cuda.empty_cache()
                gc.collect()
                del images, labels, noise
                del noise_pred, noisy_images, timesteps

                # [2] Set q^(\theta)
                for j, (unl_imgs, unl_labs) in enumerate(trainloader_unl):
                    if (j + sign * total_unlbatch) == batch_idx:
                        x_0 = unl_imgs.to(device)
                        y_t = unl_labs.to(device)
                        # print(f'unlearn {y_t[0:10]}')

                        noise = torch.randn(x_0.shape).to(device) # Gaussian noise
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (x_0.shape[0],), device=x_0.device
                        ).long()
                        noisy_images = noise_scheduler.add_noise(x_0, noise, timesteps)

                        with accelerator.accumulate(model):
                            noise_pred = model(noisy_images, timesteps, y_t, return_dict=False)[0]
                            loss_du = F.mse_loss(noise_pred, noise)
                        if j == total_unlbatch - 1:
                            sign += 1
                        break
                    elif (j + sign * total_unlbatch) < batch_idx:
                        continue
                    torch.cuda.empty_cache()
                    gc.collect()
                    del x_0, y_t, noise, timesteps
                    del noise_pred, noisy_images
                loss_q = unl_losses.avg.detach() - loss_du  # line 2 in Algorithm 1 (BOME!)

                # [3] Get lambda_bome
                if args.lambda_bome < 0:
                    optimizer.zero_grad()
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
                pipeline = DDPMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            elif args.alg == 'DDIM':
                pipeline = DDIMPipeline_cond(unet=unet, scheduler=noise_scheduler)
            else:
                raise NotImplementedError

            InS = True if epoch == args.num_epochs - 1 else False
            FIDs, ISs = eval_cond(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=InS)
            mFID = np.mean(FIDs)
            print(f'mFID: {mFID}')
    return train_time, FIDs, ISs, args.unl_eps-1

