import torch
from torch.nn import functional as F
from timm.utils import AverageMeter
import numpy as np
from tqdm.auto import tqdm
import time
import copy
import os

from ..mu_acc import train, evaluate
from ..classifier import getUnlDevNum
from ..hugf_diffusers.cond_ddpm import ConditionDDPMPipeline
from ..hugf_diffusers.cond_ddim import ConditionDDIMPipeline



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

    loss = F.l1_loss(model_output[mask], proxy_output[mask])
    if AT_beta > 0:
        at_loss = 0
        for i in range(len(proxy_activations)):
            at_loss = at_loss + AT_beta * attention_diff(model_activations[i][mask], proxy_activations[i][mask])
    else:
        at_loss = 0

    total_loss = loss + at_loss

    return total_loss


def blind_spot(args, proxy_model, noise_scheduler, optimizer, train_a, train_r, lr_scheduler,
               headstr, save_path, states_file, n_cls, device, num_examples, accelerator, save_path_root):
    train_time = 0.
    unl_clses = getUnlDevNum(args.unl_cls)

    # get the proxy model
    proxy_time = train(args, proxy_model, noise_scheduler, optimizer, train_r, lr_scheduler,
                       headstr, save_path, states_file, n_cls, device,
                       num_examples["trainset_rem"], accelerator)[0]
    proxy_model.eval()
    train_time += proxy_time

    # get the unlearned model
    origin_model = copy.deepcopy(proxy_model).to(device)
    origin_model.load_state_dict(torch.load(save_path_root + '/ori/weights/ckp_449.pth')['model'])
    optimizer = torch.optim.Adam(origin_model.parameters(), lr = args.blind_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.blind_epochs, eta_min=0, last_epoch=-1)
    origin_model.train()

    train_loss = AverageMeter()
    global_step = 0
    FIDs, ISs = None, None
    for epoch in range(args.blind_epochs):
        progress_bar = tqdm(total=len(train_a), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"{headstr}-Epoch {epoch}")

        train_st = time.time()
        for step, (gt_imgs, gt_labs) in enumerate(train_a):
            clean_images = gt_imgs.to(device)
            labels = gt_labs.to(device)

            # if images has labels in unl_cls, then give it mask 1, else 0
            mask = torch.zeros_like(labels).to(device)
            for unl_cls in range(unl_clses):
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

            # Predict the noise residual
            noise_pred = origin_model(noisy_images, timesteps, labels, return_dict=False)[0]
            with torch.no_grad():
                proxy_pred = proxy_model(noisy_images, timesteps, labels, return_dict=False)[0]

            remain_loss = 0.
            if mask.sum() < bs:
                remain_loss = F.mse(noise_pred[mask==0], labels[mask==0])
            proxy_loss = 0.
            if mask.sum() > 0:
                proxy_loss = forget_loss(noise_pred, None, proxy_pred, None, mask, 0)

            coeff = mask.sum()/bs
            loss = coeff*proxy_loss + (1-coeff)*remain_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            train_loss.update(loss.detach().item(), bs)
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr'], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        train_time += time.time() - train_st

        print('{} | Epoch: {}\tLoss: {:.4f}\tTrainImg: {}\tLR: {:.6f}'.format(
              headstr, epoch, train_loss.avg, num_examples["trainset_all"], optimizer.param_groups[0]['lr']))
        accelerator.log({"loss": train_loss.avg, "lr": optimizer.param_groups[0]['lr']}, step=epoch)
        # After each epoch, optionally sample some demo images with eval() and save the model
        if args.alg == 'DDPM':
            pipeline = ConditionDDPMPipeline(unet=origin_model, scheduler=noise_scheduler)
        elif args.alg == 'DDIM':
            pipeline = ConditionDDIMPipeline(unet=origin_model, scheduler=noise_scheduler)
        else:
            raise NotImplementedError

        if ((epoch + 1) % args.save_epochs == 0 or epoch == args.num_epochs - 1) and accelerator.is_main_process:
            FIDs, ISs = evaluate(args, epoch, pipeline, save_path, states_file, n_cls, device, InS=False)
            mFID = np.mean(FIDs)
            print(f'mFID: {mFID}')
            pipeline.save_pretrained(save_path)
            save_ckp = os.path.join(save_path, 'weights')
            os.makedirs(save_ckp, exist_ok=True)
            # accelerator.wait_for_everyone()
            torch.save({
                'epoch': epoch,
                'model': origin_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'fid': mFID,
            }, os.path.join(save_ckp, f'ckp_{epoch}.pth'))
            if best_fid > mFID:
                best_fid = mFID
                best_epoch = epoch

    return train_time, FIDs, ISs, best_epoch