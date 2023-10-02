import torch
import numpy as np
import torch.distributed as dist
from scipy import linalg
from scipy.stats import wasserstein_distance


def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    m = np.square(mu1 - mu2).sum()
    s, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fd = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return fd


def compute_fid(n_samples, n_gpus, sampling_shape, sampler, inception_model, stats_paths, device, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))

    def generator(num_samples):
        num_sampling_rounds = int(
            np.ceil(num_samples / sampling_shape[0]))
        for _ in range(num_sampling_rounds):
            x = torch.randn(sampling_shape, device=device)

            if n_classes is not None:
                y = torch.randint(n_classes, size=(
                    sampling_shape[0],), dtype=torch.int32, device=device)
                x = sampler(x, y=y)

            else:
                x = sampler(x)

            x = (x / 2. + .5).clip(0., 1.)
            x = (x * 255.).to(torch.uint8)
            yield x

    act = get_activations(generator(num_samples_per_gpu),
                          inception_model, device=device, max_samples=n_samples)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    m = torch.from_numpy(mu).cuda()
    s = torch.from_numpy(sigma).cuda()
    average_tensor(m)
    average_tensor(s)

    all_pool_mean = m.cpu().numpy()
    all_pool_sigma = s.cpu().numpy()

    fid = []
    for stats_path in stats_paths:
        stats = np.load(stats_path)
        data_pools_mean = stats['mu']
        data_pools_sigma = stats['sigma']
        fid.append(calculate_frechet_distance(data_pools_mean,
                   data_pools_sigma, all_pool_mean, all_pool_sigma))
    return fid



def get_activations(dl, model, device, max_samples):
    pred_arr = []
    total_processed = 0

    print('Starting to sample.')
    for batch in dl:
        # ignore labels
        if isinstance(batch, list):
            batch = batch[0]

        batch = batch.to(device)
        if batch.shape[1] == 1:  # if image is gray scale
            batch = batch.repeat(1, 3, 1, 1)
        elif len(batch.shape) == 3:  # if image is gray scale
            batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

        with torch.no_grad():
            batch = batch * 0.5 + 0.5  # [0 ~ 1]
            batch = (batch.clip(0.,1.) * 255.).to(torch.uint8)
            pred = model(batch.to(device),
                         return_features=True).unsqueeze(-1).unsqueeze(-1)

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)
        total_processed += pred.shape[0]
        if max_samples is not None and total_processed > max_samples:
            print('Max of %d samples reached.' % max_samples)
            break

    pred_arr = np.concatenate(pred_arr, axis=0)
    if max_samples is not None:
        pred_arr = pred_arr[:max_samples]

    return pred_arr


def calculate_inception_score(probs, splits=10, use_torch=False):
    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        inception_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()
    else:
        inception_score, std = (np.mean(scores), np.std(scores))
    del probs, scores
    return inception_score, std


# ================================Distance==============================================
def get_outputs(model, loader, noise_scheduler, device):

    model.eval()
    outputs = []
    with torch.no_grad():
        for clean_images, labels in loader:
            clean_images, labels = clean_images.to(device), labels.to(device)

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
            noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]

            outputs.append(noise_pred.detach())

    return torch.cat(outputs, dim=0).flatten().cpu().numpy()


# TODO: Check Wassertein Distance
def calculate_wasserstein_distance(scrubbed_model, gold_model, unlearn_loader, noise_scheduler, device):
    scrub_outputs = get_outputs(
        scrubbed_model, unlearn_loader, noise_scheduler, device)
    gold_outputs = get_outputs(
        gold_model, unlearn_loader, noise_scheduler, device)
    wa_d = wasserstein_distance(scrub_outputs, gold_outputs)
    return wa_d


# TODO: Check Activation Distance
def ActivationDistance(layer1, layer2):
    distance = 0.
    for i in range(layer1.size(0)):
        layer1_norm = layer1[i].flatten() / torch.linalg.norm(layer1[i].flatten())
        layer2_norm = layer2[i].flatten() / torch.linalg.norm(layer2[i].flatten())
        dist = torch.linalg.norm(layer1_norm - layer2_norm)
        distance += dist
    return (distance / layer1.size(0))


# TODO: Check Weight Distance
def WeightDistance(model,model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        space='  ' if 'bias' in k else ''
        current_dist=(p-p0).pow(2).sum().item()
        current_norm=p.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    # print(f'Distance: {np.sqrt(distance)}')
    # print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)

