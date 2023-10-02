import numpy as np
import torch
import torch.nn.functional as F
from sklearn import linear_model, model_selection
import matplotlib.pyplot as plt


def compute_losses(model, loader, noise_scheduler, device, MIA_t=-1):
    """Auxiliary function to compute per-sample losses"""

    all_losses = []
    for clean_images, labels in loader:
        clean_images, labels = clean_images.to(device), labels.to(device)

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
        ).long()
        if MIA_t != -1:
            timesteps = torch.ones_like(timesteps) * MIA_t
        # print(f'bs: {bs}, timesteps: {timesteps.size()}')
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise residual
        noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]
        # compute the loss for each image
        losses = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2, 3)).numpy(force=True)
        # print(bs, losses.shape)

        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


# should have good accuracy on the test set, then compare mia scores
def compute_mia(model, forget_loader, test_loader, noise_scheduler, device, unl_method, save_path, MIA_t=-1):
    forget_losses = compute_losses(model, forget_loader, noise_scheduler, device, MIA_t)
    test_losses = compute_losses(model, test_loader, noise_scheduler, device, MIA_t)

    # make sure we have a balanced dataset for the MIA
    if len(forget_losses) > len(test_losses):
        np.random.shuffle(forget_losses)
        forget_losses = forget_losses[: len(test_losses)]
    else:
        np.random.shuffle(test_losses)
        test_losses = test_losses[: len(forget_losses)]

    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)

    print(
        f"The MIA has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images"
    )

    import csv
    loss_name = ['test_losses', 'forget_losses']
    rows = zip(test_losses, forget_losses)
    with open(save_path + '/mia.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(loss_name)
        for row in rows:
            writer.writerow(row)

    # draw the histogram of losses
    fig = plt.figure(figsize=(8, 4))
    plt.hist(test_losses, density=True, bins=50, alpha=0.5, label="Test set")
    plt.hist(forget_losses, density=True, bins=50, alpha=0.5, label="Forget set")
    # plt.hist(test_losses, density=True, bins=50, alpha=0.5, label="Test set (remain cls)")
    # plt.hist(forget_losses, density=True, bins=50, alpha=0.5, label="Train set (remain cls)")

    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=14)
    plt.title(f"{unl_method} has attack accuracy {mia_scores.mean():.4f} with T={MIA_t+1}.")
    # plt.title(f"Unscrubbed model has MIA accuracy {mia_scores.mean():.4f} on UTKFace.")

    # set spine top and right visibility to false
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 10))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # save the figure
    plt.tight_layout()
    plt.savefig(f"{save_path}/mia_{unl_method}_1_T{MIA_t+1}.png", dpi=300, bbox_inches="tight")

    return mia_scores.mean()


