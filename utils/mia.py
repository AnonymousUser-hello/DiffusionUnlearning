import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


def get_attack_dataset(trainset, testset, batch_size, split=True, div=2):

    if split:
        total_len = min(len(trainset), len(testset))
        nonmem_len = total_len // div
        mem_train, mem_test, _ = random_split(
            trainset, [nonmem_len, nonmem_len, len(trainset) - nonmem_len * 2])
        nonmem_train, nonmem_test, _ = random_split(
            testset, [nonmem_len, nonmem_len, len(testset) - nonmem_len * 2])
        mem_train, mem_test = list(mem_train), list(mem_test)
        nonmem_train, nonmem_test = list(nonmem_train), list(nonmem_test)

        for i in range(len(mem_train)):
            mem_train[i] = mem_train[i] + (1,)
        for i in range(len(nonmem_train)):
            nonmem_train[i] = nonmem_train[i] + (0,)
        for i in range(len(nonmem_test)):
            nonmem_test[i] = nonmem_test[i] + (0,)
        for i in range(len(mem_test)):
            mem_test[i] = mem_test[i] + (1,)

        attack_train = mem_train + nonmem_train
        attack_test = mem_test + nonmem_test

        attack_trainloader = DataLoader(attack_train, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
        attack_testloader = DataLoader(attack_test, batch_size=batch_size,
                                       shuffle=True, num_workers=2)
        print(f'attack_train: {len(attack_train)}, attack_test: {len(attack_test)}')

        return attack_trainloader, attack_testloader  # for train the attack model

    else:
        # if len(trainset) > len(testset):
        #     trainset, _ = random_split(trainset, [len(testset), len(trainset) - len(testset)])
        # else:
        #     testset, _ = random_split(testset, [len(trainset), len(testset) - len(trainset)])
        mem_set = list(trainset)
        nonmem_set = list(testset)
        for i in range(len(mem_set)):
            mem_set[i] = mem_set[i] + (1,)
        for i in range(len(nonmem_set)):
            nonmem_set[i] = nonmem_set[i] + (0,)
        member_loader = DataLoader(mem_set, batch_size=batch_size,
                                   shuffle=True, num_workers=2)
        nonmember_loader = DataLoader(nonmem_set, batch_size=batch_size,
                                      shuffle=False, num_workers=2)
        print(f'member: {len(mem_set)}, nonmember: {len(nonmem_set)}')

        return member_loader, nonmember_loader


class MIAClassifier(nn.Module):
    def __init__(self):
        super(MIAClassifier, self).__init__()

        self.Loss_Component = nn.Sequential(
			nn.Linear(1, 1024),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(1024, 512),
			nn.ReLU(),
            nn.Dropout(p=0.2),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, 2),
		)

    def forward(self, loss):
        final_result = self.Loss_Component(loss)
        return final_result


def train_mia(args, attack_model, model, trainset_rem, testset_rem, noise_scheduler, device):

    optimizer = torch.optim.Adam(attack_model.parameters(), lr=args.MIA_lr)
    criterion = nn.CrossEntropyLoss()
    attack_trainloader, attack_testloader = get_attack_dataset(trainset_rem, testset_rem, args.eval_batch_size, split=True, div=2)

    model.eval()
    for e in range(args.MIA_epochs):

        attack_model.train()
        for i, (clean_images, labels, members) in enumerate(attack_trainloader):
            clean_images, labels, members = clean_images.to(device), labels.to(device), members.to(device)
            losses = get_loss(clean_images, labels, model, noise_scheduler, args.MIA_t)

            # fed losses to MIA classifier
            outputs = attack_model(losses.unsqueeze(1))
            loss = criterion(outputs, members)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if e % 1 == 0:
            attack_model.eval()
            with torch.no_grad():
                total = 0.
                correct = 0.
                for j, (imgs, labs, membs) in enumerate(attack_testloader):
                    imgs, labs, membs = imgs.to(device), labs.to(device), membs.to(device)
                    losses = get_loss(imgs, labs, model, noise_scheduler, args.MIA_t)
                    outputs = attack_model(losses.unsqueeze(1))
                    testloss = criterion(outputs, membs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += membs.size(0)
                    correct += (predicted == membs).sum().item()
                print(f'MIA_Epoch {e} | Accuracy: {float(correct/total):.4f} | Loss: {testloss:.4f}')


def test_mia(args, attack_model, model, trainset_unl, testset_unl, noise_scheduler, device, save_path, unl_method):
    attack_model.eval()
    model.eval()
    member_loader, nonmember_loader = get_attack_dataset(trainset_unl, testset_unl, args.eval_batch_size, split=False, div=2)
    with torch.no_grad():
        total = 0.
        correct = 0.
        mem_losses = []
        for i, (imgs, labs, membs) in enumerate(member_loader):
            imgs, labs, membs = imgs.to(device), labs.to(device), membs.to(device)
            losses = get_loss(imgs, labs, model, noise_scheduler, args.MIA_t)
            outputs = attack_model(losses.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            total += membs.size(0)
            correct += (predicted == membs).sum().item()
            mem_losses.append(losses)
            torch.cuda.empty_cache()
            gc.collect()
            mia_trainset_unl = float(correct/total)
        print(f'{unl_method} has accuracy: {mia_trainset_unl:.4f} on Trainset_unl')

        total = 0.
        correct = 0.
        nonmem_losses = []
        for i, (imgs, labs, membs) in enumerate(nonmember_loader):
            imgs, labs, membs = imgs.to(device), labs.to(device), membs.to(device)
            losses = get_loss(imgs, labs, model, noise_scheduler, args.MIA_t)
            outputs = attack_model(losses.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            total += membs.size(0)
            correct += (predicted == membs).sum().item()
            nonmem_losses.append(losses)
            torch.cuda.empty_cache()
            gc.collect()
            mia_testset_unl = float(correct/total)
        print(f'{unl_method} has accuracy: {mia_testset_unl:.4f} on Testset_unl')

    mem_losses = torch.cat(mem_losses, dim=0).cpu().numpy()
    nonmem_losses = torch.cat(nonmem_losses, dim=0).cpu().numpy()
    if len(mem_losses) > len(nonmem_losses):
        np.random.shuffle(mem_losses)
        mem_losses = mem_losses[: len(nonmem_losses)]
    else:
        np.random.shuffle(nonmem_losses)
        nonmem_losses = nonmem_losses[: len(mem_losses)]

    # draw the histogram of losses
    fig = plt.figure(figsize=(8, 4))
    plt.hist(nonmem_losses, density=True, bins=50, alpha=0.5, label="Test set")
    plt.hist(mem_losses, density=True, bins=50, alpha=0.5, label="Forget set")

    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=14)
    plt.title(f"{unl_method} has attack accuracy {mia_trainset_unl:.4f} on Trainset_unl with T={args.MIA_t+1}.")

    # set spine top and right visibility to false
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # save the figure
    plt.tight_layout()
    plt.savefig(f"{save_path}/mia_{unl_method}_T{args.MIA_t+1}.png", dpi=300, bbox_inches="tight")


def get_loss(clean_images, labels, model, noise_scheduler, MIA_t=-1):
    with torch.no_grad():
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
        ).long()
        if MIA_t != -1:
            timesteps = torch.ones_like(timesteps) * MIA_t
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise residual
        noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]
        # compute the loss for each image
        losses = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2, 3)) # [bs]
        # print(bs, losses.shape)

        del clean_images, labels, noise, noisy_images, noise_pred
        torch.cuda.empty_cache()
        gc.collect()

    return losses

