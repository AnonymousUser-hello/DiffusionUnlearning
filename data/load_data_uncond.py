import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .loadCelebAHQ_attr import CelebAHQDataset


def getUnlDevNum(unl_dev):
    if unl_dev == '':
        return []
    unl_dev_list = []
    for dev in unl_dev.split('+'):
        unl_dev_list.append(int(dev))
    return unl_dev_list


def load_data(args):
    assert args.unl_ratio < 1., 'unl_ratio should be less than 1.'

    if args.dataset == 'CelebA':
        attr_cls = [
            '5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', \
            'Bald', 'Bangs','Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', \
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', \
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', \
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', \
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', \
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', \
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
            ]
        unl_attrs = getUnlDevNum(args.unl_cls)
        for j in unl_attrs:
            print(j, attr_cls[j])
        # rm_attrs = list(set(range(0, int(len(attr_cls)))) - set(unl_attrs))
        dataset_unl = CelebADataset(root=args.data_path + '/CelebA', train=True,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs)
        dataset_rem = CelebADataset(root=args.data_path + '/CelebA', train=True,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs, unl=False)
        testset_unl = CelebADataset(root=args.data_path + '/CelebA', train=False,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs)
        testset_rem = CelebADataset(root=args.data_path + '/CelebA', train=False,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs, unl=False)
        # unlearned data
        train_length = int(len(dataset_rem) * 0.1)
        trainset_rem, _ = torch.utils.data.random_split(
            dataset_rem, [train_length, len(dataset_rem) - train_length])
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
        trainset_unl, _ = torch.utils.data.random_split(
            dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data and test data
        trainset_all = trainset_rem + trainset_unl
        testset_all = testset_rem + testset_unl

    elif args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        unl_clses = getUnlDevNum(args.unl_cls)
        # train data
        trainset_all = CIFAR10(root=args.data_path + '/CIFAR10', train=True, download=True, transform=transform)
        targets = trainset_all.targets
        idxx = [i for i, x in enumerate(targets) if x in unl_clses]
        trainset_unl = Subset(trainset_all, idxx)
        trainset_rem = Subset(trainset_all, [i for i in range(len(trainset_all)) if i not in idxx])
        # test data
        testset_all = CIFAR10(root=args.data_path + '/CIFAR10', train=False, download=True, transform=test_transform)
        test_idxx = [i for i, x in enumerate(testset_all.targets) if x in unl_clses]
        testset_unl = Subset(testset_all, test_idxx)
        testset_rem = Subset(testset_all, [i for i in range(len(testset_all)) if i not in test_idxx])

    elif args.dataset == 'CelebA_hq':
        attr_cls = [
            '5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', \
            'Bald', 'Bangs','Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', \
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', \
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', \
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', \
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', \
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', \
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
            ]
        unl_attrs = getUnlDevNum(args.unl_cls)
        for j in unl_attrs:
            print(j, attr_cls[j])
        # rm_attrs = list(set(range(0, int(len(attr_cls)))) - set(unl_attrs))
        dataset_unl = CelebAHQDataset(root=args.data_path + '/CelebAHQ/CelebAMask-HQ', train=True,
                          transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs)
        dataset_rem = CelebAHQDataset(root=args.data_path + '/CelebAHQ/CelebAMask-HQ', train=True,
                          transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs, unl=False)
        testset_unl = CelebAHQDataset(root=args.data_path + '/CelebAHQ/CelebAMask-HQ', train=False,
                          transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs)
        testset_rem = CelebAHQDataset(root=args.data_path + '/CelebAHQ/CelebAMask-HQ', train=False,
                          transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          attr_indx=unl_attrs, unl=False)
        # unlearned data
        train_length = int(len(dataset_rem) * 1.0)
        trainset_rem, _ = torch.utils.data.random_split(
            dataset_rem, [train_length, len(dataset_rem) - train_length])
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
        trainset_unl, _ = torch.utils.data.random_split(
            dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data and test data
        trainset_all = trainset_rem + trainset_unl
        testset_all = testset_rem + testset_unl

    else:
        assert False, 'not support the dataset yet.'

    print(f'trainset_all: {len(trainset_all)}, trainset_rem: {len(trainset_rem)}, trainset_unl: {len(trainset_unl)}')
    trainloader_all = DataLoader(trainset_all, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    trainloader_rem = DataLoader(trainset_rem, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    trainloader_unl = DataLoader(trainset_unl, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    num_examples = {"trainset_all": len(trainset_all), "trainset_rem": len(trainset_rem), "trainset_unl": len(trainset_unl), \
                    "testset_all": len(testset_all), "testset_rem": len(testset_rem), "testset_unl": len(testset_unl)}
    return trainloader_all, trainloader_rem, trainloader_unl, num_examples, trainset_rem, testset_rem, trainset_unl, testset_unl

