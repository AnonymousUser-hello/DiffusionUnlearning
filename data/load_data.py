import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from .loadUTKFace import UTKFaceDataset
from .loadCelebA import CelebADataset
from .loadVGGFace2 import VGG_Faces2HQ
from .loadLSUN import LSUN


def getUnlDevNum(unl_dev):
    if unl_dev == '':
        return []
    unl_dev_list = []
    for dev in unl_dev.split('+'):
        unl_dev_list.append(int(dev))
    return unl_dev_list


def load_data(args):
    assert args.unl_ratio < 1., 'unl_ratio should be less than 1.'

    if args.dataset == 'CIFAR10':
        num_classes = 10
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
        rem_clses = list(set(range(0, num_classes)) - set(unl_clses))
        # train data
        dataset_all = CIFAR10(root=args.data_path + '/CIFAR10', train=True, download=True, transform=transform)
        targets = dataset_all.targets
        idxx = [i for i, x in enumerate(targets) if x in unl_clses]
        dataset_unl = Subset(dataset_all, idxx)
        trainset_rem = Subset(dataset_all, [i for i in range(len(dataset_all)) if i not in idxx])
        # test data
        testset_all = CIFAR10(root=args.data_path + '/CIFAR10', train=False, download=True, transform=test_transform)
        test_idxx = [i for i, x in enumerate(testset_all.targets) if x in unl_clses]
        testset_unl = Subset(testset_all, test_idxx)
        testset_rem = Subset(testset_all, [i for i in range(len(testset_all)) if i not in test_idxx])
        # unlearned data
        train_length = int(len(trainset_rem))
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
        trainset_unl, _ = torch.utils.data.random_split(dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data
        trainset_all = trainset_rem + trainset_unl
        # print(len(trainset_all), len(trainset_rem), len(trainset_unl), len(testset_rem))

    elif args.dataset == 'UTKFace':
        ratio = 0.8
        num_classes = 4
        attr = 'age_gender_race'.split('_')
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # dataset_all = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform,
        #                              myfilter=[0, 1, 2, 3], myfilter_attr=2)  # white, black, asian, indian
        unl_clses = getUnlDevNum(args.unl_cls)
        rem_clses = list(set(range(0, num_classes)) - set(unl_clses))
        dataset_rem = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform,
                                    myfilter=rem_clses, myfilter_attr=2) # 14604
        dataset_unl = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform,
                                    myfilter=unl_clses, myfilter_attr=2) # 7409
        # remaning train data and test data
        train_length = int(len(dataset_rem)*ratio)
        test_length = len(dataset_rem) - int(len(dataset_rem)*ratio)
        trainset_rem, testset_rem= torch.utils.data.random_split(dataset_rem, [train_length, test_length])
        # unlearned data
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
        trainset_unl, testset_unl = torch.utils.data.random_split(
            dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data and test data
        trainset_all = trainset_rem + trainset_unl
        testset_all = testset_rem + testset_unl

    elif args.dataset == 'VGGFace2':
        ratio = 0.9
        # num_classes = 8631
        num_classes = 10
        unl_identities = getUnlDevNum(args.unl_cls)
        # rm_identities = list(set(range(1, 8631)) - set(unl_identities))
        rm_identities = list(set(range(0, 10)) - set(unl_identities))
        dataset_unl = VGG_Faces2HQ(root=args.data_path + '/VGGFace2',
                                            transform=transforms.Compose([
                                                transforms.Resize((128, 128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]),
                                            identity=unl_identities)
        dataset_rem = VGG_Faces2HQ(root=args.data_path + '/VGGFace2',
                                    transform=transforms.Compose([
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]),
                                    identity=rm_identities)
        # remaning train data and test data
        train_length = int(len(dataset_rem)*ratio)
        test_length = len(dataset_rem) - int(len(dataset_rem)*ratio)
        trainset_rem, testset_rem= torch.utils.data.random_split(dataset_rem, [train_length, test_length])
        # unlearned data
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio)) if args.unl_ratio < 1. else train_length
        trainset_unl, testset_unl = torch.utils.data.random_split(
            dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data and test data
        trainset_all = trainset_rem + trainset_unl
        testset_all = testset_rem + testset_unl

    elif args.dataset == 'MNIST':
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        unl_clses = getUnlDevNum(args.unl_cls)
        rem_clses = list(set(range(0, num_classes)) - set(unl_clses))
        # train data
        dataset_all = MNIST(root=args.data_path, train=True, download=True, transform=transform)
        targets = dataset_all.targets
        idxx = [i for i, x in enumerate(targets) if x in unl_clses]
        dataset_unl = Subset(dataset_all, idxx)
        trainset_rem = Subset(dataset_all, [i for i in range(len(dataset_all)) if i not in idxx])
        # test data
        testset_all = MNIST(root=args.data_path, train=False, download=True, transform=transform)
        test_idxx = [i for i, x in enumerate(testset_all.targets) if x in unl_clses]
        testset_unl = Subset(testset_all, test_idxx)
        testset_rem = Subset(testset_all, [i for i in range(len(testset_all)) if i not in test_idxx])
        # unlearned data
        train_length = int(len(trainset_rem))
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
        trainset_unl, _ = torch.utils.data.random_split(dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data
        trainset_all = trainset_rem + trainset_unl

    elif args.dataset == 'CelebA':
        num_classes = 20
        unl_identities = getUnlDevNum(args.unl_cls)
        rm_identities = list(set(range(0, 20)) - set(unl_identities))
        dataset_unl = CelebADataset(root=args.data_path + '/CelebA', train=True,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          identity=unl_identities)
        trainset_rem = CelebADataset(root=args.data_path + '/CelebA', train=True,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          identity=rm_identities)
        testset_unl = CelebADataset(root=args.data_path + '/CelebA', train=False,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          identity=unl_identities)
        testset_rem = CelebADataset(root=args.data_path + '/CelebA', train=False,
                          transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]),
                          identity=rm_identities)
        # unlearned data
        train_length = int(len(trainset_rem)) # x / (n+x) = r, x = n*r/(1-r)
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
        trainset_unl, _ = torch.utils.data.random_split(
            dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data and test data
        trainset_all = trainset_rem + trainset_unl
        testset_all = testset_rem + testset_unl

    elif args.dataset == 'LSUN':
        ratio = 0.01
        num_classes = 10
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # classes = ['bedroom_train', 'bridge_train', 'church_outdoor_train', 'classroom_train', 'conference_room_train', 'dining_room_train', 'kitchen_train', 'living_room_train', 'restaurant_train', 'tower_train']
        rem_cls = ['bridge_train', 'classroom_train', 'conference_room_train', 'dining_room_train', 'kitchen_train', 'living_room_train', 'restaurant_train', 'tower_train']
        unl_cls = ['bedroom_train', 'church_outdoor_train']
        dataset_rem = LSUN(root=args.data_path + '/LSUN/train', classes=rem_cls, transform=transform)
        dataset_unl = LSUN(root=args.data_path + '/LSUN/train', classes=unl_cls, transform=transform)
        # remaning train data and test data
        train_length = int(len(dataset_rem)*ratio)
        test_length = len(dataset_rem) - int(len(dataset_rem)*ratio)
        trainset_rem, testset_rem= torch.utils.data.random_split(dataset_rem, [train_length, test_length])
        # unlearned data
        trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio)) if args.unl_ratio < 1. else train_length
        trainset_unl, testset_unl = torch.utils.data.random_split(
            dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
        # all train data and test data
        trainset_all = trainset_rem + trainset_unl
        testset_all = testset_rem + testset_unl

    else:
        assert False, 'not support the dataset yet.'

    trainloader_all = DataLoader(trainset_all, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    trainloader_rem = DataLoader(trainset_rem, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    trainloader_unl = DataLoader(trainset_unl, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    # testloader_all = DataLoader(testset_all, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    # testloader_rem = DataLoader(testset_rem, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    # testloader_unl = DataLoader(testset_unl, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    num_examples = {"trainset_all": len(trainset_all), "trainset_rem": len(trainset_rem), "trainset_unl": len(trainset_unl), \
                    "testset_all": len(testset_all), "testset_rem": len(testset_rem), "testset_unl": len(testset_unl)}
    # return num_classes, trainloader_all, trainloader_rem, trainloader_unl, num_examples, testloader_all, testloader_rem, testloader_unl
    return num_classes, trainloader_all, trainloader_rem, trainloader_unl, num_examples, trainset_rem, testset_rem, trainset_unl, testset_unl



# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
# num_classes = 10
# unl_clses = getUnlDevNum('2+8')
# rem_clses = list(set(range(0, num_classes)) - set(unl_clses))
# print(f'unl_clses: {unl_clses}, rem_clses: {rem_clses}')
# dataset_all = CIFAR10(root='/datasets/CIFAR10', train=True, download=False, transform=transform)
# # extract dataset with labels in unl_clses
# targets = dataset_all.targets
# idxx = [i for i, x in enumerate(targets) if x in unl_clses]
# dataset_unl = Subset(dataset_all, idxx)
# dataset_rem = Subset(dataset_all, [i for i in range(len(dataset_all)) if i not in idxx])
# print(len(dataset_all), len(dataset_rem), len(dataset_unl))

# ratio = 0.8
# train_length = int(len(dataset_rem)*ratio)
# test_length = len(dataset_rem) - int(len(dataset_rem)*ratio)
# trainset_rem, testset_rem= torch.utils.data.random_split(dataset_rem, [train_length, test_length])
# # unlearned data
# trainunl_length = int((0.1 * train_length) / (1 - 0.1)) if 0.1 < 1. else train_length
# trainset_unl, _ = torch.utils.data.random_split(dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
# # all train data. combine rem and unl
# trainset_all = trainset_rem + trainset_unl
# print(len(trainset_all), len(trainset_rem), len(trainset_unl), len(testset_rem))
# trainloader_all = DataLoader(trainset_all, batch_size=16, shuffle=True, drop_last=False, num_workers=2)
# trainloader_rem = DataLoader(trainset_rem, batch_size=16, shuffle=True, drop_last=False, num_workers=2)
# trainloader_unl = DataLoader(trainset_unl, batch_size=16, shuffle=True, drop_last=False, num_workers=2)

# dataloaders = [trainloader_all, trainloader_rem, trainloader_unl]
# for dataloader in dataloaders:
#     for i, (inputs, targets) in enumerate(dataloader):
#         print(inputs.shape, targets)
#         break

