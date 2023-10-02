import os
from PIL import Image
from torch.utils import data


class VGG_Faces2HQ(data.Dataset):

    def __init__(self, root, transform=None, identity=None):

        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = os.path.join(root, 'VGGface2_None_norm_512_true_bygfpgan')
        self.files = os.listdir(self.root) # 8631 files
        self._transform = transform
        self.identity = identity

        self.imgs = []
        self.targets = []
        n = 0
        for file in self.files:
            # class_idx = file
            label = n # \eg., turn n000001 to 0
            # print(file, label)
            n += 1
            if self.identity is None or int(label) in self.identity:
                img_file = os.path.join(self.root, file)
                img_list = os.listdir(img_file)
                for img in img_list: # \eg., 0317_01.jpg
                    self.imgs.append(os.path.join(img_file, img) )
                    self.targets.append(int(label))
            else:
                continue

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img = Image.open(self.imgs[index]).convert('RGB')
        target = self.targets[index]
        if self._transform is not None:
            img = self._transform(img)

        return img, target


# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# def getUnlDevNum(unl_dev):
#     if unl_dev == '':
#         return []
#     unl_dev_list = []
#     for dev in unl_dev.split('+'):
#         unl_dev_list.append(int(dev))
#     return unl_dev_list

# ratio = 0.9
# # num_classes = 8631
# num_classes = 10
# unl_identities = getUnlDevNum('2+8')
# # rm_identities = list(set(range(1, 8631)) - set(unl_identities))
# rm_identities = list(set(range(0, 10)) - set(unl_identities))
# dataset_unl = VGG_Faces2HQ(root='/data/datasets/VGGFace2',
#                                     transform=transforms.Compose([
#                                         transforms.Resize((128, 128)),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                     ]),
#                                     identity=unl_identities)
# dataset_rem = VGG_Faces2HQ(root='/data/datasets/VGGFace2',
#                             transform=transforms.Compose([
#                                 transforms.Resize((128, 128)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                 ]),
#                             identity=rm_identities)
# # remaning train data and test data
# train_length = int(len(dataset_rem)*ratio)
# test_length = len(dataset_rem) - int(len(dataset_rem)*ratio)
# trainset_rem, testset_rem= torch.utils.data.random_split(dataset_rem, [train_length, test_length])
# # unlearned data
# trainunl_length = int((0.2 * train_length) / (1 - 0.2)) if 0.2 < 1. else train_length
# trainset_unl, testset_unl = torch.utils.data.random_split(
#     dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
# # all train data and test data
# trainset_all = trainset_rem + trainset_unl
# testset_all = testset_rem + testset_unl

# trainloader_all = DataLoader(trainset_all, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
# trainloader_rem = DataLoader(trainset_rem, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
# trainloader_unl = DataLoader(trainset_unl, batch_size=64, shuffle=True, drop_last=False, num_workers=4)

# dataloaders = [trainloader_all, trainloader_rem, trainloader_unl]
# for i in range(3):
#     print("len(dataloaders[{}]): ".format(i), len(dataloaders[i]))
#     for j, (inputs, labels) in enumerate(dataloaders[i]):
#         # print("inputs.shape: ", inputs.shape)
#         print("labels: ", labels)
#         if j == 0:
#             break
#     print("")

