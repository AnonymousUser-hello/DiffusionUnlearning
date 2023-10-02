import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class CelebADataset(Dataset):

    def __init__(self, root, train=True, transform=None, attr_indx=None, unl=True):

        self.root = root
        self.transform = transform
        self.attr_indx = attr_indx
        self.attr_cls = [
            '5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', \
            'Bald', 'Bangs','Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', \
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', \
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', \
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', \
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', \
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', \
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
            ]
        flag = 1 if unl else -1

        if train:
            ann_path = self.root + '/train.txt'
        else:
            ann_path = self.root + '/test.txt'

        images = []
        targets = []
        identities = []
        for line in open(ann_path, 'r'):
            sample = line.split()
            if len(sample) != 42:
                raise(RuntimeError('Annotated face attributes of CelebA dataset should not be different from 40'))
            if self.attr_indx is None:
                images.append(sample[0])
                identities.append(int(sample[1]))
                targets.append([int(i) for i in sample[2:]])
            else:
                for j in self.attr_indx:
                    if int(sample[j+2]) == flag: # add the samples with i attribute (positive for unl)
                        images.append(sample[0])
                        identities.append(int(sample[1]))
                        targets.append([int(i) for i in sample[2:]])
                    else:
                        continue

        self.images = [os.path.join(self.root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        self.identities = identities


    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        # Load data and get label
        img = Image.open(self.images[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # target = torch.tensor(self.targets[index][9])
        # if target == -1:
        #     target += 1
        # print(f'target: {target}')
        # return img, target

        return img



# from torchvision.utils import save_image
# from torch.utils.data import DataLoader
# from torchvision import transforms
# def getUnlDevNum(unl_dev):
#     if unl_dev == '':
#         return []
#     unl_dev_list = []
#     for dev in unl_dev.split('+'):
#         unl_dev_list.append(int(dev))
#     return unl_dev_list
# attr_cls = [
#     '5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', \
#     'Bald', 'Bangs','Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', \
#     'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', \
#     'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', \
#     'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', \
#     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', \
#     'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', \
#     'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
#     ]
# unl_attrs = getUnlDevNum('9')
# # rm_attrs = list(set(range(0, int(len(attr_cls)))) - set(unl_attrs))
# dataset_unl = CelebADataset(root='/data/datasets/CelebA', train=True,
#                     transform=transforms.Compose([
#                     transforms.Resize((64, 64)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                     ]),
#                     attr_indx=unl_attrs)
# dataset_rem = CelebADataset(root='/data/datasets/CelebA', train=True,
#                     transform=transforms.Compose([
#                     transforms.Resize((64, 64)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                     ]),
#                     attr_indx=unl_attrs, unl=False)
# testset_unl = CelebADataset(root='/data/datasets/CelebA', train=False,
#                     transform=transforms.Compose([
#                     transforms.Resize((64, 64)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                     ]),
#                     attr_indx=unl_attrs)
# testset_rem = CelebADataset(root='/data/datasets/CelebA', train=False,
#                     transform=transforms.Compose([
#                     transforms.Resize((64, 64)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                     ]),
#                     attr_indx=unl_attrs, unl=False)
# # unlearned data
# train_length = int(len(dataset_rem) * 0.1)
# trainset_rem, _ = torch.utils.data.random_split(
#     dataset_rem, [train_length, len(dataset_rem) - train_length])
# trainunl_length = int((0.2 * train_length) / (1 - 0.2))
# trainset_unl, _ = torch.utils.data.random_split(
#     dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
# # all train data and test data
# trainset_all = trainset_rem + trainset_unl
# testset_all = testset_rem + testset_unl


# for j in unl_attrs:
#     print(j, attr_cls[j])
# print(f'trainset_all: {len(trainset_all)}, trainset_rem: {len(trainset_rem)}, trainset_unl: {len(trainset_unl)}')
# trainloader_all = DataLoader(trainset_all, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
# trainloader_rem = DataLoader(trainset_rem, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
# trainloader_unl = DataLoader(trainset_unl, batch_size=64, shuffle=True, drop_last=False, num_workers=4)

# # print(unl_attrs, rm_attrs)
# dataloaders = [trainloader_all, trainloader_rem, trainloader_unl]
# for i in range(len(dataloaders)):
#     print("len(dataloaders[{}]): ".format(i), len(dataloaders[i]))
#     for j, (inputs, labels) in enumerate(dataloaders[i]):
#         print("labels: ", labels)
#         # image = (inputs / 2 + 0.5).clamp(0, 1)
#         # save_image(image, f'{str(i)}_{str(j)}.png')
#         if j == 1:
#             break
#     print("")

