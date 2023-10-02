import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# def getUnlDevNum(unl_dev):
#     if unl_dev == '':
#         return []
#     unl_dev_list = []
#     for dev in unl_dev.split('+'):
#         unl_dev_list.append(int(dev))
#     return unl_dev_list


# def load_data(args):
#     attr = args.attr.split('_')
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     num_classes = []
#     for a in attr:
#         if a == "age":
#             num_classes.append(117)
#         elif a == "gender":
#             num_classes.append(2)
#         elif a == "race":
#             # num_classes.append(4)
#             num_classes.append(5)
#         else:
#             raise ValueError("Target type \"{}\" is not recognized.".format(a))

#     # dataset_all = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform,
#     #                              myfilter=[0, 1, 2, 3], myfilter_attr=2)  # white, black, asian, indian
#     unl_clses = getUnlDevNum(args.unl_cls)
#     rem_clses = list(set(range(0, num_classes[args.y_t])) - set(unl_clses))
#     # print("rem_clses: ", rem_clses, "unl_clses: ", unl_clses)
#     dataset_rem = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform,
#                                  myfilter=rem_clses, myfilter_attr=2) # 14604
#     dataset_unl = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform,
#                                  myfilter=unl_clses, myfilter_attr=2) # 7409

#     # remaning train data and test data
#     train_length = int(len(dataset_rem)*0.8)
#     test_length = len(dataset_rem) - int(len(dataset_rem)*0.8)
#     trainset_rem, testset_rem= torch.utils.data.random_split(dataset_rem, [train_length, test_length])
#     # unlearned data
#     trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
#     trainset_unl, _ = torch.utils.data.random_split(dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
#     # all train data
#     trainset_all = trainset_rem + trainset_unl
#     # print(len(trainset_all), len(trainset_rem), len(trainset_unl), len(testset_rem))

#     trainloader_all = DataLoader(trainset_all, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
#     trainloader_rem = DataLoader(trainset_rem, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
#     trainloader_unl = DataLoader(trainset_unl, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
#     testloader = DataLoader(testset_rem, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     num_examples = {"trainset_all": len(trainset_all), "trainset_rem": len(trainset_rem), "trainset_unl": len(trainset_unl), "testset": len(testset_rem)}
#     return num_classes, trainloader_all, trainloader_rem, trainloader_unl, num_examples, testloader



class UTKFaceDataset(Dataset):
    def __init__(self, root, attr, transform=None, myfilter=None, myfilter_attr=2):
        super(UTKFaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.myfilter = myfilter
        self.myfilter_attr = myfilter_attr
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]
        self.files = os.listdir(root)

        self.imgs = []
        for file in self.files:
            img_name = file.split('jpg')[0]
            attrs = img_name.split('_')
            if len(attrs) < 4 or int(attrs[2]) >= 4:
                continue

            if self.myfilter is None or int(attrs[self.myfilter_attr]) in self.myfilter:
                self.imgs.append(img_name + 'jpg')
            else:
                continue

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        attrs = self.imgs[index].split('_')
        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        img_path = os.path.join(self.root, self.imgs[index] + '.chip.jpg').rstrip()
        img = Image.open(img_path).convert('RGB')

        target = []
        # for t in self.attr:
        #     if t == 'age':
        #         target.append(age)
        #     elif t == 'gender':
        #         target.append(gender)
        #     elif t == 'race':
        #         target.append(race)
        #     else:
        #         raise ValueError("Target type \"{}\" is not recognized.".format(t))
        target.append(race)

        if self.transform:
            img = self.transform(img)
        if target:
            target = tuple(target) if len(target) > 1 else target[0]
        else:
            target = None

        return img, target



# def getUnlDevNum(unl_dev):
#     if unl_dev == '':
#         return []
#     unl_dev_list = []
#     for dev in unl_dev.split('+'):
#         unl_dev_list.append(int(dev))
#     return unl_dev_list

# ratio = 0.8
# num_classes = 4
# attr = 'age_gender_race'.split('_')
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
# # dataset_all = UTKFaceDataset(root=args.data_path + '/UTKFace/', attr=attr, transform=transform,
# #                              myfilter=[0, 1, 2, 3], myfilter_attr=2)  # white, black, asian, indian
# unl_clses = getUnlDevNum('3')
# rem_clses = list(set(range(0, num_classes)) - set(unl_clses))
# dataset_rem = UTKFaceDataset(root='/data/datasets/UTKFace/', attr=attr, transform=transform,
#                             myfilter=rem_clses, myfilter_attr=2) # 14604
# dataset_unl = UTKFaceDataset(root='/data/datasets/UTKFace/', attr=attr, transform=transform,
#                             myfilter=unl_clses, myfilter_attr=2) # 7409
# # remaning train data and test data
# train_length = int(len(dataset_rem)*ratio)
# test_length = len(dataset_rem) - int(len(dataset_rem)*ratio)
# trainset_rem, testset_rem= torch.utils.data.random_split(dataset_rem, [train_length, test_length])
# # unlearned data
# trainunl_length = int((0.2 * train_length) / (1 - 0.2))
# trainset_unl, testset_unl = torch.utils.data.random_split(
#     dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
# # all train data and test data
# trainset_all = trainset_rem + trainset_unl
# testset_all = testset_rem + testset_unl


# trainloader_all = DataLoader(trainset_all, batch_size=16, shuffle=True, drop_last=False, num_workers=4)
# trainloader_rem = DataLoader(trainset_rem, batch_size=16, shuffle=True, drop_last=False, num_workers=4)
# trainloader_unl = DataLoader(trainset_unl, batch_size=16, shuffle=True, drop_last=False, num_workers=4)

# dataloaders = [trainloader_all, trainloader_rem, trainloader_unl]
# for i in range(3):
#     print("len(dataloaders[{}]): ".format(i), len(dataloaders[i]))
#     for j, (inputs, labels) in enumerate(dataloaders[i]):
#         # print("inputs.shape: ", inputs.shape)
#         print("labels: ", labels)
#         if j == 0:
#             break
#     print("")
