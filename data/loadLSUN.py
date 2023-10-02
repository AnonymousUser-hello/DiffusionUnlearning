import io
import os.path
import pickle
import string
from collections.abc import Iterable
from typing import Any, Callable, cast, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class LSUNClass(VisionDataset):
    def __init__(
        self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, mydict: Optional[dict] = None, cls: Optional[str] = None,
    ) -> None:
        import lmdb

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.dict = mydict
        self.cls = cls

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        target = self.dict[self.cls]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length


class LSUN(VisionDataset):
    """`LSUN <https://www.yf.io/p/lsun>`_ dataset.

    You will need to install the ``lmdb`` package to use this dataset: run
    ``pip install lmdb``

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        classes: Union[str, List[str]] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes = self._verify_classes(classes)
        self.dict = {"bedroom": 0, "bridge": 1, "church_outdoor": 2, "classroom": 3, "conference_room": 4, "dining_room": 5, "kitchen": 6, "living_room": 7, "restaurant": 8, "tower": 9}

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            # print(f'c: {c[:-4]} and {c[:-6]}') # val and train
            self.dbs.append(LSUNClass(root=os.path.join(root, f"{c}_lmdb"), transform=transform, mydict=self.dict, cls=c[:-6]))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:
        categories = [
            "bedroom",
            "bridge",
            "church_outdoor",
            "classroom",
            "conference_room",
            "dining_room",
            "kitchen",
            "living_room",
            "restaurant",
            "tower",
        ]
        dset_opts = ["train", "val", "test"]

        try:
            classes = cast(str, classes)
            verify_str_arg(classes, "classes", dset_opts)
            if classes == "test":
                classes = [classes]
            else:
                classes = [c + "_" + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = "Expected type str or Iterable for argument classes, but got type {}."
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr_type = "Expected type str for elements in argument classes, but got type {}."
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
                c_short = c.split("_")
                category, dset_opt = "_".join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class", iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        img, target = db[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return "Classes: {classes}".format(**self.__dict__)




# from torchvision import transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
# # rem_cls = ['bridge_val', 'classroom_val', 'conference_room_val', 'dining_room_val', 'kitchen_val', 'living_room_val', 'restaurant_val', 'tower_val']
# # unl_cls = ['bedroom_val', 'church_outdoor_val']
# # dataset_rem = LSUN(root='/data/dataset/LSUN/val', classes=rem_cls, transform=transform)
# # dataset_unl = LSUN(root='/data/dataset/LSUN/val', classes=unl_cls, transform=transform)
# # dataset_all = dataset_rem + dataset_unl
# # num_classes = 10
# # trainloader_rem = DataLoader(dataset_rem, batch_size=8, shuffle=True, drop_last=False, num_workers=2)
# # trainloader_unl = DataLoader(dataset_unl, batch_size=8, shuffle=True, drop_last=False, num_workers=2)
# # trainloader_all = DataLoader(dataset_all, batch_size=8, shuffle=True, drop_last=False, num_workers=2)

# # dataloaders = [trainloader_all, trainloader_rem, trainloader_unl]
# # for dataloader in dataloaders:
# #     for j, (imgs, labs) in enumerate(dataloader):
# #         print(j, imgs.shape, labs.shape, labs)
# #         break


# dataset = LSUN(root='/data/dataset/LSUN/train', classes=['bedroom_train'], transform=transform)
# trainloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False, num_workers=2)

# print(len(dataset)) # 3033042

# from torchvision import transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
# rem_cls = ['bridge_train', 'classroom_train', 'conference_room_train', 'dining_room_train', 'kitchen_train', 'living_room_train', 'restaurant_train', 'tower_train']
# unl_cls = ['bedroom_train', 'church_outdoor_train']
# dataset_rem = LSUN(root='/data/dataset/LSUN/train', classes=rem_cls, transform=transform)
# dataset_unl = LSUN(root='/data/dataset/LSUN/train', classes=unl_cls, transform=transform)
# dataset_all = dataset_rem + dataset_unl
# num_classes = 10
# trainloader_rem = DataLoader(dataset_rem, batch_size=8, shuffle=True, drop_last=False, num_workers=2)
# trainloader_unl = DataLoader(dataset_unl, batch_size=8, shuffle=True, drop_last=False, num_workers=2)
# trainloader_all = DataLoader(dataset_all, batch_size=8, shuffle=True, drop_last=False, num_workers=2)

# dataloaders = [trainloader_all, trainloader_rem, trainloader_unl]
# for dataloader in dataloaders:
#     for j, (imgs, labs) in enumerate(dataloader):
#         print(j, imgs.shape, labs.shape, labs)
#         break

