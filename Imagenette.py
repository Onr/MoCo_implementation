import os
from PIL import Image
from torch.utils.data import Dataset

class Imagenette2_dataset(Dataset):
    def __init__(self, dir_path, mode, transform=None, target_transform=None, do_all_dataset_in_memory=True):
        self.mode = mode
        dir_path = os.path.join(dir_path, mode)
        class_dirs = [os.path.join(dir_path, cur_path) for cur_path in os.listdir(dir_path) if
                      not cur_path.startswith('.')]
        self.images_paths = []
        self.images_labels = []
        # class_name_to_class_id = {
        #     'chainsaw': 0,
        #     'church': 1,
        #     'dog': 2,
        #     'fish': 3,
        #     'gas_station': 4,
        #     'golf': 5,
        #     'parachute': 6,
        #     'radio': 7,
        #     'truck': 8,
        #     'trumpet': 9,
        # }
        class_names = [class_name for class_name in os.listdir(dir_path)]

        class_name_to_class_id = {}
        for idx, class_name in enumerate(class_names):
            class_name_to_class_id[class_name] = idx

        for cur_dir in class_dirs:
            curr_images_paths = [os.path.join(cur_dir, img_name) for img_name in os.listdir(cur_dir)]
            curr_images_labels = [cur_dir.split('/')[-1]] * len(curr_images_paths)
            self.images_paths += curr_images_paths
            self.images_labels += [class_name_to_class_id[cur_lab] for cur_lab in curr_images_labels]

        self.do_all_dataset_in_memory = do_all_dataset_in_memory
        if self.do_all_dataset_in_memory:
            self.imgs = []
            self.labels = []
            for img_path, img_label in zip(self.images_paths, self.images_labels):
                image_pil = Image.open(img_path).convert('RGB')
                self.imgs += [image_pil]
                self.labels += [img_label]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if not self.do_all_dataset_in_memory:
            img_path = self.images_paths[idx]
            img_label = self.images_labels[idx]
            image_pil = Image.open(img_path).convert('RGB')
        else:
            image_pil = self.imgs[idx]
            img_label = self.labels[idx]
        if self.transform:
            image_q = self.transform(image_pil)
            if self.mode == 'train':
                image_k = self.transform(image_pil)
        if self.target_transform:
            img_label = self.target_transform(img_label)
        if self.mode == 'train':
            return image_q, image_k, img_label
        else:
            return image_q, img_label