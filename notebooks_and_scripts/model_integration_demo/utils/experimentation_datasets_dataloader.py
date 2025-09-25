import glob
import os

import torch.utils.data as data


class ExperimentationDatasets(data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.mesh_paths = []
        self.labels = []
        self.class_names = []

        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_dirs.sort()

        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(root_dir, class_name)

            split_dir = os.path.join(class_dir, split)
            if os.path.exists(split_dir):
                search_dir = split_dir
            else:
                search_dir = class_dir

            obj_files = glob.glob(os.path.join(search_dir, "*.obj"))

            for obj_file in obj_files:
                self.mesh_paths.append(obj_file)
                self.labels.append(class_idx)

            self.class_names.append(class_name)

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        return self.mesh_paths[idx], self.labels[idx]
