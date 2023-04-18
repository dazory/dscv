import torch
import numpy as np

from .augmentations import type_to_op

class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self,
                 dataset,
                 preprocess,
                 augmentations=None,
                 no_jsd=False,
                 aug_prob_coeff=1.0,
                 mixture_width=3,
                 mixture_depth=-1,
                 aug_severity=3,
                 **kwargs,
                 ):
        self.dataset = dataset
        self.preprocess = preprocess
        self.aug_list = self._get_augmentations(augmentations)
        self.no_jsd = no_jsd

        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return self.aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), self.aug(x, self.preprocess),
                        self.aug(x, self.preprocess))
            return im_tuple, y

    def aug(self, image, preprocess):
        """Perform AugMix augmentations and compute mixture.

        Args:
          image: PIL.Image input image
          preprocess: Preprocessing function which should return a torch tensor.

        Returns:
          mixed: Augmented and mixed image.
        """
        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))
        img_size = image.size[0]

        mix = torch.zeros_like(preprocess(image))
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity, img_size)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix
        return mixed

    def _get_augmentations(self, augmentations):
        aug_list = []
        for aug in augmentations:
            aug_list.append(type_to_op[aug])
        return aug_list

    def __len__(self):
        return len(self.dataset)