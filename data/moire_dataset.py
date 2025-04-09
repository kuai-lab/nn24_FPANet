# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR), NAFNet(https://github.com/megvii-research/NAFNet) 
# Copyright 2018-2020 BasicSR Authors
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_util import (paired_paths_from_folder)
from data.transform import augment, paired_random_crop, imfrombytes, padding
from utils.utils import FileClient, img2tensor
import os


class VDMDataset(data.Dataset):
    def __init__(self, opt):
        super(VDMDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.mode = opt['mode']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
       
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # self.paths = self.paths[:10]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        
        video_number = lq_path.split('/')[-1][0:7]
        frame = int(lq_path.split('/')[-1][7:12])
        file_type = lq_path.split('/')[-1].split('.')[-1]
        
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        # auxiliary frame around img_lq
        if self.mode == 'multi':       
            if frame == 0 or frame == 59:
                if frame == 0:
                    frame = 1
                else:
                    frame = 58

            frame_next = video_number + '%05d' % (frame+1) + '.' + file_type
            frame_prev = video_number + '%05d' % (frame-1) + '.' + file_type
            
            next_path = os.path.join(self.lq_folder, frame_next)
            prev_path = os.path.join(self.lq_folder, frame_prev)
    
            img_bytes = self.file_client.get(next_path, 'next')
            try:
                img_next = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("next path {} not working".format(next_path))

            img_bytes = self.file_client.get(prev_path, 'prev')
            try:
                img_prev = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("next path {} not working".format(prev_path))

            imgs = [img_gt, img_lq, img_next, img_prev]
            # augmentation for training
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                
                # padding
                imgs = padding(imgs, gt_size)

                # random crop
                imgs= paired_random_crop(imgs, gt_size)
                
                # flip, rotation
                imgs = augment(imgs, self.opt['use_flip'], self.opt['use_rot'])

            
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq, img_next, img_prev = img2tensor(imgs, bgr2rgb=True, float32=True)
            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)
                normalize(img_next, self.mean, self.std, inplace=True)
                normalize(img_prev, self.mean, self.std, inplace=True)
            
            return {
                'lq': img_lq,
                'gt': img_gt,
                'next' : img_next,
                'prev' : img_prev,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'next_path': next_path,
                'prev_path': prev_path
            }
        else:
            # augmentation for training
            imgs = [img_gt, img_lq]
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                
                # padding
                imgs = padding(imgs, gt_size)

                # random crop
                imgs = paired_random_crop(imgs, gt_size)
                
                # flip, rotation
                imgs = augment(imgs, self.opt['use_flip'], self.opt['use_rot'])

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor(imgs, bgr2rgb=True, float32=True)
            
            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)

            return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
            }

    def __len__(self):
        return len(self.paths)
