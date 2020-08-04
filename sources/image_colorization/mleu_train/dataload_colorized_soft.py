"""
Update: 2020/06/20
ColorDataLoader:
+ output_type: 
  * normal --> file_path --> (gray, ab)
  * softencoding --> file_path --> (gray, category ab)
"""

import os, pandas as pd, numpy as np, cv2
import matplotlib.pyplot as plt
import sklearn.neighbors as nn
import pprint

from image_colorization.datasets import ColorizedSoftEncoding
from image_colorization.datasets import decode_soft_encoding_image_rgb

default_output_cfg = dict(
    softencoding = dict(
        nb_neighbors     = 5,  
        pts_in_hull_path = "pts_in_hull.npy",
        sigma_neighbor   = 5,
    )
)

class SoftColorizedDataLoader(object):
    """
        x_data   : dict(image = array of image files)
        transform: data augmentation
        mode     : train, valid, test
    """
    def __init__(self, 
                 x_data,
                 transforms = None, 
                 mode = "train", 
                 capacity = 0,
                 
                 output_size  = (64, 64),
                 output_cfg   = default_output_cfg,

                 x_preprocessing_image_fn = None, 
                 y_preprocessing_image_fn = None,
                 
                 verbose = 0, 
                 **kwargs):
        
        # init param
        self.x_data = x_data
        self.n_data = len(self.x_data["image"])
        
        self.capacity   = capacity
        self.mode       = mode
        self.transforms = transforms
        self.x_preprocessing_image_fn = x_preprocessing_image_fn
        self.y_preprocessing_image_fn = y_preprocessing_image_fn
        
        self.output_type   = "soft"
        self.output_cfg    = output_cfg
        self.output_size   = output_size
        self.soft_encoding = ColorizedSoftEncoding(**self.output_cfg["softencoding"])
        
        self.idx_all = np.arange(self.n_data)
        self._verbose = verbose # debug image
        pass
    # __init__

    @property
    def verbose(self):
        return self._verbose
    # verbose getter

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
    # verbose setter
    
    def __len__(self):
        if self.capacity == 0: return self.n_data
        return self.capacity
    # __len__
    
    def __getitem__(self, index):
        if index >= self.n_data: index = index % self.n_data
        self.org_index = index
            
        result = {}
            
        # read original images
        image_path    = self.x_data["image"][index]
        org_img_color = cv2.imread(image_path)    # BGR Image
        org_img_color = org_img_color[:, :, ::-1] # RGB convert

        self.org_height, self.org_width = org_img_color.shape[0:2]
               
        # only convert original in verbose mode
        if self.verbose == 1:
            org_img_gray = cv2.cvtColor(org_img_color, cv2.COLOR_RGB2GRAY)
            org_img_Lab  = cv2.cvtColor(org_img_color, cv2.COLOR_RGB2Lab)
            
            result.update({"org_img_gray": org_img_gray})
            result.update({"org_img_color": org_img_color})
            result.update({"org_img_Lab": org_img_Lab})
        # if
        
        transform_data = {"image": org_img_color}
        transform_type = {"image": "image"}
        if self.transforms is not None:
            self.transforms.add_targets(transform_type)
            transform_result = self.transforms(**transform_data)
        else:
            transform_result = transform_data                     
        # if
        transform_type["resize_img_color"] = "resize_img_color"
        transform_result["resize_img_color"] = transform_result["image"]
        transform_result.pop("image")
        transform_type.pop("image")

        transform_result["resize_img_Lab"]   = cv2.cvtColor(transform_result["resize_img_color"], cv2.COLOR_RGB2Lab)    
        if self.verbose == 1:    
            transform_result["resize_img_gray"] = cv2.cvtColor(transform_result["resize_img_color"], cv2.COLOR_RGB2GRAY)
            result.update(**transform_result)
        # if

        transform_output = {}
        if self.output_size is not None:
            for key in transform_type:
                transform_output[f"{key}_output"] = cv2.resize(transform_result[key], self.output_size, 0, 0, cv2.INTER_NEAREST)
            # for
        # if
        transform_output["resize_img_Lab_output"]   = cv2.cvtColor(transform_output["resize_img_color_output"], cv2.COLOR_RGB2Lab)    
        if self.verbose == 1:    
            transform_output["resize_img_gray_output"] = cv2.cvtColor(transform_output["resize_img_color_output"], cv2.COLOR_RGB2GRAY)
            result.update(**transform_output)
        # if
        
        x, y = None, None
        x = transform_result["resize_img_Lab"][:, :, 0] # L channel
        
        if self.x_preprocessing_image_fn is not None and x is not None:
            x = self.x_preprocessing_image_fn(x)
        # if
        
        resize_img_Lab_new = transform_output["resize_img_Lab_output"]
        if self.y_preprocessing_image_fn is not None:
            resize_img_Lab_new = self.y_preprocessing_image_fn(resize_img_Lab_new)
        # if

        y = self.soft_encoding(resize_img_Lab_new)
        
        result.update({"x": x})
        result.update({"y": y})
        return result
    # __getitem__

    def decode_image(self, x, y, x_post_fn = None, y_post_fn = None, version = 0, usegpu = False, epsilon = 1e-8, T = 0.38, **kwargs):
        result = decode_soft_encoding_image_rgb(x, y, self.soft_encoding.q_ab, 
                                                x_post_fn = x_post_fn, epsilon = epsilon, T = T, 
                                                version = version, usegpu = usegpu, **kwargs);
        return result
    # decode_image
    
    def view_image(self, idx, save_path = None, figsize = (10, 8), only_info = False, print = print):
        _verbose = self.verbose
        self.verbose = True
        result = self[idx]     
        self.verbose = _verbose
        
        decode_image = None

        print("Output information: ")
        if result.get("x") is not None:
            print(f'x: shape={result["x"].shape} - [min={np.min(result["x"])}, max={np.max(result["x"])}]')
        if result.get("y") is not None:
            decode_image = self.decode_image(result["x"], result["y"])
            print(f'y: shape={result["y"].shape} - [min={np.min(result["y"])}, max={np.max(result["y"])}]')

        print(f'output_cfg   : shape={pprint.pformat(self.output_cfg)}')
        print(f'output_size  : shape={pprint.pformat(self.output_size)}')
        if only_info == True: return


        fig1 = plt.figure(figsize=figsize)
        plt.subplot(1,5,1), plt.imshow(result["org_img_color"]), plt.axis("off"), plt.title('Color Image')
        plt.subplot(1,5,2), plt.imshow(result["org_img_gray"], cmap='gray'), plt.axis("off"), plt.title('Gray')
        plt.subplot(1,5,3), plt.imshow(result["org_img_Lab"][...,0], cmap='gray'), plt.axis("off"), plt.title('L channel')
        plt.subplot(1,5,4), plt.imshow(result["org_img_Lab"][...,1], cmap='gray'), plt.axis("off"), plt.title('a channel')
        plt.subplot(1,5,5), plt.imshow(result["org_img_Lab"][...,2], cmap='gray'), plt.axis("off"), plt.title('a channel')

        fig2 = plt.figure(figsize=figsize)
        plt.subplot(1,2,1), plt.imshow(result["x"], cmap='gray'), plt.axis("off"), plt.title("Input");
        if decode_image is not None:
            plt.subplot(1,2,2), plt.imshow(decode_image), plt.axis("off"), plt.title("Decode Output");

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir != "" and os.path.exists(save_dir) == False: os.makedirs(save_dir)
            fig1.savefig(f'{os.path.splitext(save_path)[0]}.view1{os.path.splitext(save_path)[1]}')
            fig2.savefig(f'{os.path.splitext(save_path)[0]}.view2{os.path.splitext(save_path)[1]}')
        # if        
        del fig1
        del fig2
        pass
    # view_image

    def view_images(self, idx_show = None, ncols = 3, figsize = (20, 8), save_path = None):
        if idx_show is None: 
            idx_show = np.random.choice(len(self.x_data["image"]), 15)
        # if
        nrows = (len(idx_show) + ncols - 1) // ncols

        for irow in range(nrows):
            fig = plt.figure(figsize = figsize)
            for icol in range(ncols):
                idx = irow * ncols + icol
                if idx>=len(idx_show): break
                _verbose = self.verbose
                self.verbose = True
                index = idx_show[idx]
                self.verbose = _verbose
                
                result = self[index]

                decode_image = None
                if result.get("y") is not None: decode_image = self.decode_image(result["x"], result["y"])
                
                org_image    = cv2.resize(result["org_img_color"], (decode_image.shape[1],decode_image.shape[0]))
                
                plt.subplot(1,3 * ncols, 3 * icol + 1), plt.imshow(org_image), plt.axis("off"), plt.title("Original");
                plt.subplot(1,3 * ncols, 3 * icol + 2), plt.imshow(result["x"], cmap='gray'), plt.axis("off"), plt.title("Input");
                if decode_image is not None:
                    plt.subplot(1,3 * ncols, 3 * icol + 3), plt.imshow(decode_image), plt.axis("off"), plt.title("Decode Output");
            # for
            plt.show();

            if save_path is not None:
                save_dir = os.path.dirname(save_path)
                if save_dir != "" and os.path.exists(save_dir) == False: 
                    os.makedirs(save_dir)
                fig.savefig(f'{os.path.splitext(save_path)[0]}.view{irow}{os.path.splitext(save_path)[1]}')
            # if        
            del fig
        # for
        plt.show(), plt.close();
        pass
    # view_images
# SoftColorizedDataLoader