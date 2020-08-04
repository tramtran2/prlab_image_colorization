import matplotlib.pyplot as plt
import time, cv2
import numpy as np, os
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import Iterator as KerasIterator
import numpy as np

from image_colorization.datasets import decode_soft_encoding_batch_rgb

class SoftColorizedDataGenerator(KerasIterator):
    def __init__(self, dataloader, 
                 batch_size = 32, 
                 shuffle = True, 
                 x_preprocessing_image_fn = None, 
                 y_preprocessing_image_fn = None, 
                 x_postprocessing_image_fn = None, 
                 y_postprocessing_image_fn = None, 
                 seed = None):
        """
        """
        self.dataloader = dataloader
        
        ret  = self.dataloader[0]
        x, y = ret["x"], ret["y"]
        
        self.x_shape = x.shape
        self.y_shape = y.shape if y is not None else None
        self.current_batch = np.zeros(batch_size) # save current batch for easy debug
        
        self.x_preprocessing_image_fn  = x_preprocessing_image_fn
        self.y_preprocessing_image_fn  = y_preprocessing_image_fn
        self.x_postprocessing_image_fn = x_postprocessing_image_fn
        self.y_postprocessing_image_fn = y_postprocessing_image_fn
        
        super().__init__(len(self.dataloader), batch_size, shuffle, seed)
    # __init__

    def _get_batches_of_transformed_samples(self, index_array):
        self.current_batch = index_array
        
        _backup_verbose = self.dataloader.verbose
        self.dataloader.verbose = 0

        # (height, width, channel)
        batch_x = np.zeros((len(index_array),) + self.x_shape + (1,), dtype=np.float32)
        batch_y = None
        if self.y_shape is not None: 
            batch_y = np.zeros((len(index_array),) + self.y_shape, dtype=np.float32)
        for idx, value in enumerate(index_array):
            ret = self.dataloader[value]
            batch_x[idx, :, :, 0] = ret["x"]
            if self.y_shape is not None: 
                batch_y[idx, :, :] = ret["y"]
        # for
        
        if self.x_preprocessing_image_fn is not None: batch_x = self.x_preprocessing_image_fn(batch_x)
        if self.y_preprocessing_image_fn is not None: batch_y = self.y_preprocessing_image_fn(batch_y)
        
        output = (batch_x, batch_y)

        self.dataloader.verbose = _backup_verbose
        return output
    # _get_batches_of_transformed_samples

    def decode_batch_image(self, x_batch, y_batch, version = 0, usegpu = False, epsilon = 1e-8, T = 0.38, **kwargs):
        result = decode_soft_encoding_batch_rgb(x_batch, y_batch, self.dataloader.soft_encoding.q_ab,
                                                        epsilon = epsilon, T = T, x_post_fn = self.x_postprocessing_image_fn, 
                                                        version = version, usegpu = usegpu, **kwargs)
        return result
    # decode_batch_image

    def view_batch(self, batch_x, batch_y, only_info = False, ncols = 4, figsize = (20, 8), save_path = None, print = print):
        print("Generator Output Information: ")
    
        print(f'+ batch_x: {batch_x.shape} - [{np.min(batch_x)} - {np.max(batch_x)}]')
        if batch_y is not None:
            print(f'+ batch_y: {batch_y.shape} - [{np.min(batch_y)} - {np.max(batch_y)}]')
        if only_info == True: return

        batch_x_rgb = self.x_postprocessing_image_fn(batch_x) if self.x_postprocessing_image_fn is not None else batch_x
        batch_y_rgb = self.decode_batch_image(batch_x, batch_y)

        nrows = (len(batch_x) + ncols - 1) // ncols

        for irow in range(nrows):
            fig = plt.figure(figsize = figsize)
            for icol in range(ncols):
                idx = irow * ncols + icol
                if idx>=len(batch_x): break
                
                input_image  = batch_x_rgb[idx, :, :, 0]
                output_image = batch_y_rgb[idx, :, :, :]
                
                plt.subplot(1,2 * ncols, 2 * icol + 1), plt.imshow(input_image, cmap='gray'), plt.axis("off"), plt.title(f"Input {idx}");
                plt.subplot(1,2 * ncols, 2 * icol + 2), plt.imshow(output_image), plt.axis("off"), plt.title(f"Output {idx}");
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
    # view_batch    
# SoftColorizedDataGenerator