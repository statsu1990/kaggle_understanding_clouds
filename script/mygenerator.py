from keras.utils import Sequence
import numpy as np
import cv2

#from script.cloud_images_segmentation_utillity_script import build_masks
from script.my_util import build_masks


class DataGenerator(Sequence):
    def __init__(self, images, imageName_to_imageIdx_dict, dataframe, batch_size, n_channels, target_size,  n_classes, 
                 mode='fit', target_df=None, shuffle=True, preprocessing=None, augmentation=None, seed=0):
        
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.mode = mode
        self.target_df = target_df
        self.target_size = target_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.seed = seed
        self.mask_shape = (1400, 2100)
        self.list_IDs = self.dataframe.index
        
        self.images = images
        self.imageName_to_imageIdx_dict = imageName_to_imageIdx_dict

        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.on_epoch_end()
        return


    def __len__(self):
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            Y = self.__generate_Y(list_IDs_batch)
            
            if self.augmentation:
                X, Y = self.__augment_batch(X, Y)
            
            return X, Y
        
        elif self.mode == 'predict':
            return X
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        X = np.empty((self.batch_size, *self.target_size, self.n_channels))

        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            #img_path = self.directory + img_name
            #img = cv2.imread(img_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_idx = self.imageName_to_imageIdx_dict[img_name]
            img = self.images[img_idx]

            if self.preprocessing:
                img = self.preprocessing(img)

            X[i,] = img

        return X
    
    def __generate_Y(self, list_IDs_batch):
        Y = np.empty((self.batch_size, *self.target_size, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            image_df = self.target_df[self.target_df['image'] == img_name]
            rles = image_df['EncodedPixels'].values
            masks = build_masks(rles, input_shape=self.mask_shape, reshape=self.target_size)
            Y[i, ] = masks

        return Y
    
    def __augment_batch(self, X_batch, Y_batch):
        for i in range(X_batch.shape[0]):
            X_batch[i, ], Y_batch[i, ] = self.__random_transform(X_batch[i, ], Y_batch[i, ])
        
        return X_batch, Y_batch
    
    def __random_transform(self, X, Y):
        composed = self.augmentation(image=X, mask=Y)
        X_aug = composed['image']
        Y_aug = composed['mask']
        
        return X_aug, Y_aug

class DataGenerator2(Sequence):
    def __init__(self, 
                 images, imageName_to_imageIdx_dict, 
                 masks, imageName_to_maskIdx_dict,
                 dataframe, batch_size, n_channels, target_size,  n_classes, 
                 mode='fit', shuffle=True, preprocessing=None, augmentation=None, preproc_before_aug=True, seed=0,
                 mixup_alpha=None, mixhalf_p=None, mask_avefilter_kernel=None,
                 smooth_overlap_mask_base=None):
        
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.mode = mode
        self.target_size = target_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.preproc_before_aug = preproc_before_aug
        self.seed = seed
        self.mask_shape = (1400, 2100)
        self.list_IDs = self.dataframe.index
        
        self.images = images
        self.imageName_to_imageIdx_dict = imageName_to_imageIdx_dict
        self.masks = masks
        self.imageName_to_maskIdx_dict = imageName_to_maskIdx_dict

        self.mixup_alpha = mixup_alpha
        self.mixhalf_p = mixhalf_p
        self.mask_avefilter_kernel = mask_avefilter_kernel

        self.smooth_overlap_mask_base = smooth_overlap_mask_base

        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.on_epoch_end()
        return

    def __len__(self):
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            Y = self.__generate_Y(list_IDs_batch)
            if self.smooth_overlap_mask_base is not None:
                Y = smooth_overlap_mask_v1(Y, self.smooth_overlap_mask_base)

            if self.mask_avefilter_kernel is not None:
                Y = average_filter(Y, self.mask_avefilter_kernel)

            if self.mixhalf_p is not None:
                X, Y = mixhalf(X, Y, self.mixhalf_p)

            if self.mixup_alpha is not None:
                X, Y = mixup(X, Y, self.mixup_alpha)

            if self.augmentation:
                X, Y = self.__augment_batch(X, Y)

            if not self.preproc_before_aug:
                if self.preprocessing:
                    X = self.preprocessing(X)
            
            return X, Y
        
        elif self.mode == 'predict':
            if not self.preproc_before_aug:
                if self.preprocessing:
                    X = self.preprocessing(X)

            return X
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        # get type
        for i, ID in enumerate(list_IDs_batch[:1]):
            img_name = self.dataframe['image'].loc[ID]
            img_idx = self.imageName_to_imageIdx_dict[img_name]
            img = self.images[img_idx]
            if self.preproc_before_aug:
                if self.preprocessing:
                    img = self.preprocessing(img)
            img_type = img.dtype

        #
        X = np.empty((self.batch_size, *self.target_size, self.n_channels), dtype=img_type)

        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            #img_path = self.directory + img_name
            #img = cv2.imread(img_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_idx = self.imageName_to_imageIdx_dict[img_name]
            img = self.images[img_idx]

            if self.preproc_before_aug:
                if self.preprocessing:
                    img = self.preprocessing(img)

            X[i,] = img

        return X
    
    def __generate_Y(self, list_IDs_batch):
        Y = np.empty((self.batch_size, *self.target_size, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            #image_df = self.target_df[self.target_df['image'] == img_name]
            #rles = image_df['EncodedPixels'].values
            #masks = build_masks(rles, input_shape=self.mask_shape, reshape=self.target_size)
            mask_idx = self.imageName_to_maskIdx_dict[img_name]
            mask = self.masks[mask_idx]
            Y[i, ] = mask

        return Y
    
    def __augment_batch(self, X_batch, Y_batch):
        for i in range(X_batch.shape[0]):
            X_batch[i, ], Y_batch[i, ] = self.__random_transform(X_batch[i, ], Y_batch[i, ])
        
        return X_batch, Y_batch
    
    def __random_transform(self, X, Y):
        composed = self.augmentation(image=X, mask=Y)
        X_aug = composed['image']
        Y_aug = composed['mask']
        
        return X_aug, Y_aug

def mixup(imgs, masks, alpha):
    mixup_rate = np.random.beta(alpha, alpha, len(imgs))

    def __mixup(rate, x):
        rate_shape = list(x.shape)        
        for i in range(len(rate_shape)):
            if i > 0:
                rate_shape[i] = 1
        rate_shape = tuple(rate_shape)
        re_rate = rate.reshape(rate_shape)
        mixup_idx = np.random.permutation(np.arange(len(x)))
        mix_x = re_rate * x + (1.0 - re_rate) * x[mixup_idx]
        return mix_x

    mixed_imgs = __mixup(mixup_rate, imgs)
    mixed_masks = __mixup(mixup_rate, masks)
    return mixed_imgs, mixed_masks

def mixhalf(imgs, masks, p):
    """
    Args:
        imgs: shape = (B, H, W, Channel)
        masks: shape = (B, H, W, Class)
    """
    dev_w = int(imgs.shape[2] * 0.5)

    mix_idx = np.random.permutation(np.arange(len(imgs)))
    mix_imgs = imgs.copy()
    mix_masks = masks.copy()

    for i in range(len(imgs)):
        if np.random.rand() < p:
            mix_imgs[i] = np.concatenate([imgs[i,:,:dev_w,:], imgs[mix_idx[i],:,:dev_w,:]], axis=1)
            mix_masks[i] = np.concatenate([masks[i,:,:dev_w,:], masks[mix_idx[i],:,:dev_w,:]], axis=1)

    return mix_imgs, mix_masks

def average_filter(imgs, kernel_size):
    filtered_imgs = np.zeros_like(imgs, dtype='float32')

    for i, img in enumerate(imgs):
        filtered_imgs[i] = cv2.blur(img.astype('float32'), (kernel_size, kernel_size))

    return filtered_imgs

def smooth_overlap_mask_v1(masks, base_value):
    resi_rate = masks / (np.sum(masks, axis=-1, keepdims=True) + 1e-5)
    smooth_mask = base_value * masks + (1 - base_value) * masks * resi_rate
    return smooth_mask


