from script.cloud_images_segmentation_utillity_script import *
from keras.models import load_model
from tta_wrapper import tta_segmentation
import numpy as np

from model import base_unet, myunet, mylosses, mydeeplab
from script import mygenerator as mygen
import os
import datetime
import cv2
import matplotlib.pyplot as plt


def read_img(img_shape, directory, dataframe):
    list_IDs = dataframe.index

    imgs = np.empty((len(list_IDs), *img_shape), dtype='uint8')
    imageName_to_imageIdx_dict = []
    for i, ID in enumerate(list_IDs):
        if i % 100 == 0:
            print("\r read img {0}/{1}".format(i+1, len(list_IDs)), end="")
        img_name = dataframe['image'].loc[ID]
        img_path = directory + img_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if np.isnan(img).any():
            print('{0}, {1}'.format(i, ID))

        imgs[i,] = img
        imageName_to_imageIdx_dict.append((img_name, i))

    imageName_to_imageIdx_dict = dict(imageName_to_imageIdx_dict)
    print("")

    return imgs, imageName_to_imageIdx_dict

def calc_mask(img_shape, target_df, dataframe, mask_shape=(1400, 2100)):
    list_IDs = dataframe.index

    Y = np.empty((len(list_IDs), *img_shape[:2], 4), dtype='uint8')
    imageName_to_maskIdx_dict = []
    for i, ID in enumerate(list_IDs):
        if i % 100 == 0:
            print("\r calc mask {0}/{1}".format(i+1, len(list_IDs)), end="")
        
        img_name = dataframe['image'].loc[ID]
        image_df = target_df[target_df['image'] == img_name]
        rles = image_df['EncodedPixels'].values
        masks = build_masks(rles, input_shape=mask_shape, reshape=img_shape[:2])
        
        if np.isnan(masks).any():
            print('{0}, {1}'.format(i, ID))

        Y[i, ] = masks
        imageName_to_maskIdx_dict.append((img_name, i))

    imageName_to_maskIdx_dict = dict(imageName_to_maskIdx_dict)
    print("")

    return Y, imageName_to_maskIdx_dict

def post_process_in_black(pred_masks, img):
    not_in_black = np.ones_like(pred_masks) * np.any(img!=0, axis=-1)[...,np.newaxis]
    post_masks = pred_masks * not_in_black
    return not_in_black



def run19110601():
    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110601
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    X_train = X_train
    X_val = X_val


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 32 # 32
    EPOCHS = 30 #12
    LEARNING_RATE = 0.001 #3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'uNet_19110601_v1.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5)
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110602():
    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110602
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    #X_train = X_train[:40]
    #X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 32 # 32
    EPOCHS = 30 #12
    LEARNING_RATE = 0.001 #3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'uNet_19110602_v2.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5)
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110603():
    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110603
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    X_train = X_train
    X_val = X_val


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 30 #12
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110603_v1.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5)
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110701():
    DEBUG = False
    SHOW_IMG = False

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110701
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110701_v2.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                                       border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 albu.Cutout(num_holes=12, max_h_size=32, max_w_size=32, p=0.5),
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110702():
    DEBUG = False
    SHOW_IMG = False

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110702
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110702_v2.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                                       border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 albu.Cutout(num_holes=12, max_h_size=32, max_w_size=32, p=0.5),
                                ])
    MIXUP_ALPHA = 0.4

    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed,
                      mixup_alpha=MIXUP_ALPHA)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110703():
    DEBUG = False
    SHOW_IMG = False

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110703
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 5 #3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110703_v3.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                                       border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 #albu.Cutout(num_holes=12, max_h_size=32, max_w_size=32, p=0.5),
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110704():
    DEBUG = False
    SHOW_IMG = False

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110704
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 5 #3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110704_v4.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                 #                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 #albu.Cutout(num_holes=12, max_h_size=32, max_w_size=32, p=0.5),
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110705():
    DEBUG = False
    SHOW_IMG = False

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110705
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 5 #3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110705_v5.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                 #                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 #albu.Cutout(num_holes=12, max_h_size=32, max_w_size=32, p=0.5),
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110706():
    DEBUG = False
    SHOW_IMG = False

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110706
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 5 #3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110706_v6.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                 #                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 #albu.Cutout(num_holes=12, max_h_size=32, max_w_size=32, p=0.5),
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls02_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110707():
    DEBUG = False
    SHOW_IMG = True

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110707
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 5 #3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110707_v7.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                                       border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),
                                ])
    preproc_before_aug = False

    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      preproc_before_aug=preproc_before_aug,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls02_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            #pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        #pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())

def run19110708():
    DEBUG = False
    SHOW_IMG = True

    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110708
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=19110303)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    if DEBUG:
        X_train = X_train[:40]
        X_val = X_val[:40]


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 16 # 32
    EPOCHS = 60 if not DEBUG else 1
    LEARNING_RATE = 3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 5 #3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'deeplav_19110708_v8.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5),
                                 #albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=20, shift_limit=0.1, scale_limit=0.05, 
                                                       border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                 albu.RandomBrightness(limit=0.2, p=0.99),
                                 #albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),
                                ])
    preproc_before_aug = False

    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    if DEBUG:
        for i in range(10):
            plt.imshow(augmentation(image=train_imgs[i])['image'])
            plt.show()

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      preproc_before_aug=preproc_before_aug,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)

    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v3(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = mydeeplab.mydeeplab_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    if DEBUG:
        test = test[:10]
    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask
            #pred_masks_post = post_process_in_black(pred_masks_post, test_imgs[test_imageName_to_imageIdx_dict[filename]])

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask
        #pred_masks_post = post_process_in_black(pred_masks_post, valid_imgs[valid_imageName_to_imageIdx_dict[filename]])

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    if SHOW_IMG:
        inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    if SHOW_IMG:
        inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())


def run2():
    result_dir = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    seed = 19110303
    seed_everything(seed)
    warnings.filterwarnings("ignore")


    # Load data
    train = pd.read_csv('../input/train.csv')
    submission = pd.read_csv('../input/sample_submission.csv')

    # Preprocecss data
    train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
    test = pd.DataFrame(submission['image'].unique(), columns=['image'])

    # Create one column for each mask
    train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
    train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

    # Train and validation split
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=seed)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    X_train = X_train
    X_val = X_val


    # Model parameters
    #BACKBONE = 'mobilenetv2' #'resnet18'
    BATCH_SIZE = 32 # 32
    EPOCHS = 100 #12
    LEARNING_RATE = 0.001 #3e-4
    HEIGHT = 384
    WIDTH = 480
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5
    model_path = os.path.join(result_dir, 'uNet_191106_v1.h5')
    #
    #preprocessing = sm.get_preprocessing(BACKBONE)
    def preprocessing(_img):
        _img = (_img - 127.5) / 127.5
        return _img

    #
    augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5)
                                ])


    # Pre-process data
    train_base_path = '../input/train_images/'
    test_base_path = '../input/test_images/'
    train_images_dest_path = '../proc_input/train_images/'
    validation_images_dest_path = '../proc_input/validation_images/'
    test_images_dest_path = '../proc_input/test_images/'

    PRE_PROCESS_DATA = False

    if PRE_PROCESS_DATA:
        # Making sure directories don't exist
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)
        if os.path.exists(validation_images_dest_path):
            shutil.rmtree(validation_images_dest_path)
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)
    
        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path)
        os.makedirs(validation_images_dest_path)
        os.makedirs(test_images_dest_path)

        def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
            '''
            This function needs to be defined here, because it will be called with no arguments, 
            and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
            '''
            df = df.reset_index()
            for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['image']
                item_set = item['set']
                if item_set == 'train':
                    preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'validation':
                    preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
                if item_set == 'test':
                    preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)

        # Pre-procecss train set
        pre_process_set(X_train, preprocess_data)

        # Pre-procecss validation set
        pre_process_set(X_val, preprocess_data)

        # Pre-procecss test set
        pre_process_set(test, preprocess_data)


    # read image
    train_imgs, train_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), train_images_dest_path, X_train)
    valid_imgs, valid_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), validation_images_dest_path, X_val)

    # calc mask
    train_masks, train_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_train)
    valid_masks, valid_imageName_to_maskIdx_dict = calc_mask((HEIGHT, WIDTH, CHANNELS), train, X_val)

    # Data generator
    train_generator = mygen.DataGenerator2(
                      images=train_imgs,
                      imageName_to_imageIdx_dict=train_imageName_to_imageIdx_dict,
                      masks=train_masks,
                      imageName_to_maskIdx_dict=train_imageName_to_maskIdx_dict,
                      dataframe=X_train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,
                      seed=seed)
    valid_generator = mygen.DataGenerator2(
                      images=valid_imgs,
                      imageName_to_imageIdx_dict=valid_imageName_to_imageIdx_dict,
                      masks=valid_masks,
                      imageName_to_maskIdx_dict=valid_imageName_to_maskIdx_dict,
                      dataframe=X_val,
                      batch_size=BATCH_SIZE, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      seed=seed)


    #model = base_unet.unet(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    #model = myunet.unet_v1(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)
    model = myunet.unet_v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_class=4)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
    csvlogger = CSVLogger(os.path.join(result_dir, 'learning_log.csv'))

    metric_list = [dice_coef, sm.metrics.iou_score]
    #callback_list = [checkpoint, es, rlrop, csvlogger]
    callback_list = [checkpoint, rlrop, csvlogger]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=mylosses.bce_ls01_dice_loss, metrics=metric_list)
    model.summary()


    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callback_list,
                                  epochs=EPOCHS,
                                  verbose=1).history
    #model.save(model_path)

    # Load model trained longer
    #model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':mylosses.bce_ls01_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})


    # Threshold and mask size tunning
    #  - Here we could use some kind of parameter search, but to simplify I'm using default values
    class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
    best_tresholds = [.5, .5, .5, .35]
    best_masks = [25000, 20000, 22500, 15000]

    for index, name in enumerate(class_names):
        print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))


    # Model evaluation
    train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')
    print(train_metrics)

    validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')
    print(validation_metrics)


    # Apply model to test set
    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    test_imgs, test_imageName_to_imageIdx_dict = read_img((HEIGHT, WIDTH, CHANNELS), test_images_dest_path, test)

    test_df = []
    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        batch_set = test[batch_idx[0]: batch_idx[-1]+1]
        
        test_generator = mygen.DataGenerator2(
                          images=test_imgs,
                          imageName_to_imageIdx_dict=test_imageName_to_imageIdx_dict,
                          masks=None,
                          imageName_to_maskIdx_dict=None,
                          dataframe=batch_set,
                          batch_size=1, 
                          target_size=(HEIGHT, WIDTH),
                          n_channels=CHANNELS,
                          n_classes=N_CLASSES,
                          preprocessing=preprocessing,
                          seed=seed,
                          mode='predict',
                          shuffle=False)

        #test_generator = DataGenerator(
        #                  directory=test_images_dest_path,
        #                  dataframe=batch_set,
        #                  target_df=submission,
        #                  batch_size=1, 
        #                  target_size=(HEIGHT, WIDTH),
        #                  n_channels=CHANNELS,
        #                  n_classes=N_CLASSES,
        #                  preprocessing=preprocessing,
        #                  seed=seed,
        #                  mode='predict',
        #                  shuffle=False)
    
        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            filename = test['image'].iloc[b]
            image_df = submission[submission['image'] == filename].copy()
            pred_masks = preds[index, ].round().astype(int)
            pred_rles = build_rles(pred_masks, reshape=(350, 525))
            image_df['EncodedPixels'] = pred_rles

            ### Post procecssing
            pred_masks_post = preds[index, ].astype('float32') 
            for class_index in range(N_CLASSES):
                pred_mask = pred_masks_post[...,class_index]
                pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
                pred_masks_post[...,class_index] = pred_mask

            pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
            image_df['EncodedPixels_post'] = pred_rles_post
            ###
        
            test_df.append(image_df)

    sub_df = pd.concat(test_df)


    # Inspecting some of the validation set predictions
    # ## Without post-processing

    # Choose 3 samples at random
    images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
    inspect_set = train[train['image'].isin(images_to_inspect)].copy()
    inspect_set_temp = []

    inspect_generator = DataGenerator(
                        directory=validation_images_dest_path,
                        dataframe=inspect_set,
                        target_df=train,
                        batch_size=1, 
                        target_size=(HEIGHT, WIDTH),
                        n_channels=CHANNELS,
                        n_classes=N_CLASSES,
                        preprocessing=preprocessing,
                        seed=seed,
                        mode='fit',
                        shuffle=False)

    preds = model.predict_generator(inspect_generator)

    for index, b in enumerate(range(len(preds))):
        filename = inspect_set['image'].iloc[b]
        image_df = inspect_set[inspect_set['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels_pred'] = pred_rles
    
        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_pred_post'] = pred_rles_post
        ###
        inspect_set_temp.append(image_df)

    inspect_set = pd.concat(inspect_set_temp)
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')


    # With post-processing
    inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')

    # Inspecting some of the test set predictions
    # 
    # Without post-process
    # Choose 5 samples at random
    images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)

    # ## With post-process
    inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path, label_col='EncodedPixels_post')

    # Regular submission
    submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
    submission_df.to_csv(os.path.join(result_dir, 'submission.csv'), index=False)
    print(submission_df.head())

    # Submission with post processing
    submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
    submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
    submission_df_post.to_csv(os.path.join(result_dir, 'submission_post.csv'), index=False)
    print(submission_df_post.head())