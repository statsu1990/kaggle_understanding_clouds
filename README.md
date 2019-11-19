# kaggle_understanding_clouds
Kaggle competition Understanding Clouds from Satellite Images
https://www.kaggle.com/c/understanding_cloud_organization

## Competition overview
- Segment satellite images based on cloud shape.
- There are 4types of clouds.

## Solution overview
- ranking : 180th / 1556 teams
- score : Public 0.65691 / Private 0.65430
- method : segmentation using deep learning
  - segmentation model : deeplab v3+ (backbone MobileNetV2)<br>
  https://github.com/bonlime/keras-deeplab-v3-plus
  - optimizer : RAdam<br>
  https://pypi.org/project/keras-rectified-adam/
  - loss : binary cross entropy + dice loss
  - data augmentation : h and v flip, ShiftScaleRotate, RandomBrightness<br>
  https://github.com/albumentations-team/albumentations
  - test time augmentation : h and v flip, h shift x2, v shift x2, rotation 180<br>
  https://github.com/qubvel/tta_wrapper
  - ensemble : 6 model
