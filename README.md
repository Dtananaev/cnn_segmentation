# cnn_segmentation

## Upload data
Upload data files stage_1_test.zip and stage_1_train.zip from: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277 

## Prepare dataset:
1. Unpack stage_1_test.zip, stage_1_train.zip into `<path_to_repo>/cnn_segmentation/data`
2. Run to create one segmentation mask out of multiple: 
```
python cnn_segmentation/create_combined_mask.py 
```
3. Train network by running:
```
python train.py
```
4. Inference:
```
python inference.py --mode_dir=checkpoints/model-<number>
```
