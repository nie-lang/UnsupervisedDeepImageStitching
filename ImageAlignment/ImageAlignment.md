## Dataset preparation
##### Step 1: Stitched MS-COCO (The synthetic dataset)
Generate the synthetic dataset that proposed in VFISNet[1]. In our experiment, we generate 50,000 for training and 5,000 for testing.

Notice: Since our solution is unsupervised, we only generate the input images. We do not need the corresponding lables (homography label and stitched image label).

##### Step 2: UDIS-D (The proposed real-world dataset)
Download this dataset. 

## Training
##### Step 1: Unsupervised training on Stitched MS-COCO
Modidy the 'ImageAlignment/Codes/constant.py' to set the 'TRAIN_FOLDER'/'TEST_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, it takes 600,000 iterations to train on this synthetic dataset.

Train on Stitched MS-COCO:
```
python train_H.py
```

##### Step 2: Unsupervised training on UDIS-D
Download this dataset. 

## Testing 
##### Step 1: Stitched MS-COCO (The synthetic dataset)

#### References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
