## The architecture of the homography network
The architecture of the homography network is the same as [2]. We only modify the loss function in this work.

## Dataset preparation
#### Step 1: Stitched MS-COCO (The synthetic dataset)
Generate the synthetic dataset that proposed in VFISNet[1]. In our experiment, we generate 50,000 for training and 5,000 for testing.

Notice: Since our solution is unsupervised, we only generate the input images. We do not need the corresponding lables (homography label and stitched image label).

Modidy the 'ImageAlignment/synthetic_dataset.py' to set the 'raw_image_path'/'generate_image_path' and create the corresponding folders. Then, run this script:
```
cd ImageAlignment/
python synthetic_dataset.py
```

#### Step 2: UDIS-D (The proposed real-world dataset)
Download this dataset. 

## Training
#### Step 1: Unsupervised training on Stitched MS-COCO
Modidy the 'ImageAlignment/Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, it takes 600,000 iterations to train on this synthetic dataset.

Train on Stitched MS-COCO:
```
cd ImageAlignment/Codes/
python train_H.py
```

#### Step 2: Unsupervised training on UDIS-D
Modidy the 'ImageAlignment/Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, it takes another 400,000 iterations to finetune on this real-world dataset. So, the 'ITERATIONS' should be set to 1,000,000.

Train on UDIS-D:
```
python train_H.py
```

## Testing 
Our pretrained homography model can be available at [Google Drive](https://drive.google.com/drive/folders/10SCpFs0J05korpK0sWeSUVGtcpUdPy1e?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1uh6WRp3yWBCD0VPPiCa0eA)(Extraction code: 1234).
#### Caculate the PSNR/SSIM
Modidy the 'ImageAlignment/Codes/constant.py' to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'ImageAlignment/Codes/inference.py'.
Caculate the PSNR/SSIM:
```
python inference.py
```
#### Generate the coarsely aligned images and content masks
Modidy the 'ImageAlignment/Codes/constant.py' to set the 'GPU'. The path for the checkpoint file can be modified in 'ImageAlignment/Codes/inference.py'.
```
python output_inference.py
```
The generated images and masks are used to train the subsequent reconstruction network.

### References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
[2] L. Nie, C. Lin, K. Liao, and Y. Zhao, “Learning edge-preserved image stitching from large-baseline deep homographyn,” arXiv preprint arXiv:2012.06194, 2020. 
