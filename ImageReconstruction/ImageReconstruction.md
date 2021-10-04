## Dataset preparation
If you finish the testing of stage 1, the warped images and corresponding masks will be generated in the 'ImageAlignment/output' folder. And these images are the input of stage 2.

## Training
Modidy the 'ImageReconstruction/Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 200,000.

Train the reconstruction network on UDIS-D:
```
python train.py
```

## Testing 
Our pretrained model is available here ([Reconstruction Network]).

Modidy the 'ImageReconstruction/Codes/constant.py' to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'ImageReconstruction/Codes/inference.py'.

Generate the stitched image:
```
python inference.py
```
