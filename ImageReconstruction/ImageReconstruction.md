## Dataset preparation
If you finish the testing of stage 1, the warped images and corresponding masks will be generated in the 'ImageAlignment/output' folder. And these images are the input of stage 2.

## VGG19 pretrained model
Download the pretrained model of vgg19 from:
```
https://github.com/machrisaa/tensorflow-vgg
```
Then move the vgg model to 'ImageReconstruction/vgg19/'.

## Training
Modidy the 'ImageReconstruction/Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 200,000.

Train the reconstruction network on UDIS-D:
```
cd ImageReconstruction/Codes/
python train.py
```

## Testing 
Our pretrained reconstruction model can be available at [Google Drive](https://drive.google.com/drive/folders/1ogfVFveLjvbnf_XRKAMjG5WaUzxz_zrb?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1jYtiwibIL0dDfDalw1NR0w)(Extraction code: 1234).

Modidy the 'ImageReconstruction/Codes/constant.py' to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'ImageReconstruction/Codes/inference.py'.

Generate the stitched image:
```
python inference.py
```
#### About the pretrain reconstruction model
We are sorry that we delete the pretrained reconstruction model whose result is the same as that in the paper by mistake. We retrain the reconstruction model and release it.
