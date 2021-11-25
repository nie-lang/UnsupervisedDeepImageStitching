# Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images 
<p align="center">Lang Nie*, Chunyu Lin*, Kang Liao*, Shuaicheng Liu`, Yao Zhao*</p>
<p align="center">* Institute of Information Science, Beijing Jiaotong University</p>
<p align="center">` School of Information and Communication Engineering, University of Electronic Science and Technology of China</p>

![image](https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/network.jpg)

## Dataset (UDIS-D)
The details of the dataset can be found in our paper. ([arXiv](https://arxiv.org/pdf/2106.12859.pdf), [IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9472883))

We release our testing results with the proposed dataset together. One can download it in in [Google Drive](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/13KZ29e487datgtMgmb9laQ)(Extraction code: 1234).

## Compared with ours
You can try the testing set of the proposed dataset with your own algorithm. And our results in the testing set are also provided with the testing set. 

## Codes
#### Requirement
* python 3.6
* numpy 1.18.1
* tensorflow 1.13.1
* tensorlayer 1.8.0

#### How to run it
Due to the memory limitation of our GPU, we implement this unsupervised solution in two stages:
* Stage 1 (unsupervised coarse image alignment): please refer to [ImageAlignment/ImageAlignment.md](https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/ImageAlignment/ImageAlignment.md).
* Stage 2 (unsupervised image reconstruction): please refer to [ImageReconstruction/ImageReconstruction.md](https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/ImageReconstruction/ImageReconstruction.md).

#### For windows system
For windows OS users, you have to change '/' to '\\\\' in 'line 71 of ImageAlignment/Codes/utils.py' and 'line 48 of ImageReconstruction/Codes/utils.py'.


## Experimental results on robustness
By resizing the input images to different resolutions, we simulation the change of feature quantity to compare ours with other methods in robustness. The results can be available in https://drive.google.com/drive/folders/1URFKTiUxaZ8i6pcHIKhxVTf-LkTNnXpK?usp=sharing.

Note: Since the RANSAC algorithm randomly selects the sample points, and the feature (SIFT) detection is not strictly consistent each time, different tests on the same image may differ. But the overall performance should be close to the results reported in our experiments.

## Meta
NIE Lang -- nielang@bjtu.edu.cn
```
@ARTICLE{9472883,
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  journal={IEEE Transactions on Image Processing}, 
  title={Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images}, 
  year={2021},
  volume={30},
  number={},
  pages={6184-6197},
  doi={10.1109/TIP.2021.3092828}}
```

## References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
[2] L. Nie, C. Lin, K. Liao, and Y. Zhao, “Learning edge-preserved image stitching from large-baseline deep homographyn,” arXiv preprint arXiv:2012.06194, 2020.  
[3] T. Nguyen, S. W. Chen, S. S. Shivakumar, C. J. Taylor, and V. Kumar. Unsupervised deep homography: A fast and robust homography estimation model. IEEE Robotics and Automation Letters, 3(3):2346–2353, 2018.  
[4] J. Zhang, C. Wang, S. Liu, L. Jia, N. Ye, J. Wang, J. Zhou, and J. Sun, “Content-aware unsupervised deep homography estimation,” in European Conference on Computer Vision, pp. 653–669, Springer, 2020.  
