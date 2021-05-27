# Unsupervised Deep Image Stitching: Reconstructing Stitched Images from Feature to Pixel (paper)
<p align="center">Lang Nie*, Chunyu Lin*, Kang Liao*, Shuaicheng Liu`, Yao Zhao*</p>
<p align="center">* Institute of Information Science, Beijing Jiaotong University</p>
<p align="center">` School of Information and Communication Engineering, University of Electronic Science and Technology of China</p>


## Dataset for unsupervised deep image stitching
We also propose an unsupervised deep image stitching dataset that is obtained from variable moving videos. Of these videos, some are from [4] and the others are captured by ourselves. By extracting the frames from these videos with different interval time, we get the samples with different overlap rates. Moreover, these videos are not shot by the camera rotating around the optical center, and the shot scenes are far from the same depth plane, which means this dataset contains different degrees of parallax. Besides, this real-world dataset includes variable scenes such as indoor, outdoor, night, dark, snow, zooming, etc. In particular, we get 10,440 cases for training and 1,106 cases for testing. Although our dataset contains no ground-truth, we include our testing results in this dataset, which we hope can work as a benchmark dataset for other researchers to follow and compare.

![image](https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/figures/dataset1.jpg)
![image](https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/figures/dataset2.jpg)
![image](https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/figures/dataset3.jpg)

We release the testing set and our results here https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing.
The training dataset will be public when our paper is accept.

## Experimental results on robustness
By resizing the input images to different resolutions, we simulation the change of feature quantity to compare ours with other methods in robustness.

![image](https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/figures/robust.jpg)

The results can be available in https://drive.google.com/drive/folders/1URFKTiUxaZ8i6pcHIKhxVTf-LkTNnXpK?usp=sharing.

Note: Since the RANSAC algorithm randomly selects the sample points, and the feature (SIFT) detection is not strictly consistent each time, different tests on the same image may differ. But the overall performance should be close to the results reported in our experiments.

## Meta
NIE Lang -- nielang@bjtu.edu.cn


## References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
[2] L. Nie, C. Lin, K. Liao, and Y. Zhao, “Learning edge-preserved image stitching from large-baseline deep homographyn,” arXiv preprint arXiv:2012.06194, 2020.  
[3] T. Nguyen, S. W. Chen, S. S. Shivakumar, C. J. Taylor, and V. Kumar. Unsupervised deep homography: A fast and robust homography estimation model. IEEE Robotics and Automation Letters, 3(3):2346–2353, 2018.  
[4] J. Zhang, C. Wang, S. Liu, L. Jia, N. Ye, J. Wang, J. Zhou, and J. Sun, “Content-aware unsupervised deep homography estimation,” in European Conference on Computer Vision, pp. 653–669, Springer, 2020.  
