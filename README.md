# Unsupervised Deep Image Stitching: Reconstructing Stitched Images from Feature to Pixel (paper)


## Abstract
Traditional feature-based image stitching technologies rely heavily on the quality of feature detection, often failing to stitch images with few features or at low resolution. The recent learning-based methods can only be supervisedly trained in a synthetic dataset instead of the real dataset, for the stitched labels under large parallax are difficult to obtain. In this paper, we propose the first unsupervised deep image stitching framework that can be achieved in two stages: unsupervised coarse image alignment and unsupervised image reconstruction. In the first stage, we design an ablation-based loss to constrain the unsupervised deep homography network, which is more suitable for large-baseline scenes than the existing constraints. Then, a transformer layer is implemented to align the input images in the stitching-domain space. In the next stage, we design an unsupervised image reconstruction network consist of low-resolution deformation branch and high-resolution deformation branch to learn the deformation rules of image stitching and enhance the resolution of stitched results at the same time, eliminating artifacts by reconstructing the stitched images from feature to pixel. Also, a comprehensive real dataset for unsupervised deep image stitching, which can be availabel at www.github.com/nie-lang/UnsupervisedDeepImageStitching, is proposed to evaluate our algorithms. Extensive experiments well demonstrate the superiority of our method over other state-ofthe-art solutions.

## Dataset for unsupervised deep image stitching
To train our network, we propose an unsupervised deep image stitching dataset that is obtained from variable moving videos. Of these videos, some are from [5] and the others are captured by ourselves. By extracting the frames from these videos with different interval time, we get the samples with different overlap rates. In particular, we get 10,440 cases for training and 1,106 cases for testing. The following figure illustrates somes cases of this real world dataset which includes variable scenes such as indoor, outdoor, night, dark, snow, zooming, similar texture, etc. Although our dataset contains no groundtruth, we include our testing results in this dataset, which can work as a benchmark dataset for other methods to compare.

This dataset is available in https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing.

## Experimental results on robustness


## References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
[2] L. Nie, C. Lin, K. Liao, and Y. Zhao, “Learning edge-preserved image stitching from large-baseline deep homographyn,” arXiv preprint arXiv:2012.06194, 2020.  
[3] D. DeTone, T. Malisiewicz, and A. Rabinovich. Deep image homography estimation. arXiv preprint arXiv:1606.03798, 2016.  
[4] T. Nguyen, S. W. Chen, S. S. Shivakumar, C. J. Taylor, and V. Kumar. Unsupervised deep homography: A fast and robust homography estimation model. IEEE Robotics and Automation Letters, 3(3):2346–2353, 2018.  
[5] J. Zhang, C. Wang, S. Liu, L. Jia, N. Ye, J. Wang, J. Zhou, and J. Sun, “Content-aware unsupervised deep homography estimation,” in European Conference on Computer Vision, pp. 653–669, Springer, 2020.  
