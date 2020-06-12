Study of Seam Carving 
算法设计与分析Project
对seam carving算法的一些探究与改进

参考内容：
[1] Shai Avidan and Ariel Shamir, “Seam carving for contentaware image resizing,” ACM Transactions on Graphics (SIGGRAPH), vol. 26, no. 3, pp. 10, July 2007.
[2] Michael Rubinstein, Ariel Shamir, and Shai Avidan, “Improved seam carving for video retargeting,” ACM Transactions on Graphics (SIGGRAPH), vol. 27, 2008.
[3] Achanta, Radhakrishna, et al. Frequency-tuned salient region detection, in:CVPR 2009. IEEE Conference on Computer Vision and Pattern Recognition, 2009. IEEE, 2009.
[4] Achanta, Radhakrishna, Sabine Susstrunk. Saliency detection for contentaware image resizing, in: Sixteenth IEEE International Conference on Image Processing (ICIP), 2009. IEEE, 2009.
[5] Chen Y, Pan Y, Song M, et al. Improved seam carving combining with 3D saliency for image retargeting[J]. Neurocomputing, 2015, 151: 645-653.
[6] Johannes Kiess, Stephan Kopf, Benjamin Guthier, and Wolfgang Effelsberg. (2010). Seam Carving with Improved Edge Preservation. Proceedings of SPIE - The International Society for Optical Engineering. 7542. 10.1117/12.840263. 

运行环境：
Visual studio, opencv库

环境配置：
参考链接 https://www.cnblogs.com/YiYA-blog/p/10296224.html

主要内容：

1、 实现、测试并比较了梯度、熵、显著性、深度图等能量函数。

2、 增加了对特殊物体（直线和人脸）的检测和保护，减少了在大幅度裁剪的情况下直线或人脸被扭曲的概率。

3、 利用不同的实现方式实现seam carving，并比较实现的效果和时间效率。

(具体操作方式见各文件夹内说明)
