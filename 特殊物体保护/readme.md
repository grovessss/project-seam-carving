特殊物体的检测和保护

（1）直线检测保护

​		参考论文：[Seam Carving with Improved Edge Preservation](https://www.researchgate.net/publication/228383984_Seam_Carving_with_Improved_Edge_Preservation)

​		手动选取直线：通过滑动条(Trackbar)手动调整阈值，根据想要保护的直线数量和分布，选取合适的阈值。

​		自动选取直线：根据图像中直线分布情况，自动选定阈值。但直线选取的效果没有手动方式好。

（2）人脸检测保护

​		真人人脸：使用opencv的人脸分类器haarcascade_frontalface_default.xml

​		动漫人脸：使用基于opencv的训练好的人脸分类器lbpcascade_animeface.xml

​							分类器GitHub链接：[A Face detector for anime/manga using OpenCV](https://github.com/nagadomi/lbpcascade_animeface)	

​							需要先将该分类器下载到本地，再修改代码中该分类器的路径位置。					