# LaneSegmentation车道线检测

# week6
书籍：
不适合入门的书：
Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.；
Murphy, K. P. (2012). Machine Learning: a Probabilistic Perspective. MIT Press, Cambridge, MA, USA.；
花书http://www.deeplearningbook.org/；

适合入门的书：李沐的《动手学深度学习》；

## 部署
pytorch+flashk 轻量的，少用户的

Nginx+GUnicorn+flask;flask还需要gevent，因为flask的webserver性能一般

Docker;K8s可以自动化运维，管理很多Docker；

# week7笔记
## 怎么写Readme
1.写代码思路

2.写input、output是什么

3.week7研讨课？？

## 怎么学
1.多看课程代码，多跑代码，享受debug

2.独立思考，扔掉拐杖（课程代码）自己尝试独自完成

3.多看源码

## week7内容
需要学习得到的是参数，比如weight、bias；
人为决定的不需要学习的是超参数，比如epoch，优化器的learning rate、weight decay；
了解一下Hadoop，Spark，Yarn 


## label有错误
/root/data/LaneSeg/Image_Data/Road04/Record001/Camera 6/171206_054102961_Camera_6.jpg
/root/data/LaneSeg/Gray_Label/Label_road04/Label/Record001/Camera 6/171206_054102961_Camera_6_bin.png
