# Image Classification

## 更新日志
#### 1.0
2019.07.05
1. 根据 <https://blog.csdn.net/qq_16234613/article/details/79818370> 完成了一个简易的 VGG 网络
2. 在加载模型后添加了一个简易的模型状态输出
3. 配置文件使用了 yaml 格式和 yacs 库进行管理，放在 config 文件夹中统一管理
4. CNN 网络封装成了 class，所有模型模块在 model 文件夹中


#### 1.1
2019.07.05
1. 配置文件在各子模型中的传递方式由指针拷贝变为值拷贝；
2. 为 vgg 新建了分类模块，在 classifier.py 中；层数由2改为3；
3. vgg、backbone、classifier 加入了参数初始化；
4. 补全了 VGG13、VGG16、VGG19的配置文件

#### 2.0
2019.07.08
1. 添加了新的 backbone: ResNet-50、ResNet-101、ResNet-152；
2. 主模型名称从 VGG 改为 ImageClassifier；
3. 将卷积层的初始化方式改为 kaiming_init；

#### 2.1
2019.07.
1. 加入了 warmup optimizer 和 learning rate decay
2. 删除了模型中保存的无关参数
