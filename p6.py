import tensorflow as tf
from p7 import build_ResNeXt_block
from p1 import NUM_CLASSES

class ResNext(tf.keras.Model):
    def __init__(self,repeat_num_list,cardinality):
        if len(repeat_num_list)!=4:
            raise ValueError("bnv")
        super(ResNext,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=64,
                                          kernel_size=(7,7),
                                          strides=2,
                                          padding='same')
        
        
        self.bn1=tf.keras.layers.BatchNormalization()
        
        
        self.pool1=tf.keras.layers.MaxPool2D(pool_size=(3,3),
                                             strides=2,
                                             padding='same')
        
        
        self.block1=build_ResNeXt_block(filters=128,
                                        strides=1,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[0])
        
        
        self.block2=build_ResNeXt_block(filters=256,
                                        strides=2,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[1])
        
        
        self.block3=build_ResNeXt_block(filters=512,
                                        strides=2,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[2])
        
        
        
        self.block4=build_ResNeXt_block(filters=1024,
                                        strides=2,
                                        groups=cardinality,
                                        repeat_num=repeat_num_list[3])
        
        
        self.pool2=tf.keras.layers.GlobalAveragePooling2D()
        
        
        self.fc=tf.keras.layers.Dense(128,
                                      activation=tf.keras.activations.softmax)
    
    def call(self,inputs):
        x=self.conv1(inputs)
        x=self.bn1(x)
        x=tf.nn.relu(x)
        x=self.pool1(x)
        
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        
        x=self.pool2(x)
        x=self.fc(x)
        return x
    
def ResNext50(): 
    return ResNext(repeat_num_list=[3,4,6,3],cardinality=32)