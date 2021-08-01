import tensorflow as tf

from p2 import get_group_conv


class ResNeXt_bottleNeck(tf.keras.layers.Layer):
    def __init__(self,filters,strides,groups):
        super(ResNeXt_bottleNeck,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=(1,1),
                                          strides=1,
                                          padding='same')
        self.bn1=tf.keras.layers.BatchNormalization()
        self.group_conv=get_group_conv(in_channels=filters,
                                    out_channels=filters,
                                    kernel_size=(3,3),
                                    strides=strides,
                                    padding='same',
                                    groups=groups)
        self.bn2=tf.keras.layers.BatchNormalization()
        self.conv2=tf.keras.layers.Conv2D(filters=2*filters,
                                          kernel_size=(1,1),
                                          strides=1,
                                          padding='same')
        self.bn13=tf.keras.layers.BatchNormalization()
        self.shortcut_conv=tf.keras.layers.Conv2D(filters=2*filters,
                                                  kernel_size=(1,1),
                                                  strides=1,
                                                  padding='same')
        self.shortcut_bn=tf.keras.layers.BatchNormalization()
        
        def call(self,inputs):
            x=self.cinv1(inputs)
            x=self.bn1(x)
            x=tf.nn.relu(x)
            x=self.group_conv(x)
            x=self.bn2(x)
            x=tf.nn.relu(x)
            x=self.conv2(x)
            x=self.bn3(x)
            x=tf.nn.relu(x)
            
            shortcut=self.shortcut_conv(inputs)
            shortcut=self.shortcut_bn(shortcut)
            
            output=tf.nn.relu(tf.keras.layers.add([x,shortcut]))
            return output
        
        
def build_ResNeXt_block(filters,strides,groups,repeat_num):
    block=tf.keras.Sequential()
    block.add(ResNeXt_bottleNeck(filters=filters,
                                 strides=strides,
                                 groups=groups))
    for i in range(1,repeat_num):
        block.add(ResNeXt_bottleNeck(filters=filters,
                                     strides=1,
                                     groups=groups))
    
    return block
