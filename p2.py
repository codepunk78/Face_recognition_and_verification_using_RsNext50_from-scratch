import tensorflow as tf
from tensorflow.keras import initializers,regularizers,constraints
from tensorflow.keras import activations
from p1 import DEVICE

def get_group_conv(in_channels,
                   out_channels,
                   kernel_size,
                   strides=(1, 1),
                   padding='valid',
                   groups=1):
    if DEVICE == "cpu":
        return GroupConv2D(input_channels=in_channels,
                           output_channels=out_channels,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           groups=groups)
    elif DEVICE == "gpu":
        return tf.keras.layers.Conv2D(filters=out_channels,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      groups=groups)
    else:
        raise ValueError("Attribute 'DEVICE' must be 'cpu' or 'gpu'.")



class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=(1,1),
                padding='valid',
                activation=None,
                groups=1
                ):
        super(GroupConv2D,self).__init__()
        if in_channels%groups != 0:
            raise ValueError("xyz")
        if out_channels%groups !=0:
            raise ValueError("ABC")
            
        self.kernel_size=kernel_size
        self.stride=strides
        self.padding=padding
        self.groups=groups
            
        self.group_in_num=in_channels
        self.group_out_num=out_channels
            
        self.conv_List=[]
        for i in range(self.groups):
            self.conv_List.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                        kernel_size=kernel_size,
                                                        strides=strides,
                                                        padding=padding,
                                                        activation=activations.get(activation),
                                                        ))
    def call(self,inputs):
            features_map_list=[]
            for i in range(self.groups):
                x_i=self.conv_list[i](inputs[:,:,:,i*self.group_in_num:(i+1)*self.group_in_num])
                features_map_list.append(x_i)
            out=tf.concat(features_map_list,axis=-1)
            return out
        
        
        
        


