import tensorflow as tf

class Conv2d(object) :
    def __init__(self,name,input_dim,output_dim,k_h=4,k_w=4,d_h=2,d_w=2,
                 stddev=0.02, data_format='NCHW',padding='SAME') :
        with tf.variable_scope(name) :
            assert(data_format == 'NCHW' or data_format == 'NHWC')
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            if( data_format == 'NCHW' ) :
                self.strides = [1, 1, d_h, d_w]
            else :
                self.strides = [1, d_h, d_w, 1]
            self.data_format = data_format
            self.padding = padding
    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b

        if( self.data_format =='NCHW' ) :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,
                                    use_cudnn_on_gpu=True,data_format='NCHW',
                                    strides=self.strides, padding=self.padding),
                        b,data_format='NCHW',name=name)
        else :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,data_format='NHWC',
                                    strides=self.strides, padding=self.padding),
                        b,data_format='NHWC',name=name)
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class Linear(object) :
    def __init__(self,name,input_dim,output_dim,stddev=0.02) :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w',[input_dim, output_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim],
                                initializer=tf.constant_initializer(0.0))

    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b

        if( input_var.shape.ndims > 2 ) :
            dims = tf.reduce_prod(tf.shape(input_var)[1:])
            return tf.matmul(tf.reshape(input_var,[-1,dims]),w) + b
        else :
            return tf.matmul(input_var,w)+b
    def get_variables(self):
        return {'w':self.w,'b':self.b}
