import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
from scipy.special import binom

class Bezier(tf.keras.Model):
    def __init__(self, num_bends: int):
        super().__init__()
        self.binom = tf.Variable(tf.constant(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32)), trainable= False)
        self.range = tf.Variable(tf.range(0, float(num_bends)), trainable= False)
        self.rev_range = tf.Variable(tf.range(float(num_bends - 1), -1, delta= -1), trainable= False)

        # Not sure if this is the best way to substitute register_buffer() in PyTorch
        # The PyTorch Buffer in this example is not considered a model parameter, not trained, 
        # part of the module's state, moved to cuda() or cpu() with the rest of the model's parameters

    def call(self, t: float):
        return self.binom * \
            tf.math.pow(t, self.range) * \
            tf.math.pow((1.0 - t), self.rev_range)




class Linear(CurveModule):
    def __init__(self, in_features, out_features, fix_points, bias = True):
        super(Linear, self).__init__(fix_points)
        self.fix_points = fix_points
        self.in_features = in_features
        self.out_features = out_features
        self.l2 = 0.0
        self.weights = []
        self.bs = []
        for i, fixed in enumerate(self.fix_points):
            self.weights.append(self.add_weight(
                shape = (self.in_features, self.out_features),
                initializer = "random_normal",
                trainable = True
                ))
        for i, fixed in range(self.fix_points):
            if bias: 
                self.bs.append(self.add_weight(
                    shape = (self.out_features,), initializer = "zeros", trainable = True
                    ))
            else:
                self.bs.append(None)
    
 ## definiton of parameter reset: very unsure if this is correct

    def reset_parameters(self):
        for i in range(self.num_bends):
            session = K.get_session()
            for layer in self.layers: 
                if hasattr(layer, 'kernel_initializer'):
                     layer.kernel.initializer.run(session=session)
   

    def call(self, inputs):
       for point in range(self.num_bends):
            return tf.matmul(inputs, self.weights[point + 1]) + self.bs[point + 1]
        ## not sure if it is necessary to iterate over all fix_points again for doing the forward 
        ## pass or if simply:
        ## return tf.matmul(inputs, self.weights) + self.bs
        ## would be enough.
        
                  



#class CurveNet(tf.keras.Model):
#    def __init__(self, num_classes, curve, architecture, num_bends, 
#                 fix_start = True, fix_end = True, architecture_kwargs={}):
#        super().__init__() 
#        self.num_classes = num_classes
#        self.num_bends = num_bends
#        self.fix_points = [fix_start] + [False]*(self.num_bends - 2) + [fix_end]
#        
#        self.curve = curve
#        self.architecture = architecture#
#
#        self.l2 = 0.0
#        self.coeff_layer = self.curve(self.num_bends)
#        self.net = self.architecture(num_classes, fix_points = self.fix_points, **architecture_kwargs)
#        self.curve_modules = []
#        for module in self.net.modules():