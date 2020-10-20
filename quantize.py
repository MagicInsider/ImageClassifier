import os
import sys
import tensorflow as tf
from models import ResNet50


def root_path():
    return os.path.dirname(sys.argv[0])


def quantize_model(model_, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()
    quantized_path = path + '.optimized'
    open(quantized_path, "wb").write(quantized_model)
    print(f'\nQuantized model saved as {quantized_path}\n')
    return

if __name__ == '__main__':
    model_name = 'ResNet50_256x256_Wayne_Reilly'
    weights_name = 'weights.cat_acc_best.hdf5'
    input_shape = (256, 256, 3)
    classes = 24
    
    model = ResNet50(input_shape=input_shape, classes=classes, name='ResNet50')
 
    model.load_weights(os.path.join(root_path(), 'models', model_name, weights_name))
     
    quantize_model(model, os.path.join(root_path(), 'models', model_name))

    sys.exit(0)





