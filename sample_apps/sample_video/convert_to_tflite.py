import argparse
import tensorflow as tf

ap = argparse.ArgumentParser()

ap.add_argument(
        '-m',
        '--model_file',
        help='The only required argument is the h5 file that will be converted to the tflite format',
        required=True)

args = vars(ap.parse_args())

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(args['model_file'])
model = converter.convert()

file = open(args['model_file'] + '.tflite' , 'wb') 
file.write(model)