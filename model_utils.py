#-------------------------------------------
# import
#-------------------------------------------
import keras.backend as K


#-------------------------------------------
# public functions
#-------------------------------------------
def get_model_inputsize(model):

    input_layer = model.inputs[0]
    input_shape = K.int_shape(input_layer)
    _, input_h, input_w, input_ch = input_shape
    return (input_h, input_w)


def get_model(model_name):

    if model_name == 'VGG16':
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input, decode_predictions
    if model_name == 'VGG19':
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input, decode_predictions
        model = VGG19(weights='imagenet')
    elif model_name == 'ResNet50':
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(weights='imagenet')
    elif model_name == 'InceptionV3':
        from keras.applications.inception_v3 import preprocess_input, decode_predictions
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(weights='imagenet')
    elif model_name == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        model = InceptionResNetV2(weights='imagenet')
    elif model_name == 'Xception':
        from keras.applications.xception import preprocess_input, decode_predictions
        from keras.applications.xception import Xception
        model = Xception(weights='imagenet')
    elif model_name == 'MobileNet':
        from keras.applications.mobilenet import preprocess_input, decode_predictions
        from keras.applications.mobilenet import MobileNet
        model = MobileNet(weights='imagenet')
    elif model_name == 'MobileNetV2':
        from keras.applications.mobilenetv2 import preprocess_input, decode_predictions
        from keras.applications.mobilenetv2 import MobileNetV2
        model = MobileNetV2(weights='imagenet')
    elif model_name == 'DenseNet':
        from keras.applications.densenet import preprocess_input, decode_predictions
        from keras.applications.densenet import DenseNet121
        model = DenseNet121(weights='imagenet')
    elif model_name == 'NASNet':
        from keras.applications.nasnet import preprocess_input, decode_predictions
        from keras.applications.nasnet import NASNetMobile
        model = NASNetMobile(weights='imagenet')
    else:
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input, decode_predictions
        model = VGG16(weights='imagenet')

    return model, preprocess_input, decode_predictions


if __name__ == '__main__':
    pass
