from model import *
# from evaluation import *
from preprocess import *
from keras.utils import normalize

transform_data(data_dir='data/all/', save_data=True)

if os.path.isfile('data/X.npy') and os.path.isfile('data/y.npy'):
    X = numpy.load('data/X.npy')
    y = numpy.load('data/y.npy')
else:
    X, y = transform_data(data_dir='data/all/')
    # numpy.save('data/X.npy', X)
    # numpy.save('data/y.npy', y)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train, X_test = X_train / 255, X_test / 255

save_dir = 'weights/unet_simple_gray/'
unet = UNet(X=X, y=y, input_size=(256, 256, 1), simple=True)
unet.train(X=X_train, y=y_train, save_dir='weights/unet_simple_gray/unet.h5')

y_pred = unet.test(X=X_test)

numpy.save(save_dir + 'y_test.npy', y_test)
numpy.save(save_dir + 'y_pred.npy', y_pred)
numpy.save(save_dir + 'X_test.npy', X_test)

# print(pixel_wise_eval(y_true=unet.y_test, y_pred=y_pred))
