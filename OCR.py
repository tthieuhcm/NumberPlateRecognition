from sklearn.externals import joblib
from skimage.transform import resize
from skimage.io import imread
from skimage.filters import threshold_otsu
import os
from os import listdir
from os.path import isfile, join


def readFolderAndClassify(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    model = joblib.load('./svc.pkl')
    result = []
    for file in files:
        image_path = os.path.join(path, file)
        img_details = imread(image_path, as_grey=True)
        img_details = resize(img_details, (20, 20))
        binary_image = img_details < threshold_otsu(img_details)
        flat_bin_image = binary_image.reshape(-1)
        reshaped_image = flat_bin_image.reshape(1, -1)
        result.append(model.predict(reshaped_image)[0])
    return ''.join(result)
