from sklearn.externals import joblib
from skimage.transform import resize
from skimage.io import imread
from skimage.filters import threshold_otsu

model = joblib.load('./svc.pkl')
image_path = './number_plates/26EE610E_66C1-078.83_01022018073256_i3/8.jpg'
img_details = imread(image_path, as_gray=True)
img_details = resize(img_details, (20, 20))

binary_image = img_details < threshold_otsu(img_details)

flat_bin_image = binary_image.reshape(-1)
reshaped_image = flat_bin_image.reshape(1, -1)
print(model.predict(reshaped_image))