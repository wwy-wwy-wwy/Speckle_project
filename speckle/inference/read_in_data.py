from skimage import io
import matplotlib.pyplot as plt

def read_in_data(foldername,imgname):
    img = io.imread(foldername+imgname+'.tif')
    frame_number=img.shape[0]
    img_height=img.shape[1]
    img_width=img.shape[2]
    plt.imshow(img[1])
    return img