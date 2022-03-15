file_string = ['''

import imageio
import numpy as np
import matplotlib.pyplot as plt

pic = imageio.imread('F:/demo_2.jpg')

gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
gray = gray(pic)  

plt.figure( figsize = (10,10))
plt.imshow(gray, cmap = plt.get_cmap(name = 'gray'))
plt.show()


pic = imageio.imread('F:/demo_2.jpg')

gray = lambda rgb : np.dot(rgb[... , :3] , [0.21 , 0.72, 0.07]) 
gray = gray(pic)  

plt.figure( figsize = (10,10))
plt.imshow(gray, cmap = plt.get_cmap(name = 'gray'))
plt.show()

print('Type of the image : ' , type(gray))
print()
print('Shape of the image : {}'.format(gray.shape))
print('Image Hight {}'.format(gray.shape[0]))
print('Image Width {}'.format(gray.shape[1]))
print('Dimension of Image {}'.format(gray.ndim))
print()
print('Image size {}'.format(gray.size))
print('Maximum RGB value in this image {}'.format(gray.max()))
print('Minimum RGB value in this image {}'.format(gray.min()))
print('Random indexes [X,Y] : {}'.format(gray[100, 50]))


''',
'''

import imageio
import numpy as np
import matplotlib.pyplot as plt

pic = imageio.imread('<image location>')
gamma = 2.2
original = ((pic/255) ** (1/gamma))

plt.imshow(original)
''',
'''

import imageio
import numpy as np
import matplotlib.pyplot as plt

pic = imageio.imread('<image location>')
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
gray = gray(pic)

max_ = np.max(gray)

def log_transform():
    return 255/np.log(1+max_) * np.log(1+gray)

plt.figure(figsize = (5,5))
plt.imshow(log_transform()[:,300:1500], cmap = plt.get_cmap(name = 'gray'))

''',
'''
import imageio
import matplotlib.pyplot as plt

pic = imageio.imread('<image location>')
plt.figure(figsize = (6,6))
plt.imshow(255 - pic);
''',
'''
# importing necessary packages
from skimage import color
from skimage import exposure
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# load the image
pic = imageio.imread('<image location>')
plt.figure(figsize = (6,6))
plt.imshow(pic);

# Convert the image to grayscale
img = color.rgb2gray(pic)

# outline kernel - used for edge detection
kernel = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])

# we use 'valid' which means we do not add zero padding to our image
edges = convolve2d(img, kernel, mode = 'valid')


# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)),
                                              clip_limit = 0.03)

# plot the edges_clipped
plt.imshow(edges_equalized, cmap='gray')
plt.axis('off')
plt.show()
''',
'''
# importing necessary packages
from skimage import color
from skimage import exposure
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Convert the image to grayscale
pic = imageio.imread('<image location>')
img = color.rgb2gray(pic)

# gaussian kernel - used for blurring
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])
kernel = kernel / np.sum(kernel)

# we use 'valid' which means we do not add zero padding to our image
edges = convolve2d(img, kernel, mode = 'valid')


# Adjust the contrast of the filtered image by applying Histogram Equalization
edges_equalized = exposure.equalize_adapthist(edges/np.max(np.abs(edges)),
                                              clip_limit = 0.03)

# plot the edges_clipped
plt.imshow(edges_equalized, cmap='gray')
plt.axis('off')
plt.show()
''',
'''
# importing necessary packages
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# load the image
pic = imageio.imread('<image location>')
plt.figure(figsize = (6,6))
plt.imshow(pic);

# convolution function
def Convolution(image, kernel):
    conv_bucket = []
    for d in range(image.ndim):
        conv_channel = convolve2d(image[:,:,d], kernel,
                               mode="same", boundary="symm")
        conv_bucket.append(conv_channel)
    return np.stack(conv_bucket, axis=2).astype("uint8")

# different size of kernel
kernel_sizes = [9,15,30,60]
fig, axs = plt.subplots(nrows = 1, ncols = len(kernel_sizes), figsize=(15,15))

# iterate through all the kernel and convoluted image
for k, ax in zip(kernel_sizes, axs):
    kernel = np.ones((k,k))
    kernel /= np.sum(kernel)
    ax.imshow(Convolution(pic, kernel))
    ax.set_title("Convolved By Kernel: {}".format(k))
    ax.set_axis_off()
''',
'''

# importing necessary packages
from skimage import color
from skimage import exposure
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# right sobel
sobel_x = np.c_[
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

# top sobel
sobel_y = np.c_[
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
]

ims = []
for i in range(3):
    sx = convolve2d(pic[:,:,i], sobel_x, mode="same", boundary="symm")
    sy = convolve2d(pic[:,:,i], sobel_y, mode="same", boundary="symm")
    ims.append(np.sqrt(sx*sx + sy*sy))

img_conv = np.stack(ims, axis=2).astype("uint8")

plt.figure(figsize = (6,5))
plt.axis('off')
plt.imshow(img_conv)
''',
'''

# importing necessary packages
from skimage import color
from skimage import exposure
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import median_filter

def median_filter_(img, mask):
    """
    Applies a median filer to all channels
    """
    ims = []
    for d in range(3):
        img_conv_d = median_filter(img[:,:,d], size=(mask,mask))
        ims.append(img_conv_d)

    return np.stack(ims, axis=2).astype("uint8")

pic = imageio.imread('<image location>')
filtered_img = median_filter_(pic, 80)

# right sobel
sobel_x = np.c_[
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

# top sobel
sobel_y = np.c_[
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
]

ims = []
for d in range(3):
    sx = convolve2d(filtered_img[:,:,d], sobel_x, mode="same", boundary="symm")
    sy = convolve2d(filtered_img[:,:,d], sobel_y, mode="same", boundary="symm")
    ims.append(np.sqrt(sx*sx + sy*sy))

img_conv = np.stack(ims, axis=2).astype("uint8")

plt.figure(figsize=(7,7))
plt.axis('off')
plt.imshow(img_conv)
''',
'''
#importing necessary packages
from skimage import color
from skimage import exposure
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

pic = imageio.imread('<image location>')
# Convert the image to grayscale
img = color.rgb2gray(img)

# apply sharpen filter to the original image
sharpen_kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
image_sharpen = convolve2d(img, sharpen_kernel, mode = 'valid')

# apply edge kernel to the output of the sharpen kernel
edge_kernel = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
edges = convolve2d(image_sharpen, edge_kernel, mode = 'valid')

# apply normalize box blur filter to the edge detection filtered image
blur_kernel = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]])/9.0;
denoised = convolve2d(edges, blur_kernel, mode = 'valid')

# Adjust the contrast of the filtered image by applying Histogram Equalization
denoised_equalized = exposure.equalize_adapthist(denoised/np.max(np.abs(denoised)),
                                                 clip_limit=0.03)
plt.imshow(denoised_equalized, cmap='gray')    # plot the denoised_clipped
plt.axis('off')
plt.show()
''',
''' 

import imageio
import numpy as np
import matplotlib.pyplot as plt

pic = imageio.imread('F:/demo_1.jpg')
plt.figure(figsize = (10,10))
plt.imshow(pic)
plt.show()


low_pixel = pic < 20

# to ensure of it let's check if all values in low_pixel are True or not
if low_pixel.any() == True:
    print(low_pixel.shape)


print(pic.shape)
print(low_pixel.shape)  

# randomly choose a value 
import random

# load the orginal image
pic = imageio.imread('F:/demo_1.jpg')

# set value randomly range from 25 to 225 - these value also randomly choosen
pic[low_pixel] = random.randint(25,225)

# display the image
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()
''',
'''

import imageio
import numpy as np
import matplotlib.pyplot as plt
    
# load the image
pic = imageio.imread('F:/demo_1.jpg')

# seperate the row and column values
total_row , total_col , layers = pic.shape

x , y = np.ogrid[:total_row , :total_col]

# get the center values of the image
cen_x , cen_y = total_row/2 , total_col/2

distance_from_the_center = np.sqrt((x-cen_x)**2 + (y-cen_y)**2)

# Select convenient radius value
radius = (total_row/2)

# Using logical operator '>' 

circular_pic = distance_from_the_center > radius

pic[circular_pic] = 0
plt.figure(figsize = (10,10))
plt.imshow(pic) 
plt.show()
''',

'''
import imageio
import matplotlib.pyplot as plt

pic = imageio.imread('F:/demo_2.jpg')
plt.figure(figsize = (15,15))

plt.imshow(pic)

print('Type of the image : ' , type(pic))
print()
print('Shape of the image : {}'.format(pic.shape))
print('Image Hight {}'.format(pic.shape[0]))
print('Image Width {}'.format(pic.shape[1]))
print('Dimension of Image {}'.format(pic.ndim))

print('Image size {}'.format(pic.size))

print('Maximum RGB value in this image {}'.format(pic.max()))
print('Minimum RGB value in this image {}'.format(pic.min()))

pic[ 100, 50 ]

# A specific pixel located at Row : 100 ; Column : 50 
# Each channel's value of it, gradually R , G , B
print('Value of only R channel {}'.format(pic[ 100, 50, 0]))
print('Value of only G channel {}'.format(pic[ 100, 50, 1]))
print('Value of only B channel {}'.format(pic[ 100, 50, 2]))

# let's take a quick view of each channels in the whole image.

plt.title('R channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))

plt.imshow(pic[ : , : , 0])
plt.show()

plt.title('G channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))

plt.imshow(pic[ : , : , 1])
plt.show()

plt.title('B channel')
plt.ylabel('Height {}'.format(pic.shape[0]))
plt.xlabel('Width {}'.format(pic.shape[1]))

plt.imshow(pic[ : , : , 2])
plt.show()



# ------------------------------------------------------

pic = imageio.imread('F:/demo_2.jpg')

pic[50:150 , : , 0] = 255 # full intensity to those pixel's R channel
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()


pic[200:300 , : , 1] = 255 # full intensity to those pixel's G channel
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()

pic[350:450 , : , 2] = 255 # full intensity to those pixel's B channel
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()

# To make it more clear let's change the column section and this time 
# we'll change the RGB channel simultaneously

# set value 200 of all channels to those pixels which turns them to white
pic[ 50:450 , 400:600 , [0,1,2] ] = 200 
plt.figure( figsize = (10,10))
plt.imshow(pic)
plt.show()
''',
'''

import imageio
import numpy as np
import matplotlib.pyplot as plt

pic = imageio.imread('F:\satimg.jpg')
plt.figure(figsize = (10,10))
plt.imshow(pic)
plt.show()

print(f'Shape of the image {pic.shape}')
print(f'hieght {pic.shape[0]} pixels')
print(f'width {pic.shape[1]} pixels')

# Detecting High Pixel of Each Channel

# Only Red Pixel value , higher than 180
pic = imageio.imread('F:\satimg.jpg')
red_mask = pic[:, :, 0] < 180

pic[red_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(pic)


# Only Green Pixel value , higher than 180
pic = imageio.imread('F:\satimg.jpg')
green_mask = pic[:, :, 1] < 180

pic[green_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(pic)


# Only Blue Pixel value , higher than 180
pic = imageio.imread('F:\satimg.jpg')
blue_mask = pic[:, :, 2] < 180

pic[blue_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(pic)

# Composite mask using logical_and
pic = imageio.imread('F:\satimg.jpg')
final_mask = np.logical_and(red_mask, green_mask, blue_mask)
pic[final_mask] = 40
plt.figure(figsize=(15,15))
plt.imshow(pic)
''',
'''

import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def hough_line(img, angle_step = 1, white_lines = True, threshold = 5):

    """
    param:: img - 2D binary image
    param:: angle_step - Spacing between angles to use every n-th angle, Default step is 1.
    param:: lines_are_white - boolean indicator
    param:: value_threshold - Pixel values above or below the threshold are edges

    Returns:
    param:: accumulator - 2D array of the hough transform accumulator
    param:: theta - array of angles used in computation, in radians.
    param:: rhos - array of rho values.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > threshold if white_lines else img < threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def viz_hough_line(img, accumulator, thetas, rhos, save_path=None):

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input Image')
    ax[0].axis('image')

    ax[1].imshow( accumulator, cmap='jet', extent=[np.rad2deg(thetas[-1]),
                                                   np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough Transform')
    ax[1].set_xlabel('Angles (deg)')
    ax[1].set_ylabel('Distance (px)')
    ax[1].axis('image')

    plt.axis('off')
    plt.show()


gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])

if __name__ == '__main__':
    # import and function calling
    pic = imageio.imread('<image location>')
    gray = gray(pic)

    accumulator, thetas, rhos = hough_line(gray) # get the parameter
    viz_hough_line(gray, accumulator, thetas, rhos) # visualization

''',

'''

import numpy as np
import imageio
import matplotlib.pyplot as plt

pic = imageio.imread('<image location>')
plt.imshow(pic);

def otsu_threshold(im):

    # Compute histogram and probabilities of each intensity level
    pixel_counts = [np.sum(im == i) for i in range(256)]

    # Initialization
    s_max = (0,0)

    for threshold in range(256):

        # update
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])

        mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # calculate - inter class variance
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2

        if s > s_max[1]:
            s_max = (threshold, s)


    return s_max[0]


def threshold(pic, threshold):
    return ((pic > threshold) * 255).astype('uint8')

gray = lambda rgb : np.dot(rgb[... , :3] , [0.21 , 0.72, 0.07])
plt.imshow(threshold(gray(pic), otsu_threshold(pic)), cmap='Greys')
plt.axis('off')
''',

'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import imageio
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

pic = imageio.imread('<image location>')

h,w = pic.shape[:2]

im_small_long = pic.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h,w,3))

km = KMeans(n_clusters=2)
km.fit(im_small_long)

seg = np.asarray([(1 if i == 1 else 0)
                  for i in km.labels_]).reshape((h,w))

contours = measure.find_contours(seg, 0.5, fully_connected="high")
simplified_contours = [measure.approximate_polygon(c, tolerance=5)
                       for c in contours]

plt.figure(figsize=(5,10))
for n, contour in enumerate(simplified_contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)


plt.ylim(h,0)
plt.axes().set_aspect('equal')
''']
