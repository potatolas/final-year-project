file_string = ['''
import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def dodge(front,back):
    result=front*255/(255-back) 
    result[result>255]=255
    result[back==255]=255
    return result.astype('uint8')

def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img ="http://static.cricinfo.com/db/PICTURES/CMS/263600/263697.20.jpg"

s = imageio.imread(img)
g=grayscale(s)
i = 255-g

b = scipy.ndimage.filters.gaussian_filter(i,sigma=10)
r= dodge(b,g)

plt.imsave('img2.png', r, cmap='gray', vmin=0, vmax=255)
plt.imsave('img2.jpg', r, cmap='gray', vmin=0, vmax=255)
''',
'''
import boto3
import botocore
import hashlib
import imageio
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage
import time


def dodge(front,back):

    result=front*255/(255-back) 
    result[result>255]=255
    result[back==255]=255

    return result.astype('uint8')


def grayscale(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def hash_image(img_path):

    f = open(img_path,'rb')
    d = f.read()
    f.close()
    h = hashlib.sha256(d).hexdigest()

    return h

#
#  Main handler of lambda_function
#
def lambda_handler(event, context):
    #src_filename ="http://static.cricinfo.com/db/PICTURES/CMS/263600/263697.20.jpg"

    print("[DEBUG] event = {}".format(event))

    src_filename =event.get("name", None)
    #h = event.get("hash", None)
    sigma = event.get("sigma", 10)
    change_fullimage = event.get("sigma", False)

    filename_set = os.path.splitext(src_filename)
    basename = filename_set[0]
    ext = filename_set[1]
    h = basename.split("/")[1]

    down_filename='/tmp/my_image{}'.format(ext)
    conv_filename='/tmp/sketchify{}'.format(ext)
    down_jsonfile='/tmp/sketchify.json'

    if os.path.exists(down_filename):
        os.remove(down_filename)
    if os.path.exists(conv_filename):
        os.remove(conv_filename)

    #
    # s3 = boto3.resource('s3')
    #
    s3 = boto3.client('s3')
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    S3_KEY = src_filename

    try:
        # s3.Bucket(BUCKET_NAME).download_file(S3_KEY, down_filename)
        s3.download_file(BUCKET_NAME, S3_KEY, down_filename)        
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("===error message ===> {}".format(e))
            print("The object does not exist: s3://{}/{}".format(BUCKET_NAME, S3_KEY))
        else:
            raise

    #
    # Reading image to buffer.
    #
    s = imageio.imread(down_filename)
    #h = hash_image(down_filename)

    #
    # Split basename and extension from filename.
    #
    # filename_set = os.path.splitext(src_filename)
    # basename = filename_set[0]
    # ext = filename_set[1]
    sketchify_filename='public/{}/sketchify{}'.format(h, ext)
    sketchify_paramfile='public/{}/sketchify.json'.format(h) 
    #
    # Grayscale.
    #
    g = grayscale(s)
    i = 255 - g

    #
    # Grayscale.
    #
    b = scipy.ndimage.filters.gaussian_filter(i, sigma=sigma)
    r = dodge(b, g)

    #
    # Save the converted image to a local file.
    #
    plt.imsave(conv_filename, r, cmap='gray', vmin=0, vmax=255)

    #
    # s3 = boto3.client('s3')
    #
    s3.upload_file(conv_filename, BUCKET_NAME, sketchify_filename)

    #
    # Save params for sketchify whenever converting fullimage.
    #
    j = {'sigma': sigma}
    if change_fullimage != False:
        with open(down_jsonfile,'w') as f:
            f.write(json.dumps(j))
        s3.upload_file(down_jsonfile, BUCKET_NAME, sketchify_paramfile)


    images = {
        "source" : S3_URL.format(
            bucketName = BUCKET_NAME, 
            keyName = src_filename
        ),
        "params" : j,
        "dest" : DEST_S3_URL.format(
            bucketName = BUCKET_NAME, 
            keyName = sketchify_filename,
            timeStamp = time.time()
        )        
    }

    return {
        "statusCode": 200,
        "body": {"images": images }
    }
'''
]