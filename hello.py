import PIL
from flask import Flask,request
# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from PIL import Image
import os
import cv2
from flask_cors import CORS, cross_origin
import numpy as np

import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import nibabel as nib
import pyrebase
from google.cloud import storage
from firebase import firebase
import os
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceAccountKey.json"
config = {
    "apiKey": "AIzaSyB3JnPch1dCceWPwcL0ytq0ZVPInu38YCE",
    "authDomain": "doctorsina-f6209.firebaseapp.com",
    "databaseURL": "https://doctorsina-f6209-default-rtdb.firebaseio.com",
    "projectId": "doctorsina-f6209",
    "storageBucket": "doctorsina-f6209.appspot.com",
    "serviceAccount": "serviceAccountKey.json"
}
firebase_storage=pyrebase.initialize_app(config)
storage = firebase_storage.storage()
'''
client = storage.Client()
bucket = client.get_bucket('doctorsina-f6209.appspot.com')
'''

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    #    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss


# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, 1])) + epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, 2])) + epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, 3])) + epsilon)


# Computing Precision
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# Computing Sensitivity
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


VOLUME_SLICES = 1
IMG_SIZE = 128
VOLUME_START_AT = 0

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}


def predictByPath(model, case_path):
    files = next(os.walk(case_path))[2]
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    #  y = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE))

    vol_path = os.path.join(case_path, f'BraTS20_Training_001_flair.nii');
    flair = nib.load(vol_path).get_fdata()

    vol_path = os.path.join(case_path, f'BraTS20_Training_001_t1ce.nii');
    ce = nib.load(vol_path).get_fdata()

    #   vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_seg.nii');
    #   seg=nib.load(vol_path).get_fdata()

    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    #       y[j,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))

    #  model.evaluate(x=X,y=y[:,:,:,0], callbacks= callbacks)
    return model.predict(X / np.max(X), verbose=1)

def TumourRatio(image,mask):
    brain_count=0
    tumour_count=0
    for j in range(VOLUME_SLICES):
        arr=(image[j,:,:]).flatten()
        brain_count+=np.count_nonzero(arr)
        background=(arr==0).sum()
        marray=(mask[j,:,:]).flatten()
        tumour_count+=np.count_nonzero(marray)
    return (tumour_count/brain_count)*100

def showPredictsById(model, userid,start_slice=60):
    print('hello')
    storage.child("flair_IRM/"+userid).download("flair.png")
    storage.child("ce_IRM/"+userid).download("ce.png")
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    path = f"./"

    vol_path = os.path.join(path, f'flair.png')
    flair = PIL.Image.open(vol_path).convert('L')
    im = cv2.resize(np.asarray(flair), (IMG_SIZE, IMG_SIZE))
    img = Image.fromarray((im * 255).astype(np.uint8))
    img.save("flair_gt.png")
    storage.child("segmentation/" + userid + "_flair_gt.png").put("flair_gt.png")
    flair_gt_url = storage.child("segmentation/" + userid + "_flair_gt.png").get_url(None)

    vol_path2 = os.path.join(path, f't1ce.png')
    ce = PIL.Image.open(vol_path2).convert('L')
    im2 = cv2.resize(np.asarray(ce), (IMG_SIZE, IMG_SIZE))
    img = Image.fromarray((im2 * 255).astype(np.uint8))
    img.save("ce_gt.png")
    storage.child("segmentation/" + userid + "_ce_gt.png").put("ce_gt.png")
    ce_gt_url = storage.child("segmentation/" + userid + "_ce_gt.png").get_url(None)

    X[0, :, :, 0] = cv2.resize(np.asarray(flair), (IMG_SIZE, IMG_SIZE))
    X[0, :, :, 1] = cv2.resize(np.asarray(ce), (IMG_SIZE, IMG_SIZE))
    p = model.predict(X / np.max(X), verbose=1)

    '''
    gt = nib.load(os.path.join(path, f'BraTS20_Training_001_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_001_flair.nii')).get_fdata()
    p = predictByPath(model,path)
    '''
    core = p[:, :, :, 1]
    core[core < 0.9] = 0
    edema = p[:, :, :, 2]
    edema[edema<0.4]=0
    enhancing = p[:, :, :, 3]
    enhancing[enhancing < 0.35] = 0
    coreRatio = "%.2f" %TumourRatio(X[:, :, :, 0], core)
    edemaRatio = "%.2f" %TumourRatio(X[:, :, :, 0], edema)
    enhancingRatio = "%.2f" %TumourRatio(X[:, :, :, 0], enhancing)
    wholeRatio= coreRatio+edemaRatio+enhancingRatio
    print("TC : ", "%.2f" % TumourRatio(X[:, :, :, 0], core), "%")
    print("ED : ", "%.2f" % TumourRatio(X[:, :, :, 0], edema), "%")
    print("ET : ", "%.2f" % TumourRatio(X[:, :, :, 0], enhancing), "%")
    print("whole tumour : ", "%.2f" % (TumourRatio(X[:, :, :, 0], enhancing) + TumourRatio(X[:, :, :, 0], core) + TumourRatio(X[:, :, :, 0],edema)), "%")
    im = Image.fromarray((p[0, :, :, 1:4] * 255).astype(np.uint8))

    background = Image.fromarray((X[0, :, :, 0] * 255).astype(np.uint8)).convert("RGBA")
    overlay = im.convert("RGBA")
    im = Image.blend(background,overlay,0.5)
    im.save("allClasses.png")


    im.save("allClasses.png")
    storage.child("segmentation/" + userid + "_allClasses.png").put("allClasses.png")
    allclasses_url=storage.child("segmentation/" + userid + "_allClasses.png").get_url(None)

    im = Image.fromarray((edema[0, :, :] * 255).astype(np.uint8))

    overlay = im.convert("RGBA")
    im = Image.blend(background, overlay, 0.3)
    im.save("edema.png")
    storage.child("segmentation/" + userid + "_edema.png").put("edema.png")
    edema_url = storage.child("segmentation/" + userid + "_edema.png").get_url(None)
    im = Image.fromarray((core[0, :, :] * 255).astype(np.uint8))

    overlay = im.convert("RGBA")
    im = Image.blend(background, overlay, 0.3)
    im.save("core.png")
    storage.child("segmentation/" + userid + "_core.png").put("core.png")
    core_url = storage.child("segmentation/" + userid + "_core.png").get_url(None)
    im = Image.fromarray((enhancing[0, :, :] * 255).astype(np.uint8))

    overlay = im.convert("RGBA")
    im = Image.blend(background, overlay, 0.3)
    im.save("enhancing.png")
    storage.child("segmentation/" + userid + "_enhancing.png").put("enhancing.png")
    enhancing_url = storage.child("segmentation/" + userid + "_enhancing.png").get_url(None)
    user_ref = db.collection(u'user').document(userid)
    user_ref.update({
        u'irm': {
            u'allUrl': allclasses_url,
            u'edemaUrl': edema_url,
            u'enhancingUrl': enhancing_url,
            u'coreUrl': core_url,
            u'ce_gtUrl': ce_gt_url,
            u'flair_gtUrl': flair_gt_url,
            u'coreRatio':coreRatio,
            u'edemaRatio':edemaRatio,
            u'enhancingRatio':enhancingRatio,
            u'wholeRatio':wholeRatio,

        },
    })


@app.route('/', methods=['POST'])
@cross_origin()
def hello_world():
    model = keras.models.load_model('./model_x1_2.h5',
                                    custom_objects={'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
                                                    "dice_coef": dice_coef,
                                                    "precision": precision,
                                                    "sensitivity": sensitivity,
                                                    "specificity": specificity,
                                                    "dice_coef_necrotic": dice_coef_necrotic,
                                                    "dice_coef_edema": dice_coef_edema,
                                                    "dice_coef_enhancing": dice_coef_enhancing}, compile=False)
    userid = request.json['userid']
    showPredictsById(model, userid)

    print(request.json['userid'])
    return request.data


if __name__ == '__main__':

    app.run(port=8000, debug=True)
