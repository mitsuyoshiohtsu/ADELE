import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

#MAX_ITER = 10

#POS_W = 3
#POS_XY_STD = 1
#Bi_W = 4
#Bi_XY_STD = 67
#Bi_RGB_STD = 3

#default
#POS_W = 3
#POS_XY_STD = 3
#Bi_W = 10
#Bi_XY_STD = 80
#Bi_RGB_STD = 13

#refer source-code
#MAX_ITER = 5
#POS_W = 3
#POS_XY_STD = 20
#Bi_W = 10
#Bi_XY_STD = 30
#Bi_RGB_STD = 20

#tuning
MAX_ITER = 5
POS_W = 3
POS_XY_STD = 3
Bi_W = 4
Bi_XY_STD = 10
Bi_RGB_STD = 30

def crf(img, output_probs, use_2d=True):
    
    img = np.uint8(255 * img).transpose(2, 1, 0)
    
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q