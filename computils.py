import numpy as np
from vizutils import *
from multiprocessing import Pool
from skimage.transform import resize
from skimage.color import label2rgb
from skimage.util import view_as_windows, view_as_blocks, apply_parallel
from tqdm.notebook import tqdm 


def Calc(images, lambda_func):
    results = []
    for img in images:
        result = lambda_func(img)
        results.append(result)
    return results



def CalcFast(images, lambda_func):
    with Pool() as p:
        results = p.map(lambda_func, images)
    return results

def SortByValue(images, values):
    zipped = zip(images, values)
    sorted_zipped = sorted(zipped, key=lambda x: x[1])
    images, values = zip(*sorted_zipped)
    return images, values

def SortByValueWithHistogram(images, values, histogram):
    zipped = zip(images, values,histogram)
    sorted_zipped = sorted(zipped, key=lambda x: x[1])
    images, values,histogram = zip(*sorted_zipped)
    return images, values, histogram


def SortAndShow(images, values, amount):
    images, values = SortByValue(images, values)
    ImShowDistribution(images, values, amount)

def SortAndShowWithHistograms(images, values, amount, histograms):
    images, values, histograms = SortByValueWithHistogram(images, values, histograms)
    stride = len(images) // amount
    ImShowArrayWithHistograms(images[::stride], histograms[::stride])


def StichImages(im1, im2, im3, im4):
    top = np.concatenate((im1, im2), axis=1)
    bottom = np.concatenate((im3, im4), axis=1)
    return np.concatenate((top, bottom), axis=0)


def SegmentationExample(images, lambda_func, size, normalize = True):
    test_imgs = images
    test_imgs = [np.uint8(resize(img, (size,size))*255) for img in test_imgs]
    test_img = StichImages(test_imgs[0],test_imgs[1],test_imgs[2],test_imgs[3])
    ImShow(test_img)

    class_features = Calc(test_imgs, lambda_func)
    fe = []
    for f in class_features:
        st = ""
        for k,v in f.items():
            st += "\n{}: {:.3f}".format(k,v)
        fe.append(st)
    ImShowArray(test_imgs,fe)
    class_features = [np.array(list(f.values())) for f in class_features]
    class_features = np.array(class_features)

    windows = view_as_windows(test_img,(16,16),step=1)
    seg = np.zeros((windows.shape[0],windows.shape[1]))
    local_features = []
    for i in tqdm(range(0,windows.shape[0])):
        local_features.append([])
        for j in range(0,windows.shape[1]):
            im = np.copy(windows[i,j])
            features = np.array(list(lambda_func(windows[i,j]).values()))
        
            local_features[i].append(features)
            #res = [np.linalg.norm() for f in class_features_scaled]

        #   seg[i,j] = np.argmin(res)

    local_features = np.array(local_features)   

    max_vals =  np.zeros(local_features.shape[2])
    scaled_local_features = np.copy(local_features)
    for i in range(0,local_features.shape[2]):
        max_vals[i] = np.max(local_features[:,:,i])
        scaled_local_features[:,:,i] = scaled_local_features[:,:,i] /  max_vals[i]
    scaled_class_features = class_features / max_vals
    ImShowArray([scaled_local_features[:,:,i] for i in range(0,local_features.shape[2])])

    seg = np.zeros((scaled_local_features.shape[0],scaled_local_features.shape[1]))
    for x in range(0,scaled_local_features.shape[0]):
        for y in range(0,scaled_local_features.shape[1]):
            res = [np.linalg.norm(scaled_local_features[x,y] - f) for f in scaled_class_features]
            seg[x,y] = np.argmin(res)
    print(seg)     

    background =resize(np.copy(test_img), (seg.shape[0],seg.shape[1]))
    plt.imshow(test_img, cmap='gray')
    plt.imshow(label2rgb(seg+1),alpha=0.4)
    plt.show()



def SegmentationExampleVectorDescriptor(images, lambda_func, size, normalize = True, random_patches = False, window_size = 16, step_size=1):
    test_imgs = images
    test_imgs = [np.uint8(resize(img, (size,size))*255) for img in test_imgs]
    test_img = StichImages(test_imgs[0],test_imgs[1],test_imgs[2],test_imgs[3])
    ImShow(test_img)


    if not random_patches:
        class_features = Calc(test_imgs, lambda_func)
        class_features = np.array(class_features)
    else:
        class_features = []
        for c in range(len(test_imgs)):
            s = size // 2
            r1 = lambda_func(test_imgs[c][s:s+window_size,s:s+window_size])
            s = size // 4
            r2 = lambda_func(test_imgs[c][s:s+window_size,s:s+window_size])
            s = (size // 4)*3
            r3 = lambda_func(test_imgs[c][s:s+window_size,s:s+window_size])
            class_features.append( (r1 + r2 + r3) / 3)


    windows = view_as_windows(test_img,(window_size,window_size),step=step_size)
    seg = np.zeros((windows.shape[0],windows.shape[1]))
    local_features = []
    for i in tqdm(range(0,windows.shape[0])):
        local_features.append([])
        for j in range(0,windows.shape[1]):
            im = np.copy(windows[i,j])
            features = np.array(lambda_func(windows[i,j]))
        
            local_features[i].append(features)
            #res = [np.linalg.norm() for f in class_features_scaled]

        #   seg[i,j] = np.argmin(res)

    local_features = np.array(local_features)   

    if normalize:
        max_vals =  np.zeros(local_features.shape[2])
        scaled_local_features = np.copy(local_features)
        for i in range(0,local_features.shape[2]):
            max_vals[i] = np.max(local_features[:,:,i])
            scaled_local_features[:,:,i] = scaled_local_features[:,:,i] /  max_vals[i]
        scaled_class_features = class_features / max_vals
    else:
        scaled_local_features = local_features
        scaled_class_features = class_features

    seg = np.zeros((scaled_local_features.shape[0],scaled_local_features.shape[1]))
    for x in range(0,scaled_local_features.shape[0]):
        for y in range(0,scaled_local_features.shape[1]):
            res = [np.linalg.norm(scaled_local_features[x,y] - f) for f in scaled_class_features]
            seg[x,y] = np.argmin(res)
    print(seg)     

    seg_resized =resize(label2rgb(seg+1), (test_img.shape[0],test_img.shape[1]))
    
    plt.imshow(test_img, cmap='gray')
    plt.imshow(seg_resized,alpha=0.4)
    plt.show()
   
   
