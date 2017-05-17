from sklearn.cluster import DBSCAN
import numpy as np
import cv2
CENTER_CROP = (32, 32, 192, 192)

def simple_box_from_mask(mask, threshold, min_members=300, return_center_crop_on_failure=None):
    k = np.asarray(zip(*(mask > threshold).nonzero()))
    if len(k)<min_members:
        print 'Bad'
        return CENTER_CROP
    maxes = np.max(k, axis=0)
    mins = np.min(k, axis=0)
    return (mins[1], mins[0], maxes[1]+1, maxes[0]+1)

def box_from_mask(mask, threshold=0.5*255, eps=4, min_samples=5, min_members=300, return_center_crop_on_failure=True):
    ''' mask should be (H, W) np array'''
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    mpoints = np.asarray(zip(*(mask > threshold).nonzero()))
    if len(mpoints)<10:
        return CENTER_CROP if return_center_crop_on_failure else None
    dbscan.fit(mpoints)
    labels = dbscan.labels_
    clus = {}
    for label in set(labels):
        if label==-1:
            continue
        members = mpoints[labels==label]
        if len(members)>=min_members:
            clus[label] = members
    # now select the largest cluster and make it np array
    if not clus:
        return CENTER_CROP if return_center_crop_on_failure else None # something went wrong, just return the centre crop...
    clusters = np.array(sorted(clus.values(), key=lambda x: len(x), reverse=True)[0])
    maxes = np.max(clusters, axis=0)
    mins = np.min(clusters, axis=0)
    box = (mins[1], mins[0], maxes[1], maxes[0])
    return box


def draw_box(im, box, text=None, color='green', box_thickness=2, font_scale=0.5, font_type=cv2.FONT_HERSHEY_SIMPLEX):
    color = {
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255)
    }.get(color, color)
    im = cv2.rectangle(im.copy(), (box[0], box[1]), (box[2], box[3]), color, box_thickness)
    if text:
        x, y = cv2.getTextSize(text, font_type, font_scale, 1)
        im = cv2.putText(im, text, (box[2]-x[0], box[1]+x[1]+1), font_type, fontScale=font_scale, color=color, thickness=1)
    return im


def box_from_xywh(x, y, w, h):
    return x, y, x+w, y+h

def intersection(bx1, bx2):
    dx = max(min(bx1[2], bx2[2]) - max(bx1[0], bx2[0]), 0)
    dy = max(min(bx1[3], bx2[3]) - max(bx1[1], bx2[1]), 0)
    return 1.*dx*dy

def area(bx):
    return 1.*(bx[2]-bx[0])*(bx[3]-bx[1])

def intersection_over_union(bx1, bx2):
    i = intersection(bx1, bx2)
    a1 = area(bx1)
    a2 = area(bx2)
    return 1.*i/(a1+a2-i)

