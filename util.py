import numpy as np
import json
from PIL import Image
def load_json(json_file,image_file):
    datas = []
    f = open(json_file, encoding='utf-8')
    content = json.load(f)
    annotations = content["annotations"]
    for annotation in annotations:
        name = annotation["name"]
        image_id = annotation["id"]
        num = annotation["num"]
        ignore_region = annotation["ignore_region"]
        type_bbox = annotation["type"]
        bbox = annotation["annotation"]

        gt = np.array([[cell["x"], cell["y"]] for cell in bbox])
        datas.append((image_file+name, gt, num))
    return datas

def gaussian_filter_density(gt):
    # Generates a density map using Gaussian filter transformation

    density = np.zeros(gt.shape, dtype=np.float32)

    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    # FInd out the K nearest neighbours using a KDTree

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))

    # if gt_count > 0 and gt_count < 20:

    # leafsize = 2048

    # # build kdtree
    # tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    # query kdtree
    # distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            # sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            sigma = 25
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point

        # Convolve with the gaussian filter

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density

train_data = load_json("D:/data/baidu_star_2018/annotation/annotation_train_stage1.json", "D:/data/baidu_star_2018/image/")
print(train_data[0])
#load_json("D:/data/baidu_star_2018/annotation/annotation_train_stage2.json", "D:/data/baidu_star_2018/image/")
