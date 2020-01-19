import numpy as np
import json
from PIL import Image
import scipy.ndimage
import scipy.spatial
import pickle
from sortedcontainers import SortedDict
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib import cm as CM
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

        dot = [[cell["x"], cell["y"]] for cell in bbox]
        datas.append((image_file+name, dot))
    return datas

def compute_sigma(gt_count, distance=None, min_sigma=1, method=1, fixed_sigma=15):
    """
    Compute sigma for gaussian kernel with different methods :
    * method = 1 : sigma = (sum of distance to 3 nearest neighbors) / 10
    * method = 2 : sigma = distance to nearest neighbor
    * method = 3 : sigma = fixed value
    ** if sigma lower than threshold 'min_sigma', then 'min_sigma' will be used
    ** in case of one point on the image sigma = 'fixed_sigma'
    """
    if gt_count > 1 and distance is not None:
        if method == 1:
            sigma = np.mean(distance[1:4]) * 0.1
        elif method == 2:
            sigma = distance[1]
        elif method == 3:
            sigma = fixed_sigma
    else:
        sigma = fixed_sigma
    if sigma < min_sigma:
        sigma = min_sigma
    return sigma
def find_closest_key(sorted_dict, key):
    """
    Find closest key in sorted_dict to 'key'
    """
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))

def gaussian_filter_density(non_zero_points, map_h, map_w, distances=None, kernels_dict=None, min_sigma=2, method=1,const_sigma=15):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    gt_count = non_zero_points.shape[0]
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x = non_zero_points[i]
        sigma = compute_sigma(gt_count, distances[i], min_sigma=min_sigma, method=method, fixed_sigma=const_sigma)
        closest_sigma = find_closest_key(kernels_dict, sigma)
        kernel = kernels_dict[closest_sigma]
        full_kernel_size = kernel.shape[0]
        kernel_size = full_kernel_size // 2

        min_img_x = max(0, point_x - kernel_size)
        min_img_y = max(0, point_y - kernel_size)
        max_img_x = min(point_x + kernel_size + 1, map_h - 1)
        max_img_y = min(point_y + kernel_size + 1, map_w - 1)

        kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
        kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max,
                                                                 kernel_y_min:kernel_y_max]
    return density_map


def compute_distances(dot, n_neighbors=4, leafsize=1024):
    tree = scipy.spatial.KDTree(dot.copy(), leafsize=leafsize)  # build kdtree
    distances, _ = tree.query(dot, k=n_neighbors)  # query kdtree
    return distances

def generate_gaussian_kernels(out_kernels_path='gaussian_kernels.pkl', round_decimals=3, sigma_threshold=4, sigma_min=0,
                              sigma_max=20, num_sigmas=801):
    """
    Computing gaussian filter kernel for sigmas in linspace(sigma_min, sigma_max, num_sigmas) and saving
    them to dict.
    """
    kernels_dict = dict()
    sigma_space = np.linspace(sigma_min, sigma_max, num_sigmas)
    for sigma in sigma_space:
        sigma = np.round(sigma, decimals=round_decimals)
        kernel_size = np.ceil(sigma * sigma_threshold).astype(np.int)

        img_shape = (kernel_size * 2 + 1, kernel_size * 2 + 1)
        img_center = (img_shape[0] // 2, img_shape[1] // 2)

        arr = np.zeros(img_shape)
        arr[img_center] = 1

        arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant')
        kernel = arr / arr.sum()
        kernels_dict[sigma] = kernel

    print(f'Computed {len(sigma_space)} gaussian kernels. Saving them to {out_kernels_path}')
    with open(out_kernels_path, 'wb') as f:
        pickle.dump(kernels_dict, f)

precomputed_kernels_path = 'gaussian_kernels.pkl'
# uncomment to generate and save dict with kernel sizes
generate_gaussian_kernels(precomputed_kernels_path, round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=20,
                          num_sigmas=801)
with open(precomputed_kernels_path, 'rb') as f:
    kernels_dict = pickle.load(f)
    kernels_dict = SortedDict(kernels_dict)



def dataset(train_data):
    min_sigma = 2  ## can be set 0
    method = 1
    for cell in train_data:
        img, dot = np.array(Image.open(cell[0])), cell[1]
        img_h, img_w, ch = img.shape
        new_dot = []
        for i in range(0, len(dot)):
            if int(dot[i][0]) < img_w and int(dot[i][1]) < img_h:
                new_dot.append([int(dot[i][0]), int(dot[i][1])])
        new_dot = np.array(new_dot)
        distances = compute_distances(new_dot,n_neighbors=4,leafsize=1024)
        #(non_zero_points, map_h, map_w, distances=None, kernels_dict=None, min_sigma=2, method=1,const_sigma=15):
        density_map = gaussian_filter_density(new_dot, map_h=img_h, map_w=img_w, distances=distances, kernels_dict=kernels_dict,
                                          min_sigma=min_sigma, method=method)
        print(img.shape,density_map.shape)







train_data = load_json("E:/wangtao/density/baidu_star_2018/annotation/annotation_train_stage1.json", "E:/wangtao/density/baidu_star_2018/image/")
#load_json("D:/data/baidu_star_2018/annotation/annotation_train_stage2.json", "D:/data/baidu_star_2018/image/")
dataset(train_data)

