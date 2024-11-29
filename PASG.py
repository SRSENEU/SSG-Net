import os
import time
import cv2
import numpy as np
from skimage import segmentation
import torch
import torch.nn as nn
import torchvision

torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.set_device(1)


input_path = './BUSI/images/'
out_path_superpixel = './BUSI/superpixel_labels/'
out_path_box = './BUSI/box_labels/'
def convexHull(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    contours2, _ = cv2.findContours(image_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull2 = [cv2.convexHull(contour, False) for contour in contours2]
    filling = np.zeros_like(image_bw)

    for h in hull2:
        cv2.fillPoly(filling, [h], 255)

    return filling

def side_overlap_percentage(img1, img2):


    img1_resized = cv2.resize(img1, (256, 256))
    img2_resized = cv2.resize(img2, (256, 256))

    _, img1_mask = cv2.threshold(img1_resized, 240, 255, cv2.THRESH_BINARY)
    _, img2_mask = cv2.threshold(img2_resized, 240, 255, cv2.THRESH_BINARY)

    img1_edges = cv2.Canny(img1_mask, 100, 200)
    img2_edges = cv2.Canny(img2_mask, 100, 200)

    overlap_edges = cv2.bitwise_and(img1_edges, img2_edges)

    img2_edge_length = np.sum(img2_edges > 0)

    overlap_edge_length = np.sum(overlap_edges > 0)

    if img2_edge_length > 0:
        overlap_percentage = (overlap_edge_length / img2_edge_length) * 100
    else:
        overlap_percentage = 0
    return overlap_percentage


def discrimination(img1,box1,out_superpixel,out_box):

    box = box1.copy()
    box[box == 255] = 1
    pix_num_box = box.sum()
    img = img1.copy()
    img[img == 255] = 1
    pix_num_img = img.sum()

    overlap_rate = side_overlap_percentage(img1,box1)

    if pix_num_img < 0.5*pix_num_box or overlap_rate > 30 or pix_num_img > 0.9*pix_num_box:
        cv2.imwrite(out_box, box1)
    else:
        cv2.imwrite(out_superpixel, img1)
class Args(object):
    train_epoch = 2 ** 6
    mod_dim1 = 64  #
    mod_dim2 = 20
    gpu_id = 0

    min_label_num = 40
    max_label_num = 256


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def run():
    args = Args()
    i = 1
    for obj in os.listdir(input_path):

        start_time0 = time.time()
        n = obj
        rat = input_path + n
        box_path = './BUSI/box/'
        rat_box = box_path + n
        torch.cuda.manual_seed_all(43)
        torch.manual_seed(43)
        np.random.seed(43)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


        # image = cv2.imread(full_path)
        image_gray = cv2.imread(rat, cv2.IMREAD_GRAYSCALE)
        box = cv2.imread(rat_box, cv2.IMREAD_GRAYSCALE)
        img_np = np.array(image_gray)
        width, height = img_np.shape

        # Gray scale variation
        Q = img_np.astype('float')
        for i in range(0, width):
            for j in range(0, height):
                if 0 <= Q[i, j] < 150:
                    Q[i, j] = 0 + Q[i, j] * (50 / 100)
                elif 50 <= Q[i, j] < 200:
                    Q[i, j] = 50 + (Q[i, j] - 100) * (200 - 50) / (200 - 100)
                elif 200 <= Q[i, j] < 255:
                    Q[i, j] = 200 + (Q[i, j] - 200) * (255 - 200) / (255 - 200)

        for i in range(0, width):
            for j in range(0, height):
                if Q[i, j] <= 0:
                    Q[i, j] = 0
                elif Q[i, j] >= 255:
                    Q[i, j] = 255

        image_output = Q.astype('uint8')
        image = np.repeat(image_output[..., np.newaxis], 3, axis=-1)

        seg_map = segmentation.slic(image, n_segments=300, compactness=10, sigma=2, start_label=1)

        seg_map = seg_map.flatten()
        seg_lab = [np.where(seg_map == u_label)[0]
                   for u_label in np.unique(seg_map)]

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        tensor = image.transpose((2, 0, 1))
        tensor = tensor.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, :, :, :]
        tensor = torch.from_numpy(tensor).to(device)

        model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

        image_flatten = image.reshape((-1, 3))

        color_avg = np.random.randint(255, size=(args.max_label_num, 3))
        show = image

        model.train()

        for batch_idx in range(args.train_epoch):
            '''forward'''
            optimizer.zero_grad()
            output1 = model(tensor)[0]


            output = output1.permute(1, 2, 0).view(-1, args.mod_dim2)


            target = torch.argmax(output, 1)


            im_target = target.data.cpu().numpy()

            '''refine'''
            for inds in seg_lab:
                u_labels, hist = np.unique(im_target[inds], return_counts=True)
                im_target[inds] = u_labels[np.argmax(hist)]


            '''backward'''
            target = torch.from_numpy(im_target)
            target = target.to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            '''show image'''
            un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
            if un_label.shape[0] < args.max_label_num:  # update show
                img_flatten = image_flatten.copy()
                if len(color_avg) != un_label.shape[0]:
                    color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=int) for label in un_label]
                for lab_id, color in enumerate(color_avg):
                    img_flatten[lab_inverse == lab_id] = color
                show = img_flatten.reshape(image.shape)

            if len(un_label) < args.min_label_num:
                break

        out_superpixel = out_path_superpixel + n[:-4] + '.png'
        out_box = out_path_box + n[:-4] + '.png'

        show = np.float32(show)
        show = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
        map_box = box.copy()
        for i in range(map_box.shape[0]):
            for j in range(map_box.shape[1]):
                if map_box[i, j] == 255:
                    map_box[i, j] = show[i, j]

        # cv2.imwrite(out, box)
        box1 = box.copy()
        box1[box1 == 255] = 1
        pix_num_box = box1.sum()
        nonzero_pixels = map_box[np.nonzero(map_box)]
        unique_nonzero_pixels = np.unique(nonzero_pixels)
        if len(unique_nonzero_pixels) >= 3:
            # 获取第二小的像素值
            first_min_nonzero = unique_nonzero_pixels[0]
            second_min_nonzero = unique_nonzero_pixels[1]
            third_min_nonzero = unique_nonzero_pixels[2]
            if len(unique_nonzero_pixels) >= 4:
                fourthly_min_nonzero = unique_nonzero_pixels[3]
        else:
            print("There are no at least three nonzero pixel values in the image.")
        img_1 = map_box.copy()
        img_1[img_1 == first_min_nonzero] = 255
        img_1[(img_1 > 0) & (img_1 < 255)] = 0
        img_1_1 = img_1.copy()  # img_1_1是目标区域255
        img_1[img_1 == 255] = 1
        pix_num_img = img_1.sum()
        img_2 = map_box.copy()

        if (pix_num_img / pix_num_box) < 0.4:
            img_2[img_2 == second_min_nonzero] = 255
            img_2[(img_2 > 0) & (img_2 < 255)] = 0
            img_2_1 = img_2.copy()  # img_2_1目标区域是255
            img_2[img_2 == 255] = 1
            img_2_255 = img_1_1 + img_2_1
            img_2_true = img_1 + img_2
            pix_num_img_2 = img_2_true.sum()
            img_3 = map_box.copy()

            if (pix_num_img_2 / pix_num_box) < 0.5:
                img_3[img_3 == third_min_nonzero] = 255
                img_3[(img_3 > 0) & (img_3 < 255)] = 0
                img_3_1 = img_3.copy()  # img_3_1目标区域是255
                img_3[img_3 == 255] = 1
                img_3_255 = img_2_255 + img_3_1
                img_3_true = img_3 + img_2_true
                pix_num_img_3 = img_3_true.sum()
                if (pix_num_img_3 / pix_num_box) < 0.45:
                    i = i + 1
                    cv2.imwrite(out_box, box)
                else:
                    img_3_255 = convexHull(img_3_255)
                    discrimination(img_3_255,box,out_superpixel,out_box)
            else:
                img_2_255 = convexHull(img_2_255)
                discrimination(img_2_255,box,out_superpixel,out_box)
        else:
            img_1_1 = convexHull(img_1_1)
            discrimination(img_1_1,box,out_superpixel,out_box)

        print(f'Image: {i}')
        i += 1
if __name__ == '__main__':
    run()
