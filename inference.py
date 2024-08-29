import torch
import pathlib
import torchvision
from Noiseprint.noiseprint import *
from PIL import Image
import cv2
from pyheatmap.heatmap import HeatMap

def test_image(image):
    #first step (feature image)
    _, noise_print = getNoiseprint(image)
    noise = Image.fromarray(np.uint8(noise_print))
    noise.save(f"./data/feature1.png")
    #second step (convert to heatmap)
    im = cv2.imread("./data/feature1.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x , y = [], []
    for h in range(0, im.shape[0]):
        for w in range(0, im.shape[1]):
            if (im[h][w]):
                y.append(h)
                x.append(w)
    data = []
    for j in range(0, len(x)):
        tmp = [x[j], y[j], 1]
        data.append(tmp)
    heat = HeatMap(data)
    heat.heatmap(save_as=f"./data/heatmap1.png")

if __name__ == '__main__':
    test_image("./data/test1.jpeg")