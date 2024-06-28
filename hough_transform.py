import numpy as np
import cv2
import math
import pandas as pd
from matplotlib import pyplot as plt


EDGE_VALUE = 255

threshold_theta = 15
threshold_rho = 1
threshold = 100 #todo check threshhold

image_path = "tmuna5.jpg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 100, apertureSize=3) #np (2d) array

rows=edges.shape[0]
cols=edges.shape[1]
LARGEST_RHO = int(math.sqrt(rows ** 2 + cols ** 2))

matrix = np.zeros((int(180 / threshold_theta), LARGEST_RHO), dtype=object) #rows = theta, cols = rho
edges_indices = []

# create a list of tuples representing the indices of the edges

for index, row in enumerate(edges):
    for color_index, color in enumerate(row):
        if color == EDGE_VALUE:
            edges_indices.append((color_index, index)) #col, row so it will be x and than y

for edge in edges_indices:
    for theta in range(0, 180, threshold_theta):
        theta_rad = math.radians(theta)
        rho = edge[0] * np.cos(theta_rad) + edge[1] * np.sin(theta_rad)
        rho = round(rho)
        if matrix[int(theta/threshold_theta)][rho] == 0:
            matrix[int(theta/threshold_theta)][rho] = [1, [edge]]
        else:
            matrix[int(theta/threshold_theta)][rho][0] += 1
            matrix[int(theta/threshold_theta)][rho][1].append(edge)
#df= pd.DataFrame(matrix)

def draw_line(image, p1, p2):
    cv2.line(image, p1, p2, (20, 100, 255), 2)



for i in range(len(matrix)): #rows
    for j in range(len(matrix[0])):
        if matrix[i][j] == 0:
            continue
        if matrix[i][j][0] > threshold:
            for k in range(len(matrix[i][j][1])-2):
                p1 = matrix[i][j][1][k]
                p2 = matrix[i][j][1][k+1]
                if matrix[i][j][1][k + 2][0] - matrix[i][j][1][k + 1][0] > 1 or matrix[i][j][1][k + 2][1] - matrix[i][j][1][k + 1][1] > 1:
                    continue
                else:
                    draw_line(image, p1, p2)


plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.axis('off')
plt.show()