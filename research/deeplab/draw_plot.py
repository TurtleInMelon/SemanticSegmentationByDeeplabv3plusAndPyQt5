import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family']=['STsong']
fig = plt.figure(figsize=(13, 9))
# x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# y1 = [0.9456, 0.7590, 0.9134, 0.5722, 0.5793, 0.5056, 0.6068, 0.6388, 0.8990, 0.5782,
#       0.9227, 0.7385, 0.5329, 0.9034, 0.7976, 0.8150, 0.7635, 0.5908, 0.6894]
# ResNet
x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y2 = [0.9724, 0.7977, 0.8993, 0.4554, 0.5387, 0.4662, 0.5836, 0.6679, 0.9066, 0.5862,
      0.9362, 0.7183, 0.5149, 0.9179, 0.6988, 0.7406, 0.5759, 0.5483, 0.6887]
# Xception65
x3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y3 = [0.9593, 0.7579, 0.9003, 0.5652, 0.5668, 0.4638, 0.5766, 0.6519, 0.9035, 0.5753,
      0.9274, 0.7192, 0.5271, 0.9048, 0.7922, 0.8234, 0.7527, 0.5810, 0.6878]
# MobleNetV2
x4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y4 = [0.9587, 0.7140, 0.8464, 0.3154, 0.4406, 0.2902, 0.4258, 0.5142, 0.8453, 0.5082,
      0.8250, 0.6347, 0.4381, 0.8536, 0.6478, 0.6734, 0.5165, 0.4788, 0.6204]
group_labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
                'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
plt.title('不同模型MIoU值')
plt.xlabel('目标类别')
plt.ylabel('MIoU')

# plt.plot(x1, y1, 'y', label='改进的模型')
# plt.plot(x2, y2,'b',label='join')
# plt.xticks(x1, group_labels, rotation=0)
plt.plot(x2, y2, 'g', label='ResNet101')
# plt.plot(x3, y3, 'r', label='Xception65')
plt.plot(x4, y4, 'b', label='MobileNetV2')
plt.xticks(x3, group_labels, rotation=-15)
# y_ticks = np.arange(0.2, 1.0, 0.025)
# plt.yticks(y_ticks)
# for a, b in zip(x1, y1):
#     plt.text(a, b, b, ha='center')
#
for a, b in zip(x2, y2):
    plt.text(a, b, b, ha='center')

# for a, b in zip(x3, y3):
#     plt.text(a, b, b, ha='center')

for a, b in zip(x4, y4):
    plt.text(a, b, b, ha='center')

plt.legend(bbox_to_anchor=[0.3, 1], loc='best')
plt.savefig('fix.jpg', dpi=500)
plt.show()


