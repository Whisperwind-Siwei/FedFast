import matplotlib.pyplot as plt
import numpy as np

line1 = np.load('./save/fed_mnist_cnn_100_C0.1_iidFalse_0.7_True_0.7_1.0_high.npy')
line2 = np.load('./save/fed_mnist_cnn_100_C0.1_iidFalse_0.7_True_0.7_1.0_low.npy')
line3 = np.load('./save/fed_mnist_cnn_100_C0.1_iidFalse_0.7_False_0.5_0.5_high.npy')
line4 = np.load('./save/fed_mnist_cnn_100_C0.1_iidFalse_0.7_False_0.5_0.5_low.npy')
line5 = line1 - line2
line6 = line3 - line4
plt.plot(range(len(line1)), line1, label='1.0')
plt.plot(range(len(line2)), line2, label='0.7')
#plt.plot(range(len(line3)), line3, label='True')
#plt.plot(range(len(line4)), line4, label='False')
plt.legend()
plt.ylabel('Test Accuracy')
plt.show()
