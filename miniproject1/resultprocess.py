# import sys
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# import openpyxl
#
# respath = "D:\\res\\L2Aug"
# os.chdir(respath)
# filenames = os.listdir(respath)

#N
# av_acc = []
# max_acc = []
# for i in range(0, 5):
#     f = open("model"+str(i)+".out")
#     lines = f.readlines()
#     average_accuracy = float(lines[-1].split(' ')[2])
#     max_accuracy = float(lines[-2].split(' ')[2])
#     av_acc.append(average_accuracy)
#     max_acc.append(max_accuracy)
#
# x = np.arange(2, 7)
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, max_acc, label='max')  # Plot some data on the axes.
# ax.plot(x, av_acc, label='average')  # Plot more data on the axes...
#
# ax.set_xlabel('N')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("N vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.


#C1
# av_acc = []
# max_acc = []
# for i in range(5, 10):
#     f = open("model"+str(i)+".out")
#     lines = f.readlines()
#     average_accuracy = float(lines[-1].split(' ')[2])
#     max_accuracy = float(lines[-2].split(' ')[2])
#     av_acc.append(average_accuracy)
#     max_acc.append(max_accuracy)
#
# x = np.array([1,2,3,4,5])
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, max_acc, label='max')  # Plot some data on the axes.
# ax.plot(x, av_acc, label='average')  # Plot more data on the axes...
# ax.set_xticks(x)
# ax.set_xticklabels(['8','16','32','64','128'])
# ax.set_xlabel('C1')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("C1 vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.


#B
# av_acc = []
# max_acc = []
# for i in range(10, 15):
#     f = open("model"+str(i)+".out")
#     lines = f.readlines()
#     average_accuracy = float(lines[-1].split(' ')[2])
#     max_accuracy = float(lines[-2].split(' ')[2])
#     av_acc.append(average_accuracy)
#     max_acc.append(max_accuracy)
#
# x = np.array([1,2,3,4,5])
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, max_acc, label='max')  # Plot some data on the axes.
# ax.plot(x, av_acc, label='average')  # Plot more data on the axes...
# ax.set_xlabel('B')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("B vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.

#F
# av_acc = []
# max_acc = []
# for i in range(16, 19):
#     f = open("model"+str(i)+".out")
#     lines = f.readlines()
#
#     average_accuracy = float(lines[-1].split(' ')[2])
#     max_accuracy = float(lines[-2].split(' ')[2])
#     av_acc.append(average_accuracy)
#     max_acc.append(max_accuracy)
#
# x = np.array([3,5,7])
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, max_acc, label='max')  # Plot some data on the axes.
# ax.plot(x, av_acc, label='average')  # Plot more data on the axes...
# ax.set_xlabel('F')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("F vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.


#plt.show()

#res38random
# xlfilename = "grid18.xlsx"
# wb=openpyxl.Workbook()
# sheet = wb.active
# sheet.append(['N','B','C1','F','K','P','para count','max accuracy', 'average accuracy'])
# for i in range(0, 18):
#     f = open("model"+str(i)+".out")
#     lines = f.readlines()
#
#     N = int(lines[1].split('=')[1].strip())
#     B = lines[2].split('=')[1].strip()
#     C1 = int(lines[3].split('=')[1].strip())
#     F = lines[4].split('=')[1].strip()
#     K = lines[5].split('=')[1].strip()
#     P = int(lines[6].split('=')[1].strip())
#     ParaCnt = int(lines[7].split(' ')[3].strip())
#     max_acc = -1
#     av_acc = 0
#     cnt = 0
#     for j in range(19, 397, 13):
#         cnt+=1
#         values = lines[j].split(' ')
#         max_a = float(values[2].split('%')[0])
#         if max_a>max_acc:
#             max_acc = max_a
#         if cnt > 10:
#             av_acc += max_a
#     av_acc/=20
#     row = [N, B, C1, F, K, P, ParaCnt, max_acc, av_acc]
#     sheet.append(row)
#
#
#
# wb.save(xlfilename)

#regluar





# #ADAM
# ADAM = []
# x = range(1, 31)
# for i in range(0, 9, 2):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     ADAM.append([])
#     for j in range(12, 390, 13):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         ADAM[i//2].append(acc)
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, ADAM[0], label='ADAM')  # Plot some data on the axes.
# ax.plot(x, ADAM[1], label='ADAM+L2=1e-2')  # Plot more data on the axes...
# ax.plot(x, ADAM[2], label='ADAM+L2=1e-3')  # Plot some data on the axes.
# ax.plot(x, ADAM[3], label='ADAM+L2=1e-4')  # Plot more data on the axes...
# ax.plot(x, ADAM[4], label='ADAM+L2=1e-5')  # Plot more data on the axes...
# ax.set_xlabel('epoch')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("L2 vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.


#SGD
# SGD = []
# x = range(1, 31)
# for i in range(1, 10, 2):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     SGD.append([])
#     for j in range(12, 390, 13):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         SGD[(i-1)//2].append(acc)
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, SGD[0], label='SGD')  # Plot some data on the axes.
# ax.plot(x, SGD[1], label='SGD+L2=1e-2')  # Plot more data on the axes...
# ax.plot(x, SGD[2], label='SGD+L2=1e-3')  # Plot some data on the axes.
# ax.plot(x, SGD[3], label='SGD+L2=1e-4')  # Plot more data on the axes...
# ax.plot(x, SGD[4], label='SGD+L2=1e-5')  # Plot more data on the axes...
# ax.set_xlabel('epoch')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("L2 vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.

# ADAM = []
# x = range(1, 31)
# for i in range(0, 9, 2):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     ADAM.append([])
#     for j in range(12, 390, 13):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         ADAM[i//2].append(acc)
#
# SGD = []
# x = range(1, 31)
# for i in range(1, 10, 2):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     SGD.append([])
#     for j in range(12, 390, 13):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         SGD[(i-1)//2].append(acc)
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, ADAM[0], label='ADAM', color="red")  # Plot some data on the axes.
# ax.plot(x, ADAM[1], color="red")  # Plot more data on the axes...
# ax.plot(x, ADAM[2], color="red")  # Plot some data on the axes.
# ax.plot(x, ADAM[3], color="red")  # Plot more data on the axes...
# ax.plot(x, ADAM[4], color="red")  # Plot more data on the axes...
# ax.plot(x, SGD[0], label='SGD', color="blue")  # Plot some data on the axes.
# ax.plot(x, SGD[1], color="blue")  # Plot more data on the axes...
# ax.plot(x, SGD[2], color="blue")  # Plot some data on the axes.
# ax.plot(x, SGD[3], color="blue")  # Plot more data on the axes...
# ax.plot(x, SGD[4], color="blue")  # Plot more data on the axes...
# ax.set_xlabel('epoch')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("optimizer vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.


# ADAM = []
# x = range(1, 31)
# for i in range(0, 2):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     ADAM.append([])
#     for j in range(12, 390, 13):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         ADAM[i].append(acc)
# for i in range(2, 4):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     ADAM.append([])
#     for j in range(36, 1110, 37):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         ADAM[i].append(acc)
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
# ax.plot(x, ADAM[0], label='ADAM')  # Plot some data on the axes.
# ax.plot(x, ADAM[1], label='ADAM+L2')  # Plot more data on the axes...
# ax.plot(x, ADAM[2], label='ADAM+Aug')  # Plot some data on the axes.
# ax.plot(x, ADAM[3], label='ADAM+L2+Aug')  # Plot more data on the axes...
#
# ax.set_xlabel('epoch')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("regular vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.
#
# SGD = []
# x = range(1, 31)
# for i in range(4, 6):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     SGD.append([])
#     for j in range(12, 390, 13):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         SGD[i-4].append(acc)
# for i in range(6, 8):
#     f = open('model'+str(i)+'.out')
#     lines = f.readlines()
#     SGD.append([])
#     for j in range(36, 1110, 37):
#         values = lines[j].split(' ')
#         acc = float(values[2].split('%')[0])
#         SGD[i-4].append(acc)
#
# fig, ax = plt.subplots(figsize=(5, 2.7))
#
# ax.plot(x, SGD[0], label='SGD')  # Plot some data on the axes.
# ax.plot(x, SGD[1], label='SGD+L2')  # Plot more data on the axes...
# ax.plot(x, SGD[2], label='SGD+Aug')  # Plot some data on the axes.
# ax.plot(x, SGD[3], label='SGD+L2+Aug')  # Plot more data on the axes...
#
# ax.set_xlabel('epoch')  # Add an x-label to the axes.
# ax.set_ylabel('test accuracy')  # Add a y-label to the axes.
# ax.set_title("regular vs Accuracy")  # Add a title to the axes.
# ax.legend();  # Add a legend.
#
# plt.show()

