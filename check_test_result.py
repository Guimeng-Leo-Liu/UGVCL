import pprint, pickle

pkl_file = open(r'results\test\mnist_onlinevi_K10_1.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

pkl_file.close()

# ind = list(range(1, len(mu_p.get_shape().as_list())))
#

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

for j in range(10):
    ax1 = plt.subplot(2,5,j+1)
    x_data = list(range(10))
    y_data = []
    for i in range(10):
        if i < j:
            y_data.append(None)
        else:
            y_data.append(data1[i][j][0])
    plt.xlim(0, 9)
    ax1.plot(x_data,y_data, marker='o',)
    ax1.set_xlabel('task')
    ax1.set_ylabel('ll')
    ax1.set_title('digit %d' % j, fontsize=20, color='c')

plt.show()
