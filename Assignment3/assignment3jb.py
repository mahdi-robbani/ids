import os
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # # To work in the actual directory
    os.chdir(os.path.dirname(sys.argv[0]))   # Line not in the final handin!
    # data = np.loadtxt("murderdata2d.txt")
    # # center the data:
    # data_mean = np.mean(data, axis=0) # 0 for columnwise
    # centered_data = data - data_mean
    # # or standardize the data!
    # data_std = np.std(data, axis=0)
    # stand_data = (data - data_mean) / data_std

    # # print("centered\n", centered_data)
    # # print("not centered\n", data)

    # cov_matrix = np.cov(stand_data.T)  # row as features
    # print("Covariance matrix:\n", cov_matrix)
    # evals, evecs = np.linalg.eig(cov_matrix)  # np.linalg.heigh() for ascending order
    # print("evals:\n", evals)
    # print("evecs:\n", evecs)


    #  sort the data:
    # increasing_indexes = evals.argsort()[::-1]  # we sort backwards,
    #  the two colons: from start to end (Extended Slices)
    # # index2 = np.argsort(-evals)
    # print("index: \n", increasing_indexes)
    # # print("snd index:\n", index2)

    # evals = evals[increasing_indexes]
    # print("sorted evals\n", evals)
    # evecs = evecs[:,increasing_indexes]
    # print("sorted evecs\n", evecs)

    # # cov matrix
    # new_cov_matrix = np.diag(evals)
    # print("the associated diagonal matrix:\n", new_cov_matrix)
    # print("first PC:\n", evecs[1])    

    # # we wanna scale our new coordinate system to visualize the impact of the two features on the data:
    # s0 = np.sqrt(evals[0])
    # s1 = np.sqrt(evals[1])

    # plt.plot([0, s0*evecs[0, 0]], [0, s0*evecs[1, 0]], 'r')
    # plt.plot([0, s1*evecs[0, 1]], [0, s1*evecs[1, 1]], 'r')
    # plt.scatter(stand_data[:, 0], stand_data[:, 1])

    # plt.axis()
    # plt.show()
    
# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues
# def pca(data):
#     pass


# WITH THE SECOND DATA SET
    data_raw = np.loadtxt("IDSWeedCropTrain.csv", delimiter=",")
    datatest = np.loadtxt("IDSWeedCropTest.csv", delimiter=",")

    # data = data_raw[:, : -1]
    data = datatest[:, : -1]


    # standardize the data!
    # data_mean = np.mean(data, axis=0)
    # data_std = np.std(data, axis=0)
    # stand_data = (data - data_mean) / data_std

    # # print("centered\n", centered_data)
    # # print("not centered\n", data)

    # cov_matrix = np.cov(stand_data.T)  # row as features
    # print(cov_matrix.shape)
    # # print("Covariance matrix:\n", cov_matrix)
    # evals, evecs = np.linalg.eig(cov_matrix)  # np.linalg.heigh() for ascending order
    # print("evals:\n", evals.shape)
    # # print("evecs:\n", evecs)


    # # # sort the data:
    # increasing_indexes = evals.argsort()[::-1]  # we sort backwards, the two colons: from start to end (Extended Slices)
    # # # index2 = np.argsort(-evals)
    # print("index: \n", increasing_indexes)
    # # # print("snd index:\n", index2)

    # evals = evals[increasing_indexes]
    # print("sorted evals\n", evals)
    # evecs = evecs[:,increasing_indexes]
    # print("sorted evecs\n", evecs.shape)

    # # # cov matrix
    # new_cov_matrix = np.diag(evals)
    # # print("the associated diagonal matrix:\n", new_cov_matrix)
    # # print("first PC:\n", evecs[1])

    # # # we wanna scale our new coordinate system to visualize the impact of the two features on the data:
    # s0 = np.sqrt(evals[0])
    # print("s0: ", s0)
    # s1 = np.sqrt(evals[1])

    # data_to_plot = evecs.T[0:2, :] @ stand_data.T
    # print(data_to_plot.shape)

    # # to plot our components in the new coordinates system
    # new_evecs = evecs.T[0:2, :] @ evecs[:, 0:2]  # Q*QT! so Identity matrix

    # print("Mahdi is perpendicular", np.dot(evecs[:,0], evecs[:,1]))
    # # plt.plot([0, s0*evecs[1, 0]], [0, s0*evecs[0, 0]], 'r')
    # # plt.plot([0, s1*evecs[0, 0]], [0, s1*evecs[0, 1]], 'r')
    # plt.scatter(data_to_plot[0, :], data_to_plot[1, :])

    # plt.axis()
    # plt.show()
    
    # # get the cumulative variances:
    # c_var = np.cumsum(evals/np.sum(evals))
    # print("cumulative variances:\n", c_var, "\n and its shape: ", c_var.shape)
    # # plt.plot(c_var)
    # # plt.show()
    # print(sum(evals/np.sum(evals)))
    # # DO NOT FORGET LABELS, TITLES AND SO ON

    # var_captured = []
    # # for pc in c_var:
    # #     if var_captured >= 0.9:


# Exercise 3 ###

def get_loss(cluster1, cluster2, centroid1, centroid2):
    loss1 = np.sum(np.linalg.norm(cluster1 - centroid1, axis=1)**2)
    loss2 = np.sum(np.linalg.norm(cluster2 - centroid2, axis=1)**2)
    return loss1 + loss2


def get_cluster(data, centroid1, centroid2):
    dist1 = np.linalg.norm(data - centroid1, axis=1)  # 1000 x 1
    dist2 = np.linalg.norm(data - centroid2, axis=1)
    index1 = list(dist1 < dist2)
    index2 = list(dist2 < dist1)
    cluster1 = data[index1]
    cluster2 = data[index2]
    return cluster1, cluster2

# def get_mean(cluster1, cluster2)

# initialization
centroid1 = data[0, :]
centroid2 = data[1, :]

cluster1, cluster2 = get_cluster(data, centroid1, centroid2)
old_loss = get_loss(cluster1, cluster2, centroid1, centroid2)

# repeat
nb_iteration = 0
stop = False

while not stop:
    centroid1 = np.mean(cluster1)
    centroid2 = np.mean(cluster2)
    cluster1, cluster2 = get_cluster(data, centroid1, centroid2)
    new_loss = get_loss(cluster1, cluster2, centroid1, centroid2)
    print("the final loss is: ", new_loss)
    if new_loss - old_loss == 0:
        stop = True
    else:
        # print("old_loss: ", old_loss, "\n")
        # print("new_loss: ", new_loss, "\n")
        old_loss = new_loss
        nb_iteration += 1
        print("iteration number: ", nb_iteration)



print("the centroids are: ", centroid1, centroid2)

# we update the centroid
    # centroid1 = np.mean(cluster1) 
    # centroid2 = np.mean(cluster2)

# loss function


