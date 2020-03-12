import numpy as np

def initialize_kmeans(k, data):
    '''Create a list of k centroids (the first k values of the dataset)'''
    means = []
    for i in range(k):
        means.append(data[i])
    return means

def get_loss(means, data):
    '''Takes in a list of centroids and the data and returns the loss'''
    loss = 0
    for i in means:
        loss += sum(np.linalg.norm(data - i, axis = 1)**2) # loss = [distance(x - mu)]^2
    return loss

def get_newmean(means, data):
    '''Takes in a list of centroids and the data and returns a new list of centroids'''
    k = len(means)
    new_means = []
    distance_matrix = []
    for i in range(k): # create a matrix of distances from the centroid for each centroid
        distance = np.linalg.norm(data - means[i], axis = 1)
        distance_matrix.append(distance)
    distance_matrix = np.array(distance_matrix).T # Transpose so that the row number of the dist matrix matches the data row
    index = np.argmin(distance_matrix, axis = 1) # Find the closest centroid
    for i in range(k):
        cluster = data[np.where(index == i)] # Subset the data for each cluster
        new_mean = np.mean(cluster, axis = 0) # calculate the mean of each cluster
        new_means.append(new_mean)
    return new_means

def kmean(k, data):
    '''Calculate k mean clusters for the data'''
    means = initialize_kmeans(k, data)
    old_loss = get_loss(means, data)
    
    while True:
        means = get_newmean(means, data)
        new_loss = get_loss(means, data)
        if new_loss == old_loss:
            return means
            break
        else:
            old_loss = new_loss

#### redundant
def get_cluster(means, data):
    """
    Return the index of the clusters
    """
    k = len(means)
    clusters = []
    distance_matrix = []
    for i in range(k): # create a matrix of distances from the centroid for each centroid
        distance = np.linalg.norm(data - means[i], axis = 1)
        distance_matrix.append(distance)
    distance_matrix = np.array(distance_matrix).T # Transpose so that the row number of the dist matrix matches the data row
    index = np.argmin(distance_matrix, axis = 1) # Find the closest centroid
    for i in range(k):
        cluster_index = np.where(index == i) # get cluster index
        clusters.append(cluster_index)
    return clusters