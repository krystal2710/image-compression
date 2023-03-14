import numpy as np
import random 
class my_k_means:
    def __init__(self, k, random_state = 0):
        '''
        Constructor
        '''
        self.k = k
        self.centers = [np.nan for i in range(self.k)]
        self.data = np.array([0])
        self.current_mse = 0
        self.cluster = None
        self.random_state = random_state
        self.already_fit = False
    
    def initialize_centers(self):
        '''
        Initialize random centers
        '''
        if self.k == 1: 
            self.centers[0] = np.sum(self.data,axis = 0)/len(self.data)
            return
        min_arr = np.min(self.data, axis = 0)
        max_arr = np.max(self.data, axis = 0)
        np.random.seed(self.random_state)
        self.centers = np.array([list(np.random.uniform(min_arr,max_arr)) for i in range(self.k)])
        
    def fit(self, data):
        '''
        Runs the k-means clustering algorithm on the dataset
        
        Return value: centers, number of iterations to convergence, the final error, and labels
        '''
        self.already_fit = True

        #initialization
        self.data = np.array(data)
        self.initialize_centers()
        
        #when k = 1
        if self.k == 1:
            self.cluster = [0 for i in range(len(data))]
            sum_of_errors = sum(np.sum((self.data - self.centers[0])**2, axis = 0))
            
            self.current_mse = round(sum_of_errors/len(self.data),4)
            return self.centers, 1, self.current_mse, self.cluster
        
        #when k != 1
        num_iter = 0
        done = False
        
        while not done:
            num_iter += 1
            
            #assignment phase
            
            #compute distances to each cluster center
            distances = [np.sum((self.data - self.centers[0])**2,axis = 1)]
            for i in range(1,self.k):
                distances = np.concatenate((distances, [np.sum((self.data - self.centers[i])**2,axis = 1)]),axis = 0)
                
            #identify the closest cluster
            self.cluster = distances.argmin(axis = 0)
            
            #adjustment phase
            sum_of_errors = 0
            prev_mse = 0
            for i in range(self.k):
                this_cluster = np.where(self.cluster == i, 1, 0)

                if sum(this_cluster) > 0:
                    this_cluster_arr = self.data*np.expand_dims(this_cluster,axis = 1)
                    self.centers[i] = np.sum(this_cluster_arr,axis = 0)/sum(this_cluster)
                    sum_of_errors += sum(np.sum(((this_cluster_arr - self.centers[i])**2)*np.expand_dims(this_cluster,axis = 1), axis = 1))
            
            #calculate mse and determine if it changes
            prev_mse = self.current_mse
            self.current_mse = round(sum_of_errors/len(self.data),4)
            if self.current_mse == prev_mse:
                done = True
                        
        return self.centers, num_iter, self.current_mse, self.cluster

    def closest_to_clusters(self):
        '''
        Find k data points closest to k clusters respectively

        Return value: closest points corresponding to each center in self.centers[i]
        '''
        if self.already_fit:
            closest_points = []
            if self.k == 1:
                distance_to_centers = np.sum((self.data - self.centers[0])**2, axis = 1)
                closest_points.append(distance_to_centers.argmin(axis = 0))
            else:
                for i in range(self.k):
                    this_cluster = np.where(self.cluster == i, 1, 0)
                    if sum(this_cluster) > 0:
                        this_cluster_arr = self.data*np.expand_dims(this_cluster,axis = 1)
                        this_cluster = np.where(self.cluster == i, 1, float('inf'))
                        distance_to_centers = np.sum(((this_cluster_arr - self.centers[i])**2)*np.expand_dims(this_cluster,axis = 1), axis = 1)
                        closest_points.append(distance_to_centers.argmin(axis = 0))
                    else:
                        closest_points.append(None)
            return closest_points
        return []
    
    def predict(self, new_data):
        if self.already_fit:
            #compute distances to each cluster center
            distances = [np.sum((new_data - self.centers[0])**2,axis = 1)]
            for i in range(1,self.k):
                distances = np.concatenate((distances, [np.sum((new_data - self.centers[i])**2,axis = 1)]),axis = 0)
            predicted_clusters = distances.argmin(axis = 0)
            return predicted_clusters
        return None

def compress_im(im, k):

    k_means = my_k_means(k = k)
    centers, num_iters, current_mse, clusters = k_means.fit(im)
    vq = {'model': k_means, 'centers': centers.astype(int), 'num_iters': num_iters, 'mse': current_mse, 'clusters':clusters}
    compressed_im = centers[clusters].astype(int)
    return compressed_im, vq


def compress_rep_im(rep_imgs, img_width, img_height, k, type):
    if type == 'grayscale':
        color = 1
    else:
        color = 3
    compressed_imgs = []
    vqs = []
    for im in rep_imgs: 
        im = im.reshape(img_width*img_height,color)
        compressed_im, vq = compress_im(im, k = k)
        compressed_imgs.append(compressed_im)
        vqs.append(vq)
    return np.asarray(compressed_imgs), vqs

def compressed_other_imgs(data, clusters_rep, centers_rep, rep_vqs, img_width, img_height, type):
    if type == 'grayscale':
        color = 1
    else:
        color = 3
    
    compressed_data = np.zeros((len(data),img_width*img_height*color), dtype = int)

    for i,j in enumerate(clusters_rep):
        im = data[i].reshape(img_width*img_height,color)
        predicted_clusters = rep_vqs[j]['model'].predict(im)
        compressed_im = rep_vqs[j]['centers'][predicted_clusters]
        compressed_data[i,:] = compressed_im.reshape(img_width*img_height*color)
    return compressed_data


def mean_squared_error(data, compressed_data):
    return np.mean((data - compressed_data)**2, axis = 1)

def vq_on_diff_k(k_list, data, img_width, img_height, type):
    k_rep = int(len(data)/50)
    k_means = my_k_means(k = k_rep)
    centers_rep, num_iters_rep, current_mse_rep, clusters_rep = k_means.fit(data)
    representatives = k_means.closest_to_clusters()
    
    result = []

    for k in k_list:
        compressed_rep_imgs, rep_vqs = compress_rep_im(data[representatives], img_width, img_height, k, type)
        compressed_data = compressed_other_imgs(data, clusters_rep, centers_rep, rep_vqs, img_width, img_height, type)
        mse_arr = mean_squared_error(data, compressed_data)
        result.append(round(np.median(mse_arr),2))
        print('Done with k = {}'.format(k))
    
    return result

def vq_on_diff_k_individual(k_list, data, img_width, img_height, type):
    if type == 'grayscale':
        color = 1
    else:
        color = 3
    result = []

    for k in k_list:
        compressed_data = np.zeros((len(data),img_width*img_height*color), dtype = int)
        for i, im in enumerate(data):
            im = data[i].reshape(img_width*img_height,color)
            compressed_im, vq = compress_im(im, k)
            compressed_data[i,:] = compressed_im.reshape(img_width*img_height*color)
        mse_arr = mean_squared_error(data, compressed_data)
        result.append(round(np.median(mse_arr),2))
        print('Done with k = {}'.format(k))
    return result
