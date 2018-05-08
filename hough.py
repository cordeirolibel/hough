
# coding: utf-8

# # Hough Transform
# The Hough transform is a feature extraction technique used in image analysis, computer vision, and digital image processing. The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.

# In[1]:

import numpy as np
from matplotlib import pyplot as plt 
from skimage.filters.rank import gradient
from skimage.morphology import disk
from skimage.transform import hough_line, hough_line_peaks
from collections import deque
import cv2


# In[2]:

def imshow(img):
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(img,cmap='gray')
    plt.show()


# Simple gradient
def bin_im1(im_gray, _disk = 1):
    im_grad = gradient(im_gray, disk(_disk))
    # Otsu's thresholding 
    ret3,im_bin = cv2.threshold(im_grad,0,255,cv2.THRESH_OTSU)
    return im_bin


# Laplacian
def bin_im2(im_gray,thresholds = (50,200)):
    im_grad = cv2.Laplacian(im_gray,cv2.CV_32F)
    im_grad = np.abs(im_grad).astype('uint8')
    
    # Otsu's thresholding 
    ret3,im_bin = cv2.threshold(im_grad,0,255,cv2.THRESH_OTSU)
    return im_bin


# Canny
def bin_im3(im_gray,thresholds = (50,200)):
    im_bin = cv2.Canny(im_gray,thresholds[0],thresholds[1])
    return im_bin


def plot_lines(im_gray,dists,angles,linewidth=0.7):
    y_max, x_max = im_gray.shape
    pts1 = np.around(dists/np.sin(angles))                        #x=0     => [0,r/sin(o)]
    pts2 = np.around((dists-x_max*np.cos(angles))/np.sin(angles)) #x=x_max => [x_max,(r-x_max.cos(o))/sin(o)]
    
    #plot img
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(im_gray, cmap='gray')
    
    #plot lines
    for pt1,pt2 in zip(pts1,pts2):
        #plt.plot(x,y)
        plt.plot((0, x_max), (pt1, pt2), 'r',linewidth=linewidth)

    # [xmin, xmax, ymin, ymax]
    plt.axis((0,x_max,y_max,0))
    
    plt.show()


#simple implementation, slow
def hough_transformation1(im_bin, steps = (2000,2000)):
    y_max, x_max = im_bin.shape

    r_steps = steps[0]
    r_max = np.sqrt(y_max**2+x_max**2)
    r_min = -r_max
    r_step = (r_max-r_min)/(r_steps-1)
    r_list = np.arange(r_min,r_max+r_step,r_step)
    if r_list.shape[0]!=r_steps:
        r_list = r_list[:-1]

    o_steps = steps[1]
    o_max = -np.pi/2
    o_min = -o_max
    o_step = (o_max-o_min)/(o_steps-1)
    o_list = np.arange(o_min,o_max+o_step,o_step)
    if o_list.shape[0]!=o_steps:
        o_list = o_list[:-1]


    im_hough = np.zeros((r_steps,o_steps))
    for y in range(im_bin.shape[0]):
        for x in range(im_bin.shape[1]):
            if im_bin[y,x]:
                r_o_list = x*np.cos(o_list)+y*np.sin(o_list)
                r = np.round((r_o_list-r_min)/r_step).astype('int')
                im_hough[r,range(o_steps)]+=1

    return im_hough, o_list, r_list
            
# with Skimage, fast
def hough_transformation2(im_bin, steps = 2000):
    im_hough, angles, dists = hough_line(im_bin,np.arange(-np.pi/2,np.pi/2,np.pi/steps))
    return im_hough, angles, dists


def hough_peaks1(im_hough, o_list, r_list, alpha=0.5):
    
    #mask create
    threshold = (im_hough.max()-im_hough.min())*alpha
    mask_peaks = im_hough>threshold
    
    #find all lines
    angles = list()
    dists = list()
    peaks = list()
    for r in range(mask_peaks.shape[0]):
        for o in range(mask_peaks.shape[1]):
            if mask_peaks[r][o]:
                angles.append(o_list[o])
                dists.append(r_list[r])
                peaks.append(im_hough[r][o])
    
    #to numpy
    angles = np.array(angles)
    dists = np.array(dists) 
    peaks = np.array(peaks) 
    
    print('Threshold:',threshold,'Peaks or Lines:',len(peaks))
    
    return peaks,angles,dists



# ## Find Lines 2
# Simple Threshold <br>
# $\downarrow$  <br>
# Graph by Euclidean Distances of the peaks <br>
# $\downarrow$  <br>
# Component Connection <br>
# $\downarrow$  <br>
# Lines by weighted arithmetic mean 


def hough_peaks2(im_hough, o_list, r_list, alpha=0.4, beta = 3):

    # In[26]:

    peaks,angles,dists = hough_peaks1(im_hough, o_list, r_list,alpha=alpha)


    ## ==== Create Graph by Euclidean Distances

    adjacency_lists = list()

    for i in range(len(peaks)):
        #Euclidean distances between point i and all points before i
        x1 = angles[i]
        x2 = angles#[i+1:]
        y1 = dists[i]
        y2 = dists#[i+1:]
        euclid_dist = (x2-x1)**2+(y2-y1)**2
        # very slow function
        #euclid_dist = euclidean_distances([[x1,y1]],zip(x2,y2))[0]
        
        #find edges
        edges = euclid_dist<(beta**2)
        edges = np.arange(edges.shape[0])[edges]
        #edges += i+1
        #edges = [(i,j) for j in edges]

        #save edges
        adjacency_lists.append(edges.astype('int32'))
        #edges_list = np.append(edges_list,np.array(edges))
        #edges_list += list(edges)


    adjacency_lists = np.array(adjacency_lists)


    ## == Component Connection:

    visited = np.zeros(len(peaks))
    n_class = 0

    for v in range(len(peaks)):
        if visited[v]:
            continue
        
        #new class or component
        n_class += 1
        visited[v] = n_class
        
        #stack of vertices to look
        stack = deque()
        stack.append(v)
        
        while stack:
            #new vertice
            v = stack.pop()
            
            #add all neighbors not visited
            for neighbor in adjacency_lists[v]:
                
                if not visited[neighbor]:
                    stack.append(neighbor)
                    visited[neighbor] = n_class


    ## == Creating Lines: weighted arithmetic mean

    avg_angles = list()
    avg_dists = list()
    avg_peaks = list()

    for c in range(n_class):
        
        #class mask
        mask = visited==(c+1)
        
        #component connection points
        cc_angles = angles[mask]
        cc_dists = dists[mask]
        cc_peaks = peaks[mask]
        
        #weighted arithmetic mean 
        angle = np.average(cc_angles, weights=cc_peaks)
        dist = np.average(cc_dists, weights=cc_peaks)
        peak = np.average(cc_peaks)
        
        #save
        avg_angles.append(angle)
        avg_dists.append(dist)
        avg_peaks.append(peak)

    #to numpy
    avg_angles = np.array(avg_angles)
    avg_dists = np.array(avg_dists)
    avg_peaks = np.array(avg_peaks)

    return avg_peaks, avg_angles, avg_dists


def hough_peaks3(im_hough, o_list, r_list):
    peaks, angles, dists = hough_line_peaks(im_hough, o_list, r_list)
    return peaks, angles, dists

# ## Find Lines 3

# In[32]:

def func_conv(im_hough, function, ksize=(5,5), strides=(1,1), *args):
    
    #only odd numbers
    if not (ksize[0]%2 and ksize[1]%2):
        return
    
    #sizes
    border = np.array([int(ksize[0]/2),int(ksize[1]/2)])
    h_size = im_hough.shape
    
    #border image
    im_border = np.zeros(h_size+2*border)
    im_border[border[0]:h_size[0]+border[0],border[1]:h_size[1]+border[1]] = im_hough
    
    #convolution function
    im_out = list()
    for i in range(0,h_size[0],strides[0]):
        line = list()
        for j in range(0,h_size[1],strides[1]):
            
            im_cut = im_border[i:i+ksize[0],j:j+ksize[1]]
            pixel = function(im_cut,*args)
            
            line.append(pixel)
        im_out.append(line)
    
    #return 
    im_out = np.array(im_out).astype('float32')
    
    return im_out



def func_kernel1(im):
    
    n = im.shape[0]*im.shape[1]
    kernel = np.ones(im.shape)
    
    return np.sum(im*kernel)/n


# 

# ### 3.2 -  Kernel 2 (Gaussian)
# 
# $A = \begin{bmatrix}0.0625\\0.25\\0.375\\0.25\\0.0625 \end{bmatrix} \ \ \ \ \
# K = A.A^T = \begin{bmatrix}  0.0039&0.0156&0.0234&0.0156&0.0039 \\ 0.0156&0.0625&0.0937&0.0625&0.0156 \\ 0.0234&0.0937&0.1406&0.0937&0.0234 \\ 0.0156&0.0625&0.0937&0.0625&0.0156 \\ 0.0039&0.0156&0.0234&0.0156&0.0039  \end{bmatrix}$

# In[36]:

def func_kernel2(im,sigma=-1):
    
    #kernel 1D
    gaus1D_1 = cv2.getGaussianKernel(im.shape[0],sigma)
    gaus1D_2 = cv2.getGaussianKernel(im.shape[1],sigma)
    
    #kernel 2D
    gaus2D = np.dot(gaus1D_1,gaus1D_2.T)

    return np.sum(im*gaus2D)



def kernel3(im,kernel):
    out = np.sum(im*kernel)
    return out


