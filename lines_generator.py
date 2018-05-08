
# coding: utf-8

# # Lines Generator

# In[1]:

from matplotlib import pyplot as plt 
from skimage.exposure import equalize_hist
from skimage.draw import line_aa
import numpy as np
import cv2

# ## Aux function

# In[4]:

def norm(im):
    im = np.array(im).astype('float32')
    im = (im-np.min(im))*255/(np.max(im)-np.min(im))
    return im.astype('uint8')



# ## Add Lines

# In[6]:

def add_lines(im,delta_lines,aplha_lines):
    im = im.copy()

    #Parameters
    size = im.shape
    d_max = np.floor(np.sqrt(size[0]**2+size[1]**2))-10
    n_lines = np.random.randint(delta_lines[0],delta_lines[1])
    
    #add lines
    lines = list()
    for l in range(n_lines):

        #random line
        dist = np.random.random()*d_max
        angle = np.random.random()*2*np.pi-np.pi
        alpha = np.random.random()*(aplha_lines[1]-aplha_lines[0])+aplha_lines[0]

        #save
        lines.append([dist,angle,alpha])

        #to rectangle
        pt1 = np.around(dist/np.sin(angle))                         #x=0     => [0,d/sin(o)]
        pt2 = np.around((dist-size[1]*np.cos(angle))/np.sin(angle)) #x=x_max => [x_max,(d-x_max.cos(o))/sin(o)]

        #line pixels
        rows, cols, vals = line_aa(int(pt1), 0, int(pt2), size[1])
        for r,c,v in zip(rows, cols, vals):
            try:
                im[c][r] = v*alpha
            except:#out matrix
                break
    
    return np.array(lines), norm(im)


# ## Texture
# Load all textures:

# In[8]:

def load_textures(n_textures, size = (1280,720)):

    ims_texture = list()
    for t in range(n_textures):
        #load
        ims_texture.append(cv2.imread('imgs/texture'+str(t+1)+'.jpg',cv2.IMREAD_GRAYSCALE))
        #resize
        ims_texture[-1] = cv2.resize(ims_texture[-1], size)
        #normalize
        ims_texture[-1] = norm(ims_texture[-1])

    return np.array(ims_texture)


# Add texture:

# In[9]:

def add_texture(im, ims_texture,alpha_textures):
    im = im.copy()
    
    #random texture
    n_textures = ims_texture.shape[0]
    i_texture = np.random.randint(0,n_textures)
    alpha = np.random.random()*(alpha_textures[1]-alpha_textures[0])+alpha_textures[0]

    #add 
    im = im+ims_texture[i_texture]*alpha

    #print('i:',i_texture,' alpha:',alpha)
    return norm(im)


# ## Binary Noise

# In[11]:

def add_binary_noise(im,noise_probability):
    im = im.copy().astype('int16')
    
    #random image [0,1]
    im_prob = np.random.random(im.shape)

    #invert some pixels 
    #im[im_prob<noise_probability] = 255-im
    im[im_prob<noise_probability] *= -1
    im[im_prob<noise_probability] += 255
         
    return norm(im)


# ## Smoothing

# In[13]:

def smoothing(im,delta_kernel):
    im = im.copy()
    
    #random kernel, only odd
    kernel = np.random.randint(delta_kernel[0]/2,delta_kernel[1]/2+1)
    kernel = kernel*2+1
    
    #gaussian filter
    im = cv2.GaussianBlur(im,(kernel,kernel),-1)

    return norm(im)




# ## Noise

# In[15]:

def add_noise(im,delta_sigma):
    im = im.copy()
    
    #random noise
    im_noise = np.random.randint(delta_sigma[0],delta_sigma[1],im.shape)
    im_noise = im + im_noise
    
    #limits
    im_noise[im_noise>255] = 255
    im_noise[im_noise<0] = 0
    
    return norm(im_noise)




# ## Equalize

# In[16]:

def equalize(im):
    im = im.copy()
    im = equalize_hist(im,mask = (im!=0))
    return norm(im)




# ## Image generator

# In[17]:

def im_generator(n_images=1,
                 size = (1280,720),
                 delta_lines = [3,30],
                 aplha_lines = [0.5,1],
                 n_textures = 3,
                 alpha_textures = [0.05, 0.2],
                 binary_noise_probability = 0.005,
                 smooth_delta_kernel = [3,11],
                 noise_delta_sigma = [-30,30]):
    
    #load textures
    ims_texture = load_textures(n_textures,size)
    
    ims = list()
    lines_list = list()
    for i in range(n_images):
        #black image
        im = np.zeros((size[1],size[0]))

        #Add Lines
        lines,im = add_lines(im,delta_lines,aplha_lines)

        #Add Texture
        im = add_texture(im, ims_texture,alpha_textures)

        #Add Binary Noise
        im = add_binary_noise(im,binary_noise_probability)

        #Smoothing
        im = smoothing(im,smooth_delta_kernel)

        #Add Noise
        im = add_noise(im,noise_delta_sigma)
        
        #Equalize
        im = equalize(im)
        
        #Lines
        ims.append(im)
        lines_list.append(lines)
    
    #to numpy
    ims = np.array(ims)
    lines_list = np.array(lines_list)
    
    return lines_list,ims




