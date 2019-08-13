


import matplotlib
matplotlib.use('Agg')
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#Autograd
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm
from scipy.optimize import minimize


def load_image(fname):
    img = Image.open(fname).resize((32, 32))
    img_gray = img.convert('L')
    img_eq = ImageOps.autocontrast(img_gray)
    img_eq = np.array(img_eq.getdata()).reshape((img_eq.size[1], -1))
    return img_eq


def binarize_image(img_eq):
    img_bin = np.copy(img_eq)
    img_bin[img_bin < 128] = -1
    img_bin[img_bin >= 128] = 1
    return img_bin


def add_corruption(img):
    img = img.reshape((32, 32))
    t = np.random.choice(3)
    if t == 0:
        i = np.random.randint(32)
        img[i:(i + 8)] = -1
    elif t == 1:
        i = np.random.randint(32)
        img[:, i:(i + 8)] = -1
    else:
        mask = np.sum([np.diag(-np.ones(32 - np.abs(i)), i)
                       for i in np.arange(-4, 5)], 0).astype(np.int)
        img[mask == -1] = -1
    return img.ravel()


def learn_hebbian(imgs):
    img_size = np.prod(imgs[0].shape)
    ######################################################################
    ######################################################################
    weights = np.zeros((img_size, img_size))
    bias = np.zeros(img_size)
    # Complete this function
    # You are allowed to modify anything between these lines
    # Helper functions are allowed
    
    #flatten image
    imgs_f = np.reshape(imgs,(len(imgs),img_size))
    
    for img in imgs_f:
        outer = np.outer(img,img)
        weights += outer
    diagW = np.diag(np.diag(weights))
    weights = weights - diagW
    weights /= len(imgs)
    
    
    #######################################################################
    #######################################################################
    return weights, bias


def sigma(x):
  return 1/(1+exp(-x))

from scipy.optimize import minimize

def learn_maxpl(imgs):
    
    img_size = np.prod(imgs[0].shape)
    ######################################################################
    ######################################################################
    weights = np.zeros((img_size, img_size))
    bias = np.zeros(img_size)
    # Complete this function
    # You are allowed to modify anything between these lines
    # Helper functions are allowed
    
    # Avoid "overflow encountered in exp"
    old_settings = np.seterr(all='ignore')

    # Define log PseudoLikelihood function
    def neg_logPL(W,b):
      SUM=0
      for img in imgs:
        #flatten image
        img_f = np.reshape(img,(1,img_size))
        for i in range(len(img_f[0])):
          x=img_f[0][i]
          X=np.copy(img_f)
          X[0][i]=0
          if x==1:  
            SUM=SUM+np.log(1/(1+np.exp(-np.sum(W[i]*X[0])+b[i])))
          else:
            SUM=SUM+np.log(1-1/(1+np.exp(-np.sum(W[i]*X[0])+b[i]))) 
      return -SUM
    
    # Gradient descent on neg_logPL
    neg_logPL_dW=grad(neg_logPL,0)
    neg_logPL_db=grad(neg_logPL,1)
    W=np.zeros((img_size,img_size))
    b=np.zeros((img_size))
    n_iteration=5
    alpha=0.01
    for i in range(n_iteration):
      dW=neg_logPL_dW(W,b)
      db=neg_logPL_db(W,b)
      W=W-(dW+np.transpose(dW))*alpha
      b=b-db*alpha
    weights, bias = W,b
    
    #######################################################################
    #######################################################################
    
    return weights, bias


def plot_results(imgs, cimgs, rimgs, fname='result.png'):
    '''
    This helper function can be used to visualize results.
    '''
    img_dim = 32
    assert imgs.shape[0] == cimgs.shape[0] == rimgs.shape[0]
    n_imgs = imgs.shape[0]
    fig, axn = plt.subplots(n_imgs, 3, figsize=[8, 8])
    for j in range(n_imgs):
        axn[j][0].axis('off')
        axn[j][0].imshow(imgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 0].set_title('True')
    for j in range(n_imgs):
        axn[j][1].axis('off')
        axn[j][1].imshow(cimgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 1].set_title('Corrupted')
    for j in range(n_imgs):
        axn[j][2].axis('off')
        axn[j][2].imshow(rimgs[j].reshape((img_dim, img_dim)), cmap='Greys_r')
    axn[0, 2].set_title('Recovered')
    fig.tight_layout()
    plt.savefig(fname)


def recover(cimgs, W, b):
    img_size = np.prod(cimgs[0].shape)
    ######################################################################
    ######################################################################
    rimgs = []
    # Complete this function
    # You are allowed to modify anything between these lines
    # Helper functions are allowed
    #######################################################################
    #######################################################################
    rimgs = cimgs.copy()
    num_iter = 20
    for i in range(num_iter):
        for j in range(len(rimgs)):
            rimgs[j] = -((np.sign(1/(1+np.exp(W.dot(rimgs[j])+b))-0.5))).astype(int)
    rimgs = rimgs.reshape((len(rimgs),32,32))
    return rimgs


def main():
    # Load Images and Binarize
    ifiles = sorted(glob.glob('C:/Users/User/OneDrive/NUS/course/CS5340/Assignment_2/images/*'))
    timgs = [load_image(ifile) for ifile in ifiles]
    imgs = np.asarray([binarize_image(img) for img in timgs])

    # Add corruption
    cimgs = []
    for i, img in enumerate(imgs):
        cimgs.append(add_corruption(np.copy(imgs[i])))
    cimgs = np.asarray(cimgs)

    # Recover 1 -- Hebbian
    Wh, bh = learn_hebbian(imgs)
    rimgs_h = recover(cimgs, Wh, bh)
    np.save('hebbian.npy', rimgs_h)
    plot_results(imgs, cimgs, rimgs_h, fname='C:/Users/User/OneDrive/NUS/course/CS5340/Assignment_2/result_1.png') #C:/Users/User/OneDrive/NUS/course/CS5340/Assignment_2/

    # Recover 2 -- Max Pseudo Likelihood
    Wmpl, bmpl = learn_maxpl(imgs)
    rimgs_mpl = recover(cimgs, Wmpl, bmpl)
    np.save('mpl.npy', rimgs_mpl)
    plot_results(imgs, cimgs, rimgs_mpl, fname='C:/Users/User/OneDrive/NUS/course/CS5340/Assignment_2/result_2.png')

if __name__ == '__main__':
    main()
