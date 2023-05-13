import math
import numpy as np
import torch

#TODO: 验证计算方法的的正确性()
def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)

def centering(K):
    device = K.device
    n = K.shape[0]
    unit = torch.ones([n, n]).to(device)
    I = torch.eye(n).to(device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.t())
    L_Y = torch.matmul(Y, Y.t())
    return torch.sum(centering(L_X) * centering(L_Y))



# def rbf(X, sigma=None):
#     GX = np.dot(X, X.T)
#     KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
#     if sigma is None:
#         mdist = np.median(KX[KX != 0])
#         sigma = math.sqrt(mdist)
#     KX *= - 0.5 / (sigma * sigma)
#     KX = np.exp(KX)
#     return KX
#
#
# def kernel_HSIC(X, Y, sigma):
#     return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))
#
#
# def linear_HSIC(X, Y):
#     L_X = np.dot(X, X.T)
#     L_Y = np.dot(Y, Y.T)
#     return np.sum(centering(L_X) * centering(L_Y))
#
#
# def linear_CKA(X, Y):
#     hsic = linear_HSIC(X, Y)
#     var1 = np.sqrt(linear_HSIC(X, X))
#     var2 = np.sqrt(linear_HSIC(Y, Y))
#
#     return hsic / (var1 * var2)
#
#
# def kernel_CKA(X, Y, sigma=None):
#     hsic = kernel_HSIC(X, Y, sigma)
#     var1 = np.sqrt(kernel_HSIC(X, X, sigma))
#     var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))
#
#     return hsic / (var1 * var2)



# if __name__=='__main__':
#     X = np.random.randn(100, 64)
#     Y = np.random.randn(100, 64)
#
#     print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
#     print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))
#
#     print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
#     print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))