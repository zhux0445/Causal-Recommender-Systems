import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.sparse as spLA

from numba import jit
from numpy import linalg as LA
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def simulate_data(corr, gamma, n_users, n_items, K):
    theta_A = npr.gamma(0.3, scale=0.3, size=(n_users, K))
    beta = npr.gamma(0.3, scale=0.3, size=(n_items, K))
    A = np.minimum(npr.poisson(theta_A.dot(beta.T)), 1)
    theta_Y = corr * theta_A + (1 - corr) * npr.gamma(0.3, scale=0.3, size=(n_users, K))
    y = npr.poisson(theta_Y.dot(beta.T) + gamma * theta_A.dot(beta.T))
    y = np.minimum(y+1, 5)
    y_obs = np.multiply(A, y)
    y = sparse.coo_matrix(y)
    y_obs = sparse.coo_matrix(y_obs)
    A = sparse.coo_matrix(A)
    ydf = pd.DataFrame({'uid': y.row, 'sid': y.col, 'rating':y.data})
    ydf_obs = pd.DataFrame({'uid': y_obs.row, 'sid': y_obs.col, 'rating':y_obs.data})
    Adf = pd.DataFrame({'uid': A.row, 'sid': A.col, 'obs':A.data})
    return ydf, ydf_obs, Adf

def load_data(df, shape, colnames=["uid", "sid", "rating"]):
    user, item, rating = colnames[0], colnames[1], colnames[2]
    rows, cols, vals = np.array(df[user]), np.array(df[item]), np.array(df[rating])
    data = csr_matrix((vals, (rows, cols)), dtype=np.float32, shape=shape)
    return data

def exp_to_imp(data, cutoff=1e-10):
    data_imp = data.copy()
    data_imp.data[data_imp.data < cutoff] = 0
    data_imp.data[data_imp.data >= cutoff] = 1
    data_imp.data = data_imp.data.astype('int32')
    data_imp.eliminate_zeros()
    return data_imp

@jit(nopython=True)
def calculate_loss(U, V, R, I, ahat, gamma, lamb):
    s=0
    for i in range(len(U)):
        for j in range(len(V)):
            s=s+I[i][j]*(R[i][j] - np.dot(U[i],V[j]) - ahat[i][j]*gamma[i])**2
    k = LA.norm(U)**2
    m = LA.norm(V)**2
    return s+lamb*(m+k)

def estimate_latent_features(R, I, a_hat, n_users, n_items, K, lamb, n_iters):
    loss = []
    V = csr_matrix((n_items, K))
    U = csr_matrix(np.zeros((n_users, K)) + 0.5)
    gamma = np.zeros(n_users)+0.3
    
    loss.append(calculate_loss(U.toarray(), V.toarray(), R, I, a_hat, gamma, lamb))
    
    for j in range(n_iters):
        for i in range(n_items):
            U_i = U.toarray()[R[:, i] > 0,:]
            A = np.matmul(U_i.T,U_i) + lamb * np.eye(K)
            A = csr_matrix(A)
            B = np.matmul(U_i.T,R[ R[:,i] > 0,i])-np.matmul(U_i.T,np.multiply(gamma[ R[:,i] > 0],a_hat[ R[:,i] > 0,i]))
            B = csr_matrix(B)
            V[i,:] = spsolve(A, csr_matrix.transpose(B))
            
        for u in range(n_users):
            V_u = V.toarray()[R[u, :] > 0,:]
            A = np.matmul(V_u.T,V_u)+lamb*np.identity(K)
            B = np.matmul(V_u.T,R[u, R[u, :] > 0])-gamma[u]*np.matmul(V_u.T,a_hat[u, R[u, :] > 0])
            A = csr_matrix(A)
            B = csr_matrix(B)
            U[u,:] = spsolve(A, csr_matrix.transpose(B))
            gamma[u]=(np.dot(R[u, R[u, :] > 0],a_hat[u, R[u, :] > 0])-np.dot(np.matmul(U.toarray()[u],V_u.T),a_hat[u, R[u, :] > 0]))/np.dot(a_hat[u, R[u, :] > 0],a_hat[u, R[u, :] > 0])
            
        loss.append(calculate_loss(U.toarray(), 
                                   V.toarray(), 
                                   R, 
                                   I, 
                                   a_hat, 
                                   gamma,lamb))
    
    return (loss, U, V, gamma)

def estimate_ratings(U, V, gamma, a_hat, I):
    R_hat = np.zeros(a_hat.shape)
    for i in range(len(gamma)):
        R_hat[i,:] = gamma[i]*a_hat[i]
    R_hat += np.matmul(U.toarray(), V.toarray().T)
    return R_hat