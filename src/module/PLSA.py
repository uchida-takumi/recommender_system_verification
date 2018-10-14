#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A implementation of Probabilistic Latent Semantic Analysis(PLSA).

This code reffere to:
    https://qiita.com/HZama/items/0957f74f8da1302f7652
"""
import sys
import numpy as np


class plsa(object):
    def __init__(self, Z, random_seed=None):
        """
        ARGUMENTs:
            Z [int]: the number of latent class.
            random_seed [int]: is set as np.random_seed(random_seed)
        """
        self.Z = Z
        np.random.seed(random_seed)

                
    def fit(self, N, k=200, t=1.0e-7, 
            hold_Pz=None, hold_Px_z=None, hold_Py_z=None):
        """
        train with EMalgorithm.
        
        ARGUMENTs:
            N [2dim np.array]: 
                the count matrix of (x, y).
            k [int]  : 
                the number of iteration.
            t [float]: 
                the threshold to breake iterations if implovment
                is bellow than t.    
            hold_Pz [None or np.array]:
                if set np.array as P(z), 
                EM-algorithm uses it as constant. 
            hold_Px_z [None or np.array]:
                if set np.array as P(x|z), 
                EM-algorithm uses it as constant. 
            hold_Py_z [None or np.array]:
                if set np.array as P(y|z), 
                EM-algorithm uses it as constant. 
        """
        
        # --- set up ----
        self.N = N
        self.X, self.Y = N.shape[0], N.shape[1]
        self.hold_Pz , self.hold_Px_z, self.hold_Py_z = hold_Pz, hold_Px_z, hold_Py_z
        
        # P(x)
        if self.hold_Pz is None:
            self.Pz  = np.random.rand(self.Z)
            self.Pz /= np.sum(self.Pz)
        else:
            self.Pz = hold_Pz
        
        # P(x|z)
        if self.hold_Px_z is None:
            self.Px_z = np.random.rand(self.Z, self.X)
            self.Px_z /= np.sum(self.Px_z, axis=1, keepdims=True)
        else:
            self.Px_z = hold_Px_z
        
        # P(y|z)
        if self.hold_Py_z is None:
            self.Py_z  = np.random.rand(self.Z, self.Y)
            self.Py_z /= np.sum(self.Py_z, axis=1, keepdims=True)
        else:
            self.Py_z = hold_Py_z
        
        
        # --- train ---
        prev_log_likelihood = sys.maxsize
        for i in range(k):
            self._e_step()
            self._m_step()
            this_log_likelihood = self._log_likelihood()
            
            if abs((this_log_likelihood - prev_log_likelihood) / prev_log_likelihood) < t:
                break
            
            prev_log_likelihood = this_log_likelihood
        
        self._set_Px_and_Py_as_attribute()
        
    
    def _e_step(self):
        """
        E-step
        update P(z|x,y). dimentions of Pz_xy is [x,y,z].
        """
        # the dimmention of the arrayes are [x, y, z] 
        self.Pz_xy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]
        ## add 1/sys.maxsize to avoid 0-divison error
        self.Pz_xy /= np.sum(self.Pz_xy, axis=2, keepdims=True) + 1/sys.maxsize

    def _m_step(self):
        """
        M-step
        update P(z), P(x|z), P(y|z)
        """
        NP = self.N[:, :, None] * self.Pz_xy

        if self.hold_Pz is None:
            self.Pz  = np.sum(NP, axis=(0,1))
            self.Pz /= np.sum(self.Pz)

        if self.hold_Px_z is None:            
            self.Px_z  = np.sum(NP, axis=1).T
            self.Px_z /= np.sum(self.Px_z, axis=1, keepdims=True)

        if self.hold_Py_z is None:
            self.Py_z  = np.sum(NP, axis=0).T
            self.Py_z /= np.sum(self.Py_z, axis=1, keepdims=True)
        
    def _set_Px_and_Py_as_attribute(self):
        """
        set P(x), P(y) to attributes of self.
        """
        NP = self.N[:, :, None] * self.Pz_xy

        self.Px  = np.sum(NP, axis=(1,2))
        self.Px /= np.sum(self.Px)

        self.Py  = np.sum(NP, axis=(0,2))
        self.Py /= np.sum(self.Py)
        
    def _log_likelihood(self):
        Pxy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]
        Pxy = np.sum(Pxy, axis=2)
        Pxy /= np.sum(Pxy)
        
        # add 1/sys.maxsize to avoid log(0) error
        return np.sum(self.N * np.log(Pxy + 1/sys.maxsize))
        
        

if __name__=='__main__':
    import numpy as np
    from src.PLSA import plsa
    
    N = np.array([
        [20, 23, 1, 4],
        [25, 19, 3, 0],
        [2, 1, 31, 28],
        [0, 1, 22, 17],
        [1, 0, 18, 24]
    ])
    
    plsa = plsa(2)
    plsa.fit(N)

    print('P(z)')
    print(plsa.Pz)
    print('P(x|z)')
    print(plsa.Px_z)
    print('P(y|z)')
    print(plsa.Py_z)
    print('P(z|x)')
    Pz_x = plsa.Px_z.T * plsa.Pz[None, :]
    print(Pz_x / np.sum(Pz_x, axis=1)[:, None])
    print('P(z|y)')
    Pz_y = plsa.Py_z.T * plsa.Pz[None, :]
    print(Pz_y / np.sum(Pz_y, axis=1)[:, None])
    
    