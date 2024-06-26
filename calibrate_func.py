#!/usr/bin/env python

import sys, struct, os.path
import numpy as np
import loadh5 as lh
from scipy.linalg import svd, eigh
import multiprocessing as mp
import time


def Cov2Eig(Cov, ant_flag=[]):
    '''
    decompose the covariance matrix with Eigh (hermitian version of eig)
    input:
        Cov, shape: (nAnt, nAnt, nchan)

    return:
        W, shape: (nchan, nAnt)         # eigen values in ascending order
        V, shape: (nchan, nAnt, nAnt)   # column vectors used to map from eigen-space
    '''
    (nAnt, nAnt, nchan) = Cov.shape
    V = np.zeros((nchan, nAnt, nAnt), dtype=complex)
    W = np.zeros((nchan, nAnt))

    for i in range(nchan):
        w,v = eigh(Cov[:,:,i])
        W[i] = w
        V[i] = v

    # zero out weak modes equal to the number of flagged inputs
    for ai in range(len(ant_flag)):
        W[:,ai] = 0.

    return W, V


def Cov2SVD(Cov):
    '''
    decompose the covariance matrix with SVD
    input:
        Cov, shape: (nAnt, nAnt, nchan)

    return:
        U, shape: (nchan, nAnt, nAnt)   # column vectors used to inverse-map to original space
        S, shape: (nchan, nAnt)         # sigular values in descending order
                                        # since Cov is Hermitian, S is also the eigenvalues
        Vh, shape: (nchan, nAnt, nAnt)  # row vectors used to map to eigen-space
    '''
    (nAnt, nAnt, nchan) = Cov.shape
    U = np.zeros((nchan, nAnt, nAnt), dtype=complex)
    Vh = np.zeros((nchan, nAnt, nAnt), dtype=complex)
    S = np.zeros((nchan, nAnt))

    for i in range(nchan):
        u,s,vh = svd(Cov[:,:,i])
        S[i] = s
        U[i] = u
        Vh[i] = vh

    return U, S, Vh


def genCov(i1, i2, ftdata):    # internal function for makeCov
    print('in genCov:', i1, i2)
    nAnt = ftdata.shape[0]
    #arr1 = ftdata[:,:,i1:i2]
    #arr2 = nftdata[:,:,i1:i2]
    nTmp = i2-i1
    tmpCov = np.zeros((nAnt, nAnt, nTmp), dtype=complex)
    for ai in range(nAnt):
        tmpCov[ai,ai,:] = 1+0j
    for ai in range(nAnt-1):
        for aj in range(ai+1,nAnt):
            y = ftdata[ai] * ftdata[aj].conjugate()
            tmpCov[ai,aj] = y.mean(axis=0)
            tmpCov[aj,ai] = tmpCov[ai,aj].conjugate()
    return i1, i2, tmpCov


def makeCov(ftdata, scale=False, coeff=True, ant_flag=[], bandpass=False, nPool=1):
    '''
    generate covariance matrix from the waterfall spectra
    input:
        ftdata, shape: (nAnt, nframe, nchan)

    optional:
        scale: whether to scale the amplitude among antennas
        coeff: whether to comput covariance on correlation coefficients
        (note: choose none or one of the above, not both.)
        ant_flag: a list of antennas that should be flagged
        bandpass: whether to normalize the bandpass

    output:
        Cov, shape: (nAnt, nAnt, nChan)
        norm, shape: (nAnt, nChan)
    '''
    print('in makeCov, nPool:', nPool)

    ftdata = np.ma.asarray(ftdata)
    (nAnt, nFrame, nChan) = ftdata.shape

    #ftdata -= np.ma.median(ftdata, axis=1, keepdims=True)
    mr = np.ma.median(ftdata.real, axis=1, keepdims=True)
    mi = np.ma.median(ftdata.imag, axis=1, keepdims=True)
    nftdata = ftdata - (mr + 1j*mi)
    norm2 = np.ma.abs(nftdata)
    norm = norm2.mean(axis=1)
    avg_norm = norm.mean(axis=0, keepdims=True)


    if (coeff):
        bandpass = True
        # when using coeff, the output norm is approximated by the bandpass version

    if (bandpass):
        scale = True
        # norm contains bandpass and antenna relative gain
    else:
        norm /= avg_norm
        # norm contains only antenna relative gain
    
    if (scale):
        nftdata /= norm.reshape((nAnt, 1, nChan))

    cftdata = ftdata / np.ma.abs(ftdata)    # for coefficient

    Cov = np.zeros((nAnt, nAnt, nChan), dtype=complex)

    nChunk = nChan//nPool
    if (nChan != nPool*nChunk):
        print('warinng: inconsistent channel split in makeCov!')

    star_args = []
    for i in range(nPool):
        i1 = nChunk*i
        i2 = nChunk*(i+1)
        if (coeff):
            arr1 = cftdata[:,:,i1:i2]
        else:
            arr1 = nftdata[:,:,i1:i2]
        star_args.append((i1,i2,arr1))
    #print(star_args)

    with mp.Pool() as pool:
        for result in pool.starmap(genCov, star_args):
            i1, i2, tmpCov = result
            Cov[:,:,i1:i2] = tmpCov

    for ai in ant_flag:
        Cov[ai] = 0.j
        Cov[:,ai] = 0.j
        norm[ai] = 0.
        norm.mask[ai] = True

    return Cov, norm


def streamFilter(data, rfi_mask):
    '''
    remove the RFI contribution from the data stream using FFT-IFFT
    input:
        data is a 1D stream of complex (I,Q) samples
        rfi_mask is a 1D boolean mask of nchan (True-->bad, False-->good)
        note:: the channel order is assumed to be monotonic, i.e. after fftshift

    output:
        filt_data is also a 1D stream of complex (I,Q) samples
    '''
    nchan = len(rfi_mask)           # rfi_mask.shape = (nchan,); channel_mask
    ftdata = maskedFFT(data, nchan) # fftshift is enabled to be consistent with rfi_mask
    fft_mask = ftdata.mask         # shape: (nframe, nchan)
                                    # these are frame_mask (same value for each frame)

    ftdata[:,rfi_mask] *= 0.        # zero-out the RFI channels, regardless of the frame_mask
    tmp = np.fft.fftshift(ftdata, axes=1)   # unshift
    tmp = np.fft.ifft(tmp, axis=1)  # shape: (nframe, nchan)
    tmp = np.ma.array(tmp, mask=fft_mask)

    return tmp.flatten()


def maskedFFT(data, nchan=1024, shift=True):
    '''
    transform 1D stream of masked array data
    to 2D waterfall spectrum
    if any sample in a given FFT frame is masked,
    mask the entire frame
    '''
    #print('maskedFFT:', data.shape, data.mask.shape)
    tmp = data.flatten()
    nlen  = len(tmp)
    nlen2 = nlen//nchan*nchan
    #print('maskedFFT:', nlen, nlen2)
    tmp = tmp[:nlen2].reshape((-1,nchan))           # shape: (nframe, nchan)
    #print('maskedFFT:', tmp.shape, tmp.mask.shape)
    frame_mask = np.any(tmp.mask, axis=1)   # shape: (nframe,)
    tmp = np.fft.fft(tmp, axis=1)
    if (shift):
        tmp = np.fft.fftshift(tmp, axes=1)
    fft_mask = np.logical_or(np.zeros_like(tmp, dtype=bool), frame_mask.reshape(-1,1))
    tmp = np.ma.array(tmp, mask=fft_mask)
    #print('maskedFFT:', np.count_nonzero(tmp.mask))

    return tmp  # shape: (nframe, nchan)


