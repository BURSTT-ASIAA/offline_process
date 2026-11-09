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
        #tmpCov[ai,ai,:] = 1+0j
        tmpCov[ai,ai] = (np.abs(ftdata[ai])**2).mean()
    for ai in range(nAnt-1):
        for aj in range(ai+1,nAnt):
            y = ftdata[ai] * ftdata[aj].conjugate()
            tmpCov[ai,aj] = y.mean(axis=0)
            tmpCov[aj,ai] = tmpCov[ai,aj].conjugate()
    return i1, i2, tmpCov


def makeCov(ftdata, scale=False, coeff=True, ant_flag=[], bandpass=False, nPool=1):
    '''
    generate covariance matrix from the waterfall spectra using the legacy pool implementation
    input:
        ftdata, shape: (nAnt, nframe, nchan)

    optional:
        scale: whether to scale the amplitude among antennas
        coeff: whether to compute covariance on correlation coefficients
        ant_flag: a list of antennas that should be flagged
        bandpass: whether to normalize the bandpass
        nPool: number of channel partitions sent to the worker pool

    output:
        Cov, shape: (nAnt, nAnt, nChan)
        norm, shape: (nAnt, nChan)
    '''
    print('in makeCov, nPool:', nPool)

    ftdata = np.ma.asarray(ftdata)
    (nAnt, nFrame, nChan) = ftdata.shape

    mr = np.ma.median(ftdata.real, axis=1, keepdims=True)
    mi = np.ma.median(ftdata.imag, axis=1, keepdims=True)
    nftdata = ftdata - (mr + 1j*mi)
    norm2 = np.ma.abs(nftdata)
    norm = norm2.mean(axis=1)
    avg_norm = norm.mean(axis=0, keepdims=True)

    if coeff:
        bandpass = True
    if bandpass:
        scale = True
    else:
        norm /= avg_norm

    if scale:
        nftdata /= norm.reshape((nAnt, 1, nChan))

    cftdata = ftdata / np.ma.abs(ftdata)
    Cov = np.zeros((nAnt, nAnt, nChan), dtype=complex)

    nChunk = nChan//nPool
    if nChan != nPool*nChunk:
        print('warinng: inconsistent channel split in makeCov!')

    star_args = []
    for i in range(nPool):
        i1 = nChunk*i
        i2 = nChunk*(i+1)
        arr1 = cftdata[:,:,i1:i2] if coeff else nftdata[:,:,i1:i2]
        star_args.append((i1,i2,arr1))

    with mp.Pool() as pool:
        for i1, i2, tmpCov in pool.starmap(genCov, star_args):
            Cov[:,:,i1:i2] = tmpCov

    for ai in ant_flag:
        Cov[ai] = 0.j
        Cov[:,ai] = 0.j
        norm[ai] = 0.
        norm.mask[ai] = True

    return Cov, norm


def makeCov2(ftdata, scale=False, coeff=True, ant_flag=[], bandpass=False, nPool=1, chanBlock=32, dtype=np.complex64):
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
        nPool: number of channel partitions (the diagonal is averaged within each partition)
        chanBlock: number of channels processed together
        dtype: precision of the matrix multiply (complex64 is ~2x faster than complex128)

    output:
        Cov, shape: (nAnt, nAnt, nChan)
        norm, shape: (nAnt, nChan)

    note:
        each channel block is copied to a contiguous (chan, ant, frame) array so that
        the matrix multiply goes to BLAS; strided or masked arrays fall back to a slow loop
    '''
    print('in makeCov2, channel block:', chanBlock)

    data = np.ma.getdata(ftdata)
    mask = np.ma.getmaskarray(ftdata)
    (nAnt, nFrame, nChan) = data.shape

    if (coeff):
        bandpass = True
        # when using coeff, the output norm is approximated by the bandpass version
    if (bandpass):
        scale = True
        # norm contains bandpass and antenna relative gain

    if nPool < 1:
        raise ValueError('nPool must be at least 1')
    if chanBlock < 1:
        raise ValueError('chanBlock must be at least 1')
    nPool = min(int(nPool), nChan)
    boundaries = np.linspace(0, nChan, nPool+1, dtype=int)
    ant_index = np.arange(nAnt)

    Cov = np.zeros((nAnt, nAnt, nChan), dtype=complex)
    norm = np.ma.masked_all((nAnt, nChan))

    for partition in range(nPool):
        part_start = boundaries[partition]
        part_stop = boundaries[partition+1]
        diagonal_sum = np.zeros(nAnt, dtype=float)
        diagonal_count = np.zeros(nAnt, dtype=np.int64)

        for i1 in range(part_start, part_stop, chanBlock):
            i2 = min(i1+chanBlock, part_stop)
            # contiguous copies, shape (nb, nAnt, nFrame)
            x = np.ascontiguousarray(data[:,:,i1:i2].transpose(2,0,1))
            valid = ~np.ascontiguousarray(mask[:,:,i1:i2].transpose(2,0,1))
            partial = not (valid.all(axis=2) | ~valid.any(axis=2)).all()

            # median subtraction and normalization (masked samples excluded)
            if (partial):
                xm = np.ma.array(x, mask=~valid)
                med = np.ma.median(xm.real, axis=2, keepdims=True).filled(0.) \
                    + 1j*np.ma.median(xm.imag, axis=2, keepdims=True).filled(0.)
            else:   # each (chan, ant) is either fully valid or fully masked
                med = np.median(x.real, axis=2, keepdims=True) + 1j*np.median(x.imag, axis=2, keepdims=True)
            nx = x - med
            cnt = valid.sum(axis=2)
            nrm = np.where(valid, np.abs(nx), 0.).sum(axis=2)
            nrm = np.ma.array(nrm/np.maximum(cnt,1), mask=(cnt==0))    # (nb, nAnt)
            if (not bandpass):
                nrm /= nrm.mean(axis=1, keepdims=True)
                # norm contains only antenna relative gain
            norm[:,i1:i2] = nrm.T

            if (coeff):
                ax = np.abs(x)
                valid &= (ax > 0)
                u = np.divide(x, ax, out=np.zeros(x.shape, dtype=dtype), where=valid)
            else:
                if (scale):
                    s = nrm.filled(0.)[:,:,None]
                    valid &= (s > 0)    # zero norm is masked, as in the masked-array version
                    nx = np.divide(nx, s, out=np.zeros_like(nx), where=valid)
                u = np.where(valid, nx, 0.).astype(dtype)

            diagonal_sum += (np.abs(u)**2).sum(axis=(0,2))
            diagonal_count += valid.sum(axis=(0,2))

            sums = np.matmul(u, u.conj().transpose(0,2,1))
            if (valid.all()):
                cov_block = sums / nFrame
            else:
                vf = valid.astype(np.float32)
                counts = np.matmul(vf, vf.transpose(0,2,1))
                cov_block = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            Cov[:,:,i1:i2] = cov_block.transpose(1,2,0)

        diagonal = np.divide(diagonal_sum, diagonal_count,
                out=np.zeros_like(diagonal_sum), where=diagonal_count > 0)
        Cov[ant_index,ant_index,part_start:part_stop] = diagonal[:,None]

    for ai in ant_flag:
        Cov[ai] = 0.j
        Cov[:,ai] = 0.j
        norm[ai] = 0.
        norm[ai] = np.ma.masked

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


