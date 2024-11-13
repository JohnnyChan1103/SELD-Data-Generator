import numpy as np 
import pyroomacoustics as pra
import scipy


def sph2cart(azimuth, elevation, r, type='degree'):
    r"""
    Convert spherical to cartesian coordinates
    """
    assert type in ['degree', 'radian'], "Type must be 'degree' or 'radian'"
    if type == 'degree':
        azimuth = azimuth / 180.0 * np.pi
        elevation = elevation / 180.0 * np.pi

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return np.c_[x, y, z]


def cart2sph(x, y, z, type='degree'):
    r"""
    Convert cartesian to spherical coordinates    
    """
    assert type in ['degree', 'radian'], "Type must be 'degree' or 'radian'"

    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    if type == 'degree':
        azimuth = azimuth / np.pi * 180.0
        elevation = elevation / np.pi * 180.0

    return np.c_[azimuth, elevation, r]


def doa_estimate(
    mic_pos, receiver, fs=24000, nfft=256, num_src=2, freq_bins=np.arange(5,125), dim=3, c=343.0
    ):
    r"""
    Estimate DOA from signals
    """

    ################################
    # Compute the STFT frames needed
    X = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in receiver
        ]
    )

    ##############################################
    # Now we can test all the algorithms available
    # algo_names = sorted(pra.doa.algorithms.keys())
    algo_names = ["SRP", "NormMUSIC"]
    for algo_name in algo_names:
        # Construct the new DOA object
        doa = pra.doa.algorithms[algo_name](mic_pos, fs, nfft, dim=dim, c=c, num_src=num_src)

        # this call here perform localization on the frames in X
        doa.locate_sources(X, freq_bins=freq_bins)

        # doa.azimuth_recon contains the reconstructed location of the source
        print(algo_name)
        print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
        if dim == 3:
            print("  Recovered elevation:", - doa.colatitude_recon / np.pi * 180.0 + 90.0, "degrees")


def asarray_1d(a, **kwargs):
    r"""Squeeze the input and check if the result is one-dimensional.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.
    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result


def repeat_per_order(c):
    r"""Repeat each coefficient in 'c' m times per spherical order n.

    Parameters
    ----------
    c : (N,) array_like
        Coefficients up to SH order N.

    Returns
    -------
    c_reshaped : ((N+1)**2,) array like
        Reshaped input coefficients.
    """
    c = asarray_1d(c)
    N = len(c) - 1
    return np.repeat(c, 2*np.arange(N+1)+1)


def segment_mixtures(signal, fs, start, end, clip_length=5):
    r"""
    If the duration of the signal is less than 5 seconds, pad the signal with zeros at the beginning and
    end. Otherwise, return the first 5 seconds of the signal
    """
    
    duration = np.shape(signal)[0] / fs
    if duration < clip_length:
        pad_width_before = int(np.ceil(start * fs))
        # pad_width_after = max(0, int(np.ceil(fs*(clip_length-end))))
        pad_width_after = int(max(0, clip_length * fs - np.shape(signal)[0] - pad_width_before))
        pad_width = ((pad_width_before, pad_width_after),)
        signal =  np.pad(signal, pad_width)
    
    if len(signal) < clip_length * fs:
        print(signal.shape, start, end, duration, pad_width)
        raise ValueError('Length of audio is not equal to the mixture duration.')
    
    return signal[:int(clip_length*fs)]


def sample_from_quartiles(K, stats):
    r"""
    Uniformly sample K points from the quartiles of the distribution.
    
    :param K: number of sampling points
    :param stats: a list of the 5-number summary of the data. 
                  stat = [min, quart1, median, quart3, max].
    :return: a list of samples from the given data.
    """

    minn = stats[0]
    maxx = stats[4]
    quart1 = stats[1]
    mediann = stats[2]
    quart3 = stats[3]
    samples = minn + (quart1 - minn)*np.random.rand(K, 1)
    samples = np.append(samples,quart1)
    samples = np.append(samples, quart1 + (mediann-quart1)*np.random.rand(K,1))
    samples = np.append(samples,mediann)
    samples = np.append(samples, mediann + (quart3-mediann)*np.random.rand(K,1))
    samples = np.append(samples, quart3)
    samples = np.append(samples, quart3 + (maxx-quart3)*np.random.rand(K,1))
    
    return samples


def apply_event_gains(signal, duration, class_gains, class_idx):
    r"""Apply event gains of class_idx to the signal.
    """

    K=1000
    rand_energies_per_spec = sample_from_quartiles(K, class_gains[class_idx])
    # intr_quart_energies_per_sec = rand_energies_per_spec[K + np.arange(3*(K+1))]
    intr_quart_energies_per_sec = rand_energies_per_spec[K + np.arange(2*(K+1))]
    rand_energy_per_spec = intr_quart_energies_per_sec[np.random.randint(len(intr_quart_energies_per_sec))]
    sample_active_time = duration

    target_energy = rand_energy_per_spec*sample_active_time
    event_energy = np.sum(signal**2)
    norm_gain = np.sqrt(target_energy / (event_energy + 1e-10))

    return norm_gain * signal


def stft_ham(insig, winsize=256, fftsize=512, hopsize=128):
    nb_dim = len(np.shape(insig))
    lSig = int(np.shape(insig)[0])
    nCHin = int(np.shape(insig)[1]) if nb_dim > 1 else 1
    x = np.arange(0,winsize)
    nBins = int(fftsize/2 + 1)
    nWindows = int(np.ceil(lSig/(2.*hopsize)))
    nFrames = int(2*nWindows+1)
    
    winvec = np.zeros((len(x),nCHin))
    for i in range(nCHin):
        winvec[:,i] = np.sin(x*(np.pi/winsize))**2
    
    frontpad = winsize-hopsize
    backpad = nFrames*hopsize-lSig

    if nb_dim > 1:
        insig_pad = np.pad(insig,((frontpad,backpad),(0,0)),'constant')
        spectrum = np.zeros((nBins, nFrames, nCHin),dtype='complex')
    else:
        insig_pad = np.pad(insig,((frontpad,backpad)),'constant')
        spectrum = np.zeros((nBins, nFrames),dtype='complex')

    idx=0
    nf=0
    if nb_dim > 1:
        while nf <= nFrames-1:
            insig_win = np.multiply(winvec, insig_pad[idx+np.arange(0,winsize),:])
            inspec = scipy.fft.fft(insig_win,n=fftsize,norm='backward',axis=0)
            #inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
            inspec=inspec[:nBins,:]
            spectrum[:,nf,:] = inspec
            idx += hopsize
            nf += 1
    else:
        while nf <= nFrames-1:
            insig_win = np.multiply(winvec[:,0], insig_pad[idx+np.arange(0,winsize)])
            inspec = scipy.fft.fft(insig_win,n=fftsize,norm='backward',axis=0)
            #inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
            inspec=inspec[:nBins]
            spectrum[:,nf] = inspec
            idx += hopsize
            nf += 1
    
    return spectrum


def ctf_ltv_direct(sig, irs, ir_times, fs, win_size):
    convsig = []
    win_size = int(win_size)
    hop_size = int(win_size / 2)
    fft_size = win_size*2
    nBins = int(fft_size/2)+1
    
    # IRs
    ir_shape = np.shape(irs)
    sig_shape = np.shape(sig)
    
    lIr = ir_shape[0]

    if len(ir_shape) == 2:
        nIrs = ir_shape[1]
        nCHir = 1
    elif len(ir_shape) == 3:
        nIrs = ir_shape[2]
        nCHir = ir_shape[1]
    
    if nIrs != len(ir_times):
        return ValueError('Bad ir times')
    
    # number of STFT frames for the IRs (half-window hopsize)
    
    nIrWindows = int(np.ceil(lIr/win_size))
    nIrFrames = 2*nIrWindows+1
    # number of STFT frames for the signal (half-window hopsize)
    lSig = sig_shape[0]
    nSigWindows = np.ceil(lSig/win_size)
    nSigFrames = 2*nSigWindows+1
    
    # quantize the timestamps of each IR to multiples of STFT frames (hopsizes)
    tStamps = np.round((ir_times*fs+hop_size)/hop_size)
    
    # create the two linear interpolator tracks, for the pairs of IRs between timestamps
    nIntFrames = int(tStamps[-1])
    Gint = np.zeros((nIntFrames, nIrs))
    for ni in range(nIrs-1):
        tpts = np.arange(tStamps[ni],tStamps[ni+1]+1,dtype=int)-1
        ntpts = len(tpts)
        ntpts_ratio = np.arange(0,ntpts)/(ntpts-1)
        Gint[tpts,ni] = 1-ntpts_ratio
        Gint[tpts,ni+1] = ntpts_ratio
    
    # compute spectra of irs
    
    if nCHir == 1:
        irspec = np.zeros((nBins, nIrFrames, nIrs),dtype=complex)
    else:
        temp_spec = stft_ham(irs[:, :, 0], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
        irspec = np.zeros((nBins, np.shape(temp_spec)[1], nCHir, nIrs),dtype=complex)
    
    for ni in range(nIrs):
        if nCHir == 1:
            irspec[:, :, ni] = stft_ham(irs[:, ni], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
        else:
            spec = stft_ham(irs[:, :, ni], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
            irspec[:, :, :, ni] = spec#np.transpose(spec, (0, 2, 1))
    
    #compute input signal spectra
    sigspec = stft_ham(sig, winsize=win_size,fftsize=2*win_size,hopsize=win_size//2)
    #initialize interpolated time-variant ctf
    Gbuf = np.zeros((nIrFrames, nIrs))
    if nCHir == 1:
        ctf_ltv = np.zeros((nBins, nIrFrames),dtype=complex)
    else:
        ctf_ltv = np.zeros((nBins,nIrFrames,nCHir),dtype=complex)
    
    S = np.zeros((nBins, nIrFrames),dtype=complex)
    
    #processing loop
    idx = 0
    nf = 0
    inspec_pad = sigspec
    nFrames = int(np.min([np.shape(inspec_pad)[1], nIntFrames]))
    
    convsig = np.zeros((win_size//2 + nFrames*win_size//2 + fft_size-win_size, nCHir))
    
    while nf <= nFrames-1:
        #compute interpolated ctf
        Gbuf[1:, :] = Gbuf[:-1, :]
        Gbuf[0, :] = Gint[nf, :]
        if nCHir == 1:
            for nif in range(nIrFrames):
                ctf_ltv[:, nif] = np.matmul(irspec[:,nif,:], Gbuf[nif,:].astype(complex))
        else:
            for nch in range(nCHir):
                for nif in range(nIrFrames):
                    ctf_ltv[:,nif,nch] = np.matmul(irspec[:,nif,nch,:],Gbuf[nif,:].astype(complex))
        inspec_nf = inspec_pad[:, nf]
        S[:,1:nIrFrames] = S[:, :nIrFrames-1]
        S[:, 0] = inspec_nf
        
        repS = np.tile(np.expand_dims(S,axis=2), [1, 1, nCHir])
        convspec_nf = np.squeeze(np.sum(repS * ctf_ltv,axis=1))
        first_dim = np.shape(convspec_nf)[0]
        convspec_nf = np.vstack((convspec_nf, np.conj(convspec_nf[np.arange(first_dim-1, 1, -1)-1,:])))
        convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, norm='forward', axis=0)) ## get rid of the imaginary numerical error remain
        # convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, axis=0))
        #overlap-add synthesis
        convsig[idx+np.arange(0,fft_size),:] += convsig_nf
        #advance sample pointer
        idx += hop_size
        nf += 1
    
    convsig = convsig[(win_size):(nFrames*win_size)//2,:]
    
    return convsig