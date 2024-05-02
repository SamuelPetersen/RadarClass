
import numpy as np
import scipy as spy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def ambiguity_function(u, Tp, fs, T=1, N=100, F=1, K=100):
    '''
    %{
      plots ambiguity function of a complex signal u
            returns a plot of quadrants 1 and 2 of the ambiguity function of a signal
            The ambiguity function is defined as:

             a(t,f) = abs ( sum( u(k)*u'(i-t)*exp(j*2*pi*f*i) ) )

        - Tp is the normalization period of the ambiguity function
            for a single pulse use the pulse width
            for a pulse train normalize by the PRI

        - fs is the sample frequency of the complex input signal (u)
        - F is the maximal Dopler shift
        - T is the maximal Delay
        - K is the number of positive Doppler shifts (grid points)
        - N is the number of delay shifts on each side (for a total of 2N+1 points)

     %}'''
    m_basic = len(u)
    df = F / K / Tp

    dt = 1 /fs
    m = m_basic
    r = int(np.ceil((N+1) / T / m_basic))

    if r > 1:
        dt = dt / r
        ud = np.diag(u)
        ao = np.ones((r, m_basic))
        m = m_basic * r
        u = np.reshape(np.dot(ao, ud).T, (m,))

    phas = np.angle(u)
    uamp = np.abs(u)

    t = np.r_[0: r * m_basic] * dt
    tscale1 = np.hstack(([0], np.r_[0:r * m_basic], [r*m_basic])) * dt
    dphas = np.hstack(([np.NaN], np.diff(np.unwrap(phas)))) * r / 2/ np.pi


    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(tscale1, np.hstack(([0], np.abs(uamp), [0])).T, linewidth=1.5)
    axs[0].set_ylabel('Amplitude', fontsize=10, fontweight='bold')
    axs[0].set_ylim(0, 1.2*np.max(abs(uamp)))

    axs[1].plot(t, phas.T, linewidth=1.5)
    axs[1].set_ylabel('Phase [rad]', fontsize=10, fontweight='bold')
    axs[1].set_ylim(-np.pi, np.pi)

    axs[2].plot(t, dphas.T*np.ceil(max(t)), linewidth=1.5)
    axs[2].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    # axs[2].set_ylim(0, 1.2*np.max(abs(uamp)))

    # calculate a delay vector with N+1 points that spans from zero delay to
    # ceil(T*t(m))
    # notice that the delay vector does not have to be equally spaced but must
    # have all
    # entries as integer multiples of dt
    dtau = np.ceil(T * m) * dt / N
    tau = np.round(np.r_[0:N] * dtau / dt) * dt

    # calculate K+1 equally spaced grid points of Doppler axis with df spacing
    f = np.r_[0:K] * df

    # duplicate Doppler axis to show also negative Doppler’s (0 Doppler is
    # calculated twice)
    f = np.concatenate((-f[::-1], f))

    # calculate ambiguity function using sparse matrix manipulations (no loops)
    # define a sparse matrix based on the signal samples u1 u2 u3 ... um
    # with size m+ceil(T*m) by m (notice that u’ is the conjugate transpose of u)
    # where the top part is diagonal (u*) on the diagonal and the bottom part is a
    # zero matrix
    mat1 = np.concatenate((np.diag(np.conj(u)), np.zeros((int(np.ceil(T*m)), m))))

    '''
    % define a convolution sparse matrix based on the signal samples u1 u2 u3 ...
    % um
    % where each row is a time(index) shifted versions of u.
    % each row is shifted tau/dt places from the first row
    % the minimal shift (first row) is zero
    % the maximal shift (last row) is ceil(T*m) places
    % the total number of rows is N+1
    % number of columns is m+ceil(T*m)
    % for example, when tau/dt=[0 2 3 5 6] and N=4
    %
    % [u1 u2 u3 u4 ... ... um 0 0 0 0 0 0]
    % [ 0 0 u1 u2 u3 u4 ... ... um 0 0 0 0]
    % [ 0 0 0 u1 u2 u3 u4 ... ... um 0 0 0]
    % [ 0 0 0 0 0 u1 u2 u3 u4 ... ... um 0]
    % [ 0 0 0 0 0 0 u1 u2 u3 u4 ... ... um]
    % define a row vector with ceil(T*m)+m+ceil(T*m) places by padding u with zeros
    % on both sides
    '''
    u_padded = np.concatenate((np.zeros((int(np.ceil(T*m)),)), u, np.zeros((int(np.ceil(T*m)),))))

    # define column indexing and row indexing vectors
    cidx = np.r_[0:m + int(np.ceil(T*m))]
    ridx = np.round(tau / dt)

    #define indexing matrix with Nused+1 rows and m+ceil(T*m) columns
    #where each element is the index of the correct place in the padded version
    #ofu
    index = np.tile(cidx, (N, 1)) + np.tile(ridx.reshape(len(ridx), 1), (1, m+int(np.ceil(T*m))))

    #calculate matrix
    temp = u_padded[index.reshape(np.size(index),).astype(int)].reshape(np.shape(index))
    mat2 = spy.sparse.coo_matrix(temp)

    uu_pos = mat2 @ mat1
    # plt.figure()
    # plt.imshow(np.abs(uu_pos))
    del mat1, mat2

    '''    
    % calculate exponent matrix for full calculation of ambiguity function.
    % The exponent
    % matrix is 2*(K+1) rows by m columns where each row represents a possible
    % Doppler and
    % each column stands for a different place in u.'''
    e = np.exp(-1j * 2 * np.pi * f.reshape((len(f), 1)) * t)

    '''    % calculate ambiguity function for positive delays by calculating the integral
    % for each
    % possible delay and Doppler over all entries in u.
    % a_pos has 2*(K+1) rows (Doppler) and N+1 columns (Delay)'''
    a_pos = np.abs(e @ np.conj(uu_pos.T))

    # normalize ambiguity function to have a maximal value of 1
    a_pos = a_pos / np.max(np.max(a_pos))

    # use the symmetry properties of the ambiguity function to transform the
    # negative Doppler
    # positive delay part to negative delay, positive Doppler
    a = np.concatenate((np.flipud(a_pos[0:K, :]), np.fliplr(a_pos[K:2*K, :])), axis=1)

    # define new delay and Doppler vectors
    # Normalized Delay by the 1/PRI
    delay = np.concatenate((-tau[::-1], tau)) / Tp

    #Normalize the Freq by th PRF
    freq = f[K:2*K] * Tp

    #excludes the zero Delay that was taken twice
    delay = np.concatenate((delay[0:N], delay[N:2*N]))
    a = a[:, np.hstack((np.r_[0:N], np.r_[N:2*N]))]

    # plot the ambiguity function and autocorrelation cut
    amf, amt = np.shape(a)

    X = np.vstack((np.NaN * np.zeros((1, amt)), a))
    tau = delay
    v = np.hstack((0, freq))

    TT, VV = np.meshgrid(tau, v)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(TT, VV, X, color='royalblue')

    surf0 = ax.plot_surface(np.vstack((tau, tau)), np.zeros((2, amt)), np.vstack((np.zeros((1, amt)), a[0, :])), color='royalblue')
    ax.set_xlabel('t / T', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'$\nu$ * T', fontsize=10, fontweight='bold')
    ax.set_zlabel(r'|$\chi$($\tau$,$\nu$)|',fontsize=10, fontweight='bold')

    # Tt0, Vv0 = np.meshgrid(tau, freq)
    # ax.contourf(Tt0, Vv0, a, zdir='y', offset=F, alpha=1, cmap=cm.get_cmap('jet_r'))
    # ax.contourf(Tt0, Vv0, a, zdir='x', offset=np.round(N * dtau / dt) * dt / Tp, alpha=1, cmap=cm.get_cmap('jet_r'))
    #

    # fig, ax = plt.subplots()
    # ax.contourf(TT, VV, X)

    return