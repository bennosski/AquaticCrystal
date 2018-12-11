
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
from numpy import *
from scipy.integrate import odeint
import matplotlib.animation as animation
import sympy as sp
import os, shutil

# system size
W = 25
buoy_spacing = 6
buoy_size = 5
Nmid = (buoy_spacing-1)*30 + 1
N = 2*W + Nmid

m1 = 1.0/6.5 
m2 = 8.0/6.5 

# masses
ms = m1*ones(N)
ms[W:N-W:buoy_spacing] = m2

wis = ones(N)
wis[W:N-W:buoy_spacing] = buoy_size
sum_wis = sum(wis)

#ts = linspace(0, 450, 750)
ts = linspace(0, 4, 10)

drive = zeros(N)
def F(arr, t, f):   
    x,v = arr[:N], arr[N:]
        
    right = roll(x, -1) - x
    right[-1] = 0
    left = roll(x, 1) - x
    left[0] = 0
    
    drive[0] = cos(f*t)
    
    #damping = -0.000 * v

    forces = left + right + drive # + damping
    forces -= sum(forces)/sum_wis # conserves water volume

    a = forces/ms
    
    return concatenate((v, a))

def solve(f): return odeint(F, zeros(2*N), ts, args=(f,))[:,:N]

def test1():
    print('\n\nTest1\n')

    fs = linspace(1.3, 1.7, 30)
    maxs = []
    for f in fs:
        x = solve(f)
        s1 = amax(x[:,-W:])/amax(x[:,:W])
        s2 = std(x[:,-W:])/std(x[:,:W])
        s3 = std(x[:,-W:])
        maxs.append([s1,s2,s3])
        print('f=%1.3f scores=%1.3f, %1.3f, %1.3f'%((f,)+tuple(maxs[-1])))

    maxs = array(maxs)
    figure()
    for i in range(3):
        plot(fs, maxs[:,i]/sum(maxs[:,i]), '.-')
    legend(['max norm', 'std norm', 'std'])
    title('suppression')
    savefig('suppression.png')


def test2():    
    print('\n\nTest2\n')
    
    f = 1.41
    x = solve(f)

    figure()
    plot(x[:,0])
    plot(x[:,1])
    plot(x[:,N//2])
    plot(x[:,N-1])
    legend([0,1,N//2,N-1])
    savefig('oscillations.png')


def compute_bandstructure():
    w2, k = sp.symbols('w2 k', domain=sp.S.Reals)

    N = buoy_spacing

    perm_plus = np.diag(ones(N-1), 1)
    perm_plus[-1,0] = 1.0
    perm_plus = sp.Matrix(perm_plus)
    perm_minus = np.diag(ones(N-1), -1)
    perm_minus[0,-1] = 1.0
    perm_minus = sp.Matrix(perm_minus)

    diag1 = np.zeros([N,N])
    diag1[0,0] = 1.0
    diag2 = np.diag(ones(N))
    diag2[0,0] = 0.0

    ep = sp.exp(1j*k)
    em = sp.exp(-1j*k)

    M = -2.0*sp.eye(N) + sp.Matrix(diag1)*m2*w2 + sp.Matrix(diag2)*m1*w2 + sp.eye(N)*ep*sp.Matrix(perm_plus) + sp.eye(N)*em*sp.Matrix(perm_minus)

    p = sp.lambdify(k, sp.Poly(M.det(method='berkowitz'), w2).coeffs(), 'numpy')

    ks = linspace(-pi/N,pi/N,100)
    rs = array([sqrt(roots(real(p(k)))) for k in ks])

    figure()
    plot(ks, rs, '-k', linewidth=2)
    xlabel('kx', fontsize=15)
    ylabel('$\omega$', fontsize=15)
    savefig('bands.png')
    close()

    return ks, rs
    

def make_movie_frames():
    print('\n\nMaking Movie Frames\n')
    
    ks, rs = compute_bandstructure()

    #N = 2*W + Nmid
    print('N', N)
    fs = (0.5, 1.00, 1.41)
    ct = 0

    if os.path.exists('frames/'): shutil.rmtree('frames/')    
    os.mkdir('frames/')
    
    for f in fs:
        
        for i in range(20):
            fig = figure()
            fig.set_size_inches(30, 10)
            ax = axes([0.1, 0.1, 0.8, 0.8], frameon=False)
            ax.annotate('$\omega_\mathrm{drive}$ = %1.2f'%f, fontsize=38, xycoords='axes fraction', xy=(0.4, 0.45))
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            savefig('frames/%d.png'%ct)
            close()
            ct += 1        

        ct += len(ts)


        x = solve(f)

        bs = [i for i,m in enumerate(ms) if m==m2]
        L = N + len(bs)*(buoy_size-1)
        ex = zeros([len(ts), L])
        i = 0
        j = 0
        bs = []
        while i < shape(x)[1]:
            d = buoy_size if ms[i]==m2 else 1
            ex[:,j:j+d] = x[:,i][:,None]
            if d==buoy_size: bs.append(range(j,j+d))
            i += 1
            j += d

        print('starting to plot')
        ymin = 1.4*amin(x)
        xmax = amax(x)
        frames = []
        for i in range(len(ts)):
            fig = figure()
            fig.set_size_inches(30, 10)

            ax = axes([0.05, 0.52, 0.9, 0.4])
            surface = ax.fill_between(range(L), ex[i], ymin, animated=True, color='C0')
            for b in bs:
                ax.plot([b[0]+(buoy_size-1)/2.0, b[0]+(buoy_size-1)/2.0], [ymin, ex[i,b[0]]+0.1*ymin], '-k', lw=1.5)
                ax.plot([b[0]+(buoy_size-1)/2.0, b[0]+(buoy_size-1)/2.0], [ex[i,b[0]]-0.1*ymin, xmax], '-k', lw=1.5)
                ax.fill_between(b, ex[i,b]+0.1*ymin, ex[i,b]-0.1*ymin, color='grey')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_ylim(ymin, -ymin)
            ax.set_xlim(0, L-1)
            ax.annotate('$\omega_\mathrm{drive}$ = %1.2f'%f, fontsize=26, xycoords='axes fraction', xy=(0.47, 1.04))

            ax = axes([0.6-0.4/5.0, 0.06, 0.4/2.5, 0.4])
            ax.plot(ks*buoy_spacing, rs, '-k', linewidth=3)
            ax.set_xlabel('kx', fontsize=19)
            ax.set_ylabel('$\omega$', fontsize=20)
            ax.set_title('band structure', fontsize=22)
            ax.hlines(f, buoy_spacing*ks[0], buoy_spacing*ks[-1], linestyle='--', color='g', linewidth=3)
            ax.tick_params(axis='both', which='major', labelsize=15)

            ax = axes([0.6 - 0.4/5.0 - 0.12, 0.1, 0.1, 0.33], frameon=False)
            for ic in range(5):
                fc = 'k' if ic==2 else 'w'
                ax.add_patch(mpl.patches.Circle(((ic-2)*0.3, 0.0), radius=0.1, facecolor=fc, edgecolor='k'))
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.annotate('unit cell', fontsize=20, xycoords='axes fraction', xy=(0.3, 0.7))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)        

            print(i)
            savefig('frames/%d.png'%ct)
            close()
            ct += 1


test1()
test2()
make_movie_frames()
