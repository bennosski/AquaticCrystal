import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
from numpy import *
from scipy.integrate import odeint
import matplotlib.animation as animation
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.animation as animation

# system size
W = 31
buoy_spacing = 3
buoy_size = 1
Nmid = (buoy_spacing-1)*30 + 1
Nx = 2*W + Nmid
Ny = 41

m1 = 1.0 
m2 = 5.0 

ms = m1*ones([Nx, Ny])
ms[W:Nx-W:2, 0:Ny:2] = m2 

ts = linspace(0, 450, 750)

def F(arr, t, f): 
    arr = reshape(arr, [2, Nx, Ny])
    x = arr[0]
    v = arr[1]
    
    forces = zeros([Nx, Ny])
    for axis in (0,1):
        for shift in (-1,1):
            tmp = roll(x, shift, axis=axis) - x
            
            if axis==0: tmp[(shift-1)//2,:] = 0.0
            else: tmp[:,(shift-1)//2] = 0.0
            
            forces += tmp
            
    drive = cos(f*t)    
    forces[0, Ny//2] += drive
    forces -= mean(forces)
    a = forces/ms

    out = concatenate((v, a), axis=0)
    
    return reshape(out, [-1])

def solve(f): 
    x = odeint(F, zeros(2*Nx*Ny), ts, args=(f,))
    return reshape(x, [len(ts),2,Nx,Ny])[:,0,:,:]


def test1(): # test for some set of frequencies
    fs = (0.52, 1.021, 1.543)
    for f in fs:
        x = solve(f)

        vmax = amax(x) * 0.9
        vmin = amin(x) * 0.9

        figure()
        subplot(3,1,1)
        imshow(x[20].T, aspect=2.0, interpolation='bilinear', vmax=vmax, vmin=vmin)
        subplot(3,1,2)
        imshow(x[300].T, aspect=2.0, interpolation='bilinear', vmax=vmax, vmin=vmin)
        subplot(3,1,3)
        imshow(x[-1].T, aspect=2.0, interpolation='bilinear', vmax=vmax, vmin=vmin)
        savefig('ripples%1.1f.png'%f)
        close()

        figure()
        plot(x[:,0,Ny//2])
        plot(x[:,1,Ny//2])
        plot(x[:,Nx//2,Ny//2])
        plot(x[:,Nx-1,Ny//2])
        legend([0,1,Nx//2,Nx-1])
        savefig('oscillations%1.1f.png'%f)
        close()

        save('x%1.2f.npy'%f, x)

def test2(): # test range of frequencies and determine whether the wave is suppressed or passes through the crystal
    fs = linspace(0.5, 1.7, 70)
    maxs = []
    for f in fs:
        x = solve(f)
        s1 = amax(x[:,-W:, :])/amax(x[:, :W, :])
        s2 = std(x[:,-W:, :])/std(x[:,:W, :])
        s3 = std(x[:,-W:, :])
        maxs.append([s1,s2,s3])
        print(f, maxs[-1])

    maxs = array(maxs)
    figure()
    for i in range(3):
        plot(fs, maxs[:,i]/sum(maxs[:,i]), '.-')
    legend(['max norm', 'std norm', 'std'])
    title('suppression')
    savefig('suppression.png')



def movie(): # plotting

    w2, kx, ky = sp.symbols('w2 kx ky')

    cx = sp.cos(kx)
    cy = sp.cos(ky)
    M = sp.Matrix([[m2*w2-4, 2*cx, 2*cy, 0],
                   [2*cx, m1*w2-4, 0, 2*cy],
                   [2*cy, 0, m1*w2-4, 2*cx],
                   [0, 2*cy, 2*cx, m1*w2-4]])

    p = sp.lambdify((kx,ky), sp.Poly(M.det(), w2).coeffs())

    Nk = 40
    kxsbs, kysbs = meshgrid(linspace(-pi,pi,Nk), linspace(-pi,pi,Nk))

    bs = zeros([Nk,Nk,4])
    for i in range(Nk):
        for j in range(Nk):
            bs[i,j] = sorted(sqrt(roots(p(kxsbs[i,j], kysbs[i,j]))))

    fig = figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(4):
        ax.plot_surface(kxsbs, kysbs, bs[:,:,i])
    #ax.plot(kxs, kys, 0.5, color='grey', alpha=0.6)
    ax.view_init(7.0, 15.0+90.0)
    ax.set_xlabel('kx', fontsize=15)
    ax.set_ylabel('ky', fontsize=15)
    ax.set_zlabel('$\omega$', fontsize=15)
    ax.set_title('band structure', fontsize=19)

    savefig('bands.png')
    close()


    fs = (0.52, 1.021, 1.543)
    ct = 0

    for f in fs:

        #x = solve(f)
        x = load('x%1.2f.npy'%f)

        for i in range(20):
            fig = figure()
            fig.set_size_inches(30, 20)
            ax = axes([0.1, 0.1, 0.8, 0.8], frameon=False)
            ax.annotate('$\omega_\mathrm{drive}$ = %1.1f'%f, fontsize=55, xycoords='axes fraction', xy=(0.4, 0.45))
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            savefig('frames/%d.png'%ct)
            close()
            ct += 1        

        buoys = zeros([Nx,Ny], dtype=bool)
        buoys[W:Nx-W:2, 0:Ny:2] = True
        kys, kxs = meshgrid(range(Ny), range(Nx))
        kxs, kys = ravel(kxs[buoys]), ravel(kys[buoys])

        print('starting to plot')
        vmax = amax(x) * 0.9
        vmin = amin(x) * 0.9
                
        Nt = 500 if f==fs[0] else len(ts)
        for i in range(Nt):

            fig, ax1 = subplots(1,1)            
            fig.set_size_inches(30, 20)

            ax = fig.add_subplot(111, projection='3d')
            for j in range(4):
                ax.plot_surface(kxsbs, kysbs, bs[:,:,j])
            ax.view_init(7.0, 15.0+90.0)
            ax.text(1.2, 0, 2.91, 'band structure', size=32)
            ax.set_position([0.5-0.4/5.0, -0.05, 0.45, 0.45*1.5])
            for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(20)
            for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(20)
            for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(20)
            [t.set_va('center') for t in ax.get_yticklabels()]
            [t.set_ha('left') for t in ax.get_yticklabels()]
            [t.set_va('center') for t in ax.get_xticklabels()]
            [t.set_ha('right') for t in ax.get_xticklabels()]
            [t.set_va('center') for t in ax.get_zticklabels()]
            [t.set_ha('left') for t in ax.get_zticklabels()]
            ax.text(-4.7, -2.3, 1.2, '$\omega$', size=35)
            ax.text(-4.1, 0, -0.32, 'ky', size=30)
            ax.text(-0.7, 8.2, 0.0, 'kx', size=30)
            ax.patch.set_alpha(0.0)

            ax1.imshow(x[i].T, aspect='auto', interpolation='bilinear', vmin=vmin, vmax=vmax)
            buoy_heights = ravel(x[i][buoys])
            for h in buoy_heights:
                ax1.plot(kxs, kys, '.', color='grey', markersize=8+15*(h-vmin)/(vmax-vmin))
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.set_ylim(0, Ny-1)
            ax1.set_xlim(0, Nx-1)
            ax1.annotate('$\omega_\mathrm{drive}$ = %1.1f'%f, fontsize=35, xycoords='axes fraction', xy=(0.47, 1.04))
            ax1.set_position([0.05, 0.55, 0.9, 0.4])            
            
            ax = axes([0.35, 0.2, 0.08, 0.08*1.5], frameon=False)
            for ic1 in range(2):
                for ic2 in range(2):
                    fc = 'k' if ic1==0 and ic2==0 else 'w'
                    ax.add_patch(mpl.patches.Circle((ic1-0.5, ic2-0.5), radius=0.3, facecolor=fc, edgecolor='k', linewidth=2))
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.annotate('unit cell', fontsize=27, xycoords='axes fraction', xy=(0.22, 1.05))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)        
            
            print(ct)
            savefig('frames/%d.png'%ct)
            close()
            ct += 1


test1()
#movie()
#test2()
