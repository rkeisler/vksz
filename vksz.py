import numpy as np
import ipdb
import matplotlib.pylab as pl
pl.ion()
datadir = 'data/'
from astropy.io import fits
import cPickle as pickle
#import healpy as hp



def grid(hemisphere='south', zmin=0.1, zmax=0.55):
    lowz = fits.open(datadir+'galaxy_DR10v8_LOWZ_'+hemisphere.capitalize()+'.fits')[1].data
    cmass = fits.open(datadir+'galaxy_DR10v8_CMASS_'+hemisphere.capitalize()+'.fits')[1].data
    ra = np.concatenate([lowz['ra'], cmass['ra']])
    dec = np.concatenate([lowz['dec'], cmass['dec']])
    z = np.concatenate([lowz['z'], cmass['z']])
    wh = np.where((z>zmin)&(z<zmax))[0]
    ra=ra[wh]; dec=dec[wh]; z=z[wh]
    from cosmolopy import distance, fidcosmo
    print '...calculating distance...'
    rr = distance.comoving_distance(z, **fidcosmo)
    th = (90.-dec)*np.pi/180.
    phi = ra*np.pi/180.
    xx = rr*np.sin(th)*np.cos(phi)
    yy = rr*np.sin(th)*np.sin(phi)
    zz = rr*np.cos(th)

    reso_mpc = 8.0
    nx = 2**9
    ny = 2**9
    nz = 2**9

    x1d = np.arange(nx)*reso_mpc
    y1d = np.arange(ny)*reso_mpc
    z1d = np.arange(nz)*reso_mpc
    for thing in [x1d, y1d, z1d]: thing -= np.mean(thing)
    x1d += xx.mean()
    y1d += yy.mean()
    z1d += zz.mean()

    ind_x = np.digitize(xx, x1d)
    ind_y = np.digitize(yy, y1d)
    ind_z = np.digitize(zz, z1d)
    n = np.zeros((nx,ny,nz))
    for xtmp,ytmp,ztmp in zip(ind_x, ind_y, ind_z):
        n[xtmp, ytmp, ztmp] += 1.

    pl.clf()
    for i in range(3):
        pl.subplot(1,3,i+1)
        pl.imshow(n.sum(i))
        if i==0: pl.title('%s, %.1f Mpc resolution'%(hemisphere,reso_mpc))


    for thing in [xx,yy,zz]:
        print thing.min(), thing.max(), thing.max()-thing.min(), np.log2((thing.max()-thing.min())/reso_mpc)


    ipdb.set_trace()


    
    



def download_redmapper():
    from urllib import urlretrieve
    basepath = 'http://slac.stanford.edu/~erykoff/redmapper/catalogs/'
    file = 'redmapper_dr8_public_v5.2_catalog.fits'
    url = basepath + file
    savename = datadir + file
    urlretrieve(url, savename)


def download_lowz_and_cmass():
    from os import system
    from urllib import urlretrieve
    files=['galaxy_DR10v8_LOWZ_North.fits','galaxy_DR10v8_LOWZ_South.fits',
           'galaxy_DR10v8_CMASS_North.fits','galaxy_DR10v8_CMASS_South.fits']
    basepath = 'http://data.sdss3.org/sas/dr10/boss/lss/'
    for file in files:
        url = basepath + file + '.gz'
        savename = datadir + file + '.gz'
        urlretrieve(url, savename)
        system('gunzip '+savename)


def do_everything():
    download_redmapper()
    download_lowz_and_cmass()

