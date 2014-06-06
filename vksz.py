import numpy as np
import ipdb
import matplotlib.pylab as pl
pl.ion()
datadir = 'data/'
from astropy.io import fits
import cPickle as pickle
#import healpy as hp
from cosmolopy import distance, fidcosmo
cosmo = fidcosmo

def main( hemi='south'):
    grid = grid3d(hemi=hemi)
    n_data = num_sdss_data_both_catalogs(hemi, grid)
    n_rand = num_sdss_rand_both_catalogs(hemi, grid)
    n_rand *= (1.*n_data.sum()/n_rand.sum())
    delta, weight = num_to_delta(n_data, n_rand, fwhm_sm=2)
    vels = delta_to_raw_vels(delta, weight, grid)
    vlos = vels_to_vel_los_from_observer(vels, grid)
    ipdb.set_trace()

    
def delta_to_raw_vels(delta, weight, grid):
    from numpy.fft import fftn, ifftn
    vels = np.zeros((3, delta.shape[0], delta.shape[1], delta.shape[2]))
    ks = get_wavenumbers(delta.shape, grid.reso_mpc)
    kmag = ks[3]
    kmag2 = (kmag+1e-9)**(2.)
    ii = np.complex(0,1)
    for i in range(3):
        vels[i,:, :, :] = np.real(ifftn(ii*fftn(delta*weight, axes=[i]) * ks[i] / kmag2, axes=[i]))
    return vels

def vels_to_vel_los_from_observer(vels, grid):
    shape = vels.shape[1:]
    ones = np.ones(shape)
    x0, y0, z0 = grid.get_voxel_indices([0.], [0.], [0.])
    xx = np.arange(shape[0])[:,np.newaxis,np.newaxis]*ones
    yy = np.arange(shape[1])[np.newaxis,:,np.newaxis]*ones    
    zz = np.arange(shape[2])[np.newaxis,np.newaxis,:]*ones        
    dd = np.array([xx-x0[0], yy-y0[0], zz-z0[0]])
    mag = np.sqrt((dd**2.).sum(0))
    inv_mag = 1./mag
    inv_mag[mag==0]=0.
    for i in range(3): dd[i,:,:,:] *= inv_mag
    vel_los_from_observer = (dd*vels).sum(0)
    return vel_los_from_observer



    
def get_wavenumbers(shape, reso_mpc):
    assert(len(shape)==3)
    output=[]
    ones = np.ones(shape)
    for i in range(3):
        tmp = np.fft.fftfreq(shape[i], d=reso_mpc)
        if i==0:
            ktmp = tmp[:, np.newaxis, np.newaxis]*ones
        if i==1:
            ktmp = tmp[np.newaxis, :, np.newaxis]*ones
        if i==2:
            ktmp = tmp[np.newaxis, np.newaxis, :]*ones
        output.append(ktmp)
    output.append(np.sqrt(output[0]**2. + output[1]**2. + output[2]**2.))
    return output

    
def num_sdss_data_both_catalogs(hemi, grid):    
    d_data = load_sdss_data_both_catalogs(hemi)
    return grid.num_from_radecz(d_data['ra'],d_data['dec'], d_data['z'])

def num_sdss_rand_both_catalogs(hemi, grid):    
    d_rand = load_sdss_rand_both_catalogs(hemi)
    return grid.num_from_radecz(d_rand['ra'],d_rand['dec'], d_rand['z'])




def num_to_delta(n_data, n_rand, fwhm_sm=2, delta_max=3.):
    from scipy.ndimage import gaussian_filter
    sigma_sm = fwhm_sm/2.355
    # smooth rand
    n_rand = gaussian_filter(n_rand, sigma_sm)

    # smooth data
    # tmpp, replace with a filter that knows about LSS and shot noise.
    n_data = gaussian_filter(n_data, sigma_sm)

    delta = (n_data-n_rand)/n_rand
    delta[n_rand==0]=0.
    delta[delta>delta_max]=delta_max
    weight = n_rand/np.max(n_rand)
    return delta, weight

    

class grid3d(object):
    def __init__(self, hemi='south', reso_mpc=16.0, 
                 nx=2**8, ny=2**8, nz=2**8, 
                 zmin=0.1, zmax=0.55):
        self.hemi = hemi
        self.reso_mpc = reso_mpc
        self.zmin = zmin
        self.zmax = zmax
        self.nx=nx
        self.ny=ny
        self.nz=nz
        d = load_sdss_data_both_catalogs(hemi)
        ra = d['ra']; dec=d['dec']; z=d['z']
        wh = np.where((z>self.zmin)&(z<self.zmax))[0]
        ra=ra[wh]; dec=dec[wh]; z=z[wh]
        rr = interp_comoving_distance(z)
        th = (90.-dec)*np.pi/180.
        phi = ra*np.pi/180.
        xx = rr*np.sin(th)*np.cos(phi)
        yy = rr*np.sin(th)*np.sin(phi)
        zz = rr*np.cos(th)
        x1d = np.arange(nx)*reso_mpc
        y1d = np.arange(ny)*reso_mpc
        z1d = np.arange(nz)*reso_mpc
        for thing in [x1d, y1d, z1d]: thing -= np.mean(thing)
        x1d += xx.mean()
        y1d += yy.mean()
        z1d += zz.mean()
        self.x1d = x1d
        self.y1d = y1d
        self.z1d = z1d

        
    def get_voxel_indices(self, x, y, z):
        ind_x = np.digitize(x, self.x1d)
        ind_y = np.digitize(y, self.y1d)
        ind_z = np.digitize(z, self.z1d)
        return ind_x, ind_y, ind_z
        
    def num_from_xyz(self, x, y, z):
        n = np.zeros((self.nx, self.ny, self.nz))
        ind_x, ind_y, ind_z = self.get_voxel_indices(x, y, z)
        for xtmp,ytmp,ztmp in zip(ind_x, ind_y, ind_z):
            n[xtmp, ytmp, ztmp] += 1.        
        return n

    def num_from_radecz(self, ra, dec, z):
        wh = np.where((z>self.zmin)&(z<self.zmax))[0]
        ra=ra[wh]; dec=dec[wh]; z=z[wh]
        rr = interp_comoving_distance(z)
        th = (90.-dec)*np.pi/180.
        phi = ra*np.pi/180.
        xx = rr*np.sin(th)*np.cos(phi)
        yy = rr*np.sin(th)*np.sin(phi)
        zz = rr*np.cos(th)        
        n = self.num_from_xyz(xx, yy, zz)
        return n

    def vec_to_observer(self):
        shape = (self.nx, self.ny, self.nz)
        ones = np.ones(shape)
        x0, y0, z0 = self.get_voxel_indices([0.], [0.], [0.])
        xx = np.arange(shape[0])[:,np.newaxis,np.newaxis]*ones
        yy = np.arange(shape[1])[np.newaxis,:,np.newaxis]*ones
        zz = np.arange(shape[2])[np.newaxis,np.newaxis,:]*ones
        return np.array([xx-x0[0], yy-y0[0], zz-z0[0]])


    def distance_to_observer(self):
        rtmp = self.vec_to_observer()
        return np.sqrt((rtmp**2.).sum(0))
        


def interp_comoving_distance(z, deltaz=1e-3):
    # It should be fine to do linear interpolation between points 
    # separated by deltaz=1e-3.  The comoving distance function will be 
    # very linear across that deltaz.  I checked and yes, agrees with 
    # truth to ~1e-4 Mpc.
    zgrid = np.arange(np.min(z)-deltaz, np.max(z)+deltaz, deltaz)
    dgrid = distance.comoving_distance(zgrid, **cosmo)
    from scipy import interpolate
    f = interpolate.interp1d(zgrid, dgrid)
    return f(z)


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
           'galaxy_DR10v8_CMASS_North.fits','galaxy_DR10v8_CMASS_South.fits',
           'random1_DR10v8_CMASS_North.fits', 'random1_DR10v8_CMASS_South.fits', 
           'random1_DR10v8_LOWZ_North.fits', 'random1_DR10v8_LOWZ_South.fits', 
           'random2_DR10v8_CMASS_North.fits', 'random2_DR10v8_CMASS_South.fits', 
           'random2_DR10v8_LOWZ_North.fits', 'random2_DR10v8_LOWZ_South.fits']
    basepath = 'http://data.sdss3.org/sas/dr10/boss/lss/'
    for file in files:
        url = basepath + file + '.gz'
        savename = datadir + file + '.gz'
        urlretrieve(url, savename)
        system('gunzip '+savename)

def load_sdss_rand(catalog, hemi):
    catup = catalog.upper()
    hemic = hemi.capitalize()
    ra = []; dec=[]; z=[]
    for i in range(2):
        filename = datadir+'random%i_DR10v8_'%(i+1)+catup+'_'+hemic+'.fits'
        tmp = fits.open(filename)[1].data
        ra.append(tmp['ra'])
        dec.append(tmp['dec'])
        z.append(tmp['z'])
    ra = np.hstack(ra)
    dec = np.hstack(dec)
    z = np.hstack(z)    
    return {'ra':ra, 'dec':dec, 'z':z}


def load_sdss_data(catalog, hemi):
    catup = catalog.upper()
    hemic = hemi.capitalize()
    filename = datadir+'galaxy_DR10v8_'+catup+'_'+hemic+'.fits'
    tmp = fits.open(filename)[1].data
    output = {'ra':tmp['ra'], 'dec':tmp['dec'], 'z':tmp['z']}
    return output

def load_sdss_rand_both_catalogs(hemi):
    lowz = load_sdss_rand('lowz', hemi)
    cmass = load_sdss_rand('cmass', hemi)
    ra = np.hstack([lowz['ra'],cmass['ra']])
    dec = np.hstack([lowz['dec'],cmass['dec']])
    z = np.hstack([lowz['z'],cmass['z']])        
    return {'ra':ra, 'dec':dec, 'z':z}
    
def load_sdss_data_both_catalogs(hemi):
    lowz = load_sdss_data('lowz', hemi)
    cmass = load_sdss_data('cmass', hemi)
    ra = np.hstack([lowz['ra'],cmass['ra']])
    dec = np.hstack([lowz['dec'],cmass['dec']])
    z = np.hstack([lowz['z'],cmass['z']])        
    return {'ra':ra, 'dec':dec, 'z':z}


    

        
def do_everything():
    download_redmapper()
    download_lowz_and_cmass()

