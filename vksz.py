import numpy as np
import ipdb
import matplotlib.pylab as pl
pl.ion()
datadir = 'data/'
from astropy.io import fits
import cPickle as pickle
import healpy as hp
from cosmolopy import distance, fidcosmo, constants
cosmo = fidcosmo  #probably want to put in Planck cosmology at some point.
mpc2km = constants.Mpc_km
# Unless otherwise noted, distances are in Mpc and velocities are in Mpc/s.
TCMB = 2.72548 #K


def main(hemi='south'):
    rm = get_cluster_velocities(quick=True)
    template = create_healpix_template(rm)
    amp = cross_template_with_planck(template)
    ipdb.set_trace()


def cross_template_with_planck(template, lmax=4000):
    planck, mask = load_planck_data()

    print '...anafast...'
    cl_template_planck = hp.anafast(template*mask, map2=planck*mask, lmax=lmax)
    cl_template = hp.anafast(template*mask, lmax=lmax)
    cl_planck = hp.anafast(planck*mask, lmax=lmax)
    
    # correct some of these for Planck beam function.
    bl, l_bl = load_planck_bl()
    l_bl = l_bl[0:lmax+1]
    bl = bl[0:lmax+1]
    cl_template_planck /= (bl)
    cl_planck /= (bl**2.)

    # get the weighted sum of the template amplitude in this data.
    amp = np.sum(cl_template_planck/cl_planck)/np.sum(cl_template/cl_planck)

    # tmpp, actually, i should use a theory curve for c_planck.
    # CL_PLANCK = CMB + noise/bl^2.
    ipdb.set_trace()    
    return amp



def load_planck_data(quick=True):
    #tmpp, need to add these to download functions.
    planck = fits.open(datadir+'HFI_SkyMap_217_2048_R1.10_nominal.fits')[1].data['I_STOKES']
    planck = hp.reorder(planck, n2r=True)
    mask = fits.open(datadir+'HFI_Mask_GalPlane_2048_R1.10.fits')[1].data['GAL060']#tmpp, 40 vs 60?
    mask = hp.reorder(mask, n2r=True)
    return planck, mask 


def load_planck_bl():
    # tmpp, need to add these to download functions.
    x=np.loadtxt(datadir+'HFI_RIMO_R1.10.BEAMWF_217X217.txt')
    bl = x[:, 0]
    l_bl = np.arange(len(bl))
    return bl, l_bl

    
def create_healpix_template(rm, nside=2**11, n_theta_core=5.):
    fill_free_electron_parameters(rm)
    # only add those above some weight threshold
    npix = hp.nside2npix(nside)
    template = np.zeros(npix)
    ncl = len(rm['ra'])
    vec_hpix = hp.ang2vec(rm['th_gal'], rm['phi_gal'])
    ind_hpix = hp.ang2pix(nside, rm['th_gal'], rm['phi_gal'], nest=False)
    for i in range(ncl):
        this_vec = vec_hpix[i,:]
        this_theta_c = rm['theta_c'][i] #radians
        this_ind = ind_hpix[i]
        ind_nearby = hp.query_disc(nside, this_vec, n_theta_core*this_theta_c, nest=False)
        vec_nearby = hp.pix2vec(nside, ind_nearby, nest=False)
        theta_nearby = hp.rotator.angdist(this_vec, vec_nearby)
        values_nearby = 1./(1.+(theta_nearby/this_theta_c)**2.)
        values_nearby *= rm['t0'][i]
        template[ind_nearby] += values_nearby
    return template
    
    
def fill_free_electron_parameters(rm, tau20=0.002):
    # tau20 is the optical depth to Thomson scattering for a
    # CMB photon traversing through the center of a Lambda=20 cluster.
    ncl = len(rm['ra'])
    tau = []
    theta_c = []
    dang = distance.angular_diameter_distance(rm['z_lam'], **cosmo)
    for i in range(ncl):
        this_lambda = rm['lam'][i]
        this_tau = tau20*(this_lambda/20.)
        this_rc_mpc = 0.36 #tmpp.  should scale with some power of lambda.
        this_theta_c = this_rc_mpc/dang[i]
        tau.append(this_tau)
        theta_c.append(this_theta_c)
    tau = np.array(tau)
    theta_c = np.array(theta_c)
    rm['tau'] = tau
    rm['theta_c'] = theta_c
    t0 = -(rm['vlos']/constants.c_light_Mpc_s)*TCMB*rm['tau']
    rm['t0'] = t0
    return
        

    
def show_vlos(hemi='south'):
    vlos = vlos_for_hemi(hemi)
    pl.clf(); pl.imshow(vlos[:,:,128]*mpc2km,vmin=-200,vmax=200)
    pl.colorbar()
    pl.title(hemi+', LOS velocities (km/s)')


def get_cluster_velocities(quick=False):
    savename = datadir+'get_cluster_velocities.pkl'
    if quick: return pickle.load(open(savename,'r'))
    rm_s = get_cluster_velocities_one_hemi('south')
    rm_n = get_cluster_velocities_one_hemi('north')
    rm={}
    for k in rm_s.keys():
        rm[k] = np.hstack([rm_s[k],rm_n[k]])
    pickle.dump(rm, open(savename,'w'))
    return rm

    
    
def get_cluster_velocities_one_hemi(hemi):
    vlos, weight, grid = vlos_for_hemi(hemi)
    rm = load_redmapper(hemi)
    ix, iy, iz = grid.voxel_indices_from_radecz(rm['ra'], rm['dec'], rm['z_lam'], applyzcut=False)
    vlos_rm = vlos[ix, iy, iz]
    weight_rm = weight[ix, iy, iz]
    rm['vlos'] = vlos_rm
    rm['weight'] = weight_rm
    return rm

    
def vlos_for_hemi(hemi):
    grid = grid3d(hemi=hemi)
    n_data = num_sdss_data_both_catalogs(hemi, grid)
    n_rand = num_sdss_rand_both_catalogs(hemi, grid)
    n_rand *= (1.*n_data.sum()/n_rand.sum())
    delta, weight = num_to_delta(n_data, n_rand, fwhm_sm=2)
    vels = delta_to_vels(delta, weight, grid)
    vlos = vels_to_vel_los_from_observer(vels, grid)
    return vlos, weight, grid
    
def delta_to_vels(delta, weight, grid):
    from numpy.fft import fftn, ifftn
    vels = np.zeros((3, delta.shape[0], delta.shape[1], delta.shape[2]))
    ks = get_wavenumbers(delta.shape, grid.reso_mpc)
    kmag = ks[3]
    kmag2 = (kmag+1e-9)**(2.)
    ii = np.complex(0,1)
    for i in range(3):
        vels[i,:, :, :] = np.real(ifftn(ii*fftn(delta*weight, axes=[i]) * ks[i] / kmag2, axes=[i]))

    # calculate the z-dependent factor, which will put these velocities into Mpc/s.
    zgrid3d = grid.redshift_grid()
    z_dep_fac = interp_z_dependent_velocity_factors(zgrid3d)
    for i in range(3): vels[i, :, :] *= z_dep_fac
    return vels

    
def vels_to_vel_los_from_observer(vels, grid):
    '''
    shape = vels.shape[1:]
    ones = np.ones(shape)
    x0, y0, z0 = grid.get_voxel_indices([0.], [0.], [0.])
    xx = np.arange(shape[0])[:,np.newaxis,np.newaxis]*ones
    yy = np.arange(shape[1])[np.newaxis,:,np.newaxis]*ones    
    zz = np.arange(shape[2])[np.newaxis,np.newaxis,:]*ones        
    dd = np.array([xx-x0[0], yy-y0[0], zz-z0[0]])
    '''
    dd = grid.vec_from_observer()
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

    
def num_to_delta(n_data, n_rand, fwhm_sm=2, delta_max=3., linear_bias=2.0):
    from scipy.ndimage import gaussian_filter
    sigma_sm = fwhm_sm/2.355
    # smooth rand
    n_rand = gaussian_filter(n_rand, sigma_sm)

    # smooth data
    # tmpp, replace with a filter that knows about LSS and shot noise.
    n_data = gaussian_filter(n_data, sigma_sm)

    delta = (n_data-n_rand)/n_rand
    delta /= linear_bias
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


    def voxel_indices_from_radecz(self, ra, dec, z, applyzcut=False):
        xx, yy, zz = self.xyz_from_radecz(ra, dec, z, applyzcut=applyzcut)
        ix, iy, iz = self.get_voxel_indices(xx, yy, zz)
        return ix, iy, iz

        
    def xyz_from_radecz(self, ra, dec, z, applyzcut=True):
        if applyzcut:
            wh = np.where((z>self.zmin)&(z<self.zmax))[0]
            ra=ra[wh]; dec=dec[wh]; z=z[wh]
        rr = interp_comoving_distance(z)
        th = (90.-dec)*np.pi/180.
        phi = ra*np.pi/180.
        xx = rr*np.sin(th)*np.cos(phi)
        yy = rr*np.sin(th)*np.sin(phi)
        zz = rr*np.cos(th)
        return xx, yy, zz

        
    def num_from_radecz(self, ra, dec, z):
        '''
        wh = np.where((z>self.zmin)&(z<self.zmax))[0]
        ra=ra[wh]; dec=dec[wh]; z=z[wh]
        rr = interp_comoving_distance(z)
        th = (90.-dec)*np.pi/180.
        phi = ra*np.pi/180.
        xx = rr*np.sin(th)*np.cos(phi)
        yy = rr*np.sin(th)*np.sin(phi)
        zz = rr*np.cos(th)
        '''
        xx, yy, zz = self.xyz_from_radecz(ra, dec, z)
        n = self.num_from_xyz(xx, yy, zz)
        return n

    def vec_from_observer(self):
        shape = (self.nx, self.ny, self.nz)
        ones = np.ones(shape)
        x0, y0, z0 = self.get_voxel_indices([0.], [0.], [0.])
        xx = np.arange(shape[0])[:,np.newaxis,np.newaxis]*ones
        yy = np.arange(shape[1])[np.newaxis,:,np.newaxis]*ones
        zz = np.arange(shape[2])[np.newaxis,np.newaxis,:]*ones
        return np.array([xx-x0[0], yy-y0[0], zz-z0[0]])*self.reso_mpc


    def distance_from_observer(self):
        rtmp = self.vec_from_observer()
        return np.sqrt((rtmp**2.).sum(0))


    def redshift_grid(self):
        d = self.distance_from_observer()
        zgrid = np.arange(0,2,1e-3)
        dgrid = distance.comoving_distance(zgrid, **cosmo)
        from scipy import interpolate
        f = interpolate.interp1d(dgrid, zgrid)
        return f(d)
        


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


def interp_z_dependent_velocity_factors(z, deltaz=1e-3):
    from cosmolopy.perturbation import fgrowth
    zgrid = np.arange(np.min(z)-deltaz, np.max(z)+deltaz, deltaz)
    hz_grid = distance.hubble_z(zgrid, **cosmo)
    fgrowth_grid = fgrowth(zgrid, cosmo['omega_M_0'])
    a_grid = 1./(1.+zgrid)
    tmp_grid = fgrowth_grid*a_grid*hz_grid
    from scipy import interpolate
    f = interpolate.interp1d(zgrid, tmp_grid)
    return f(z)
    


def download_planck():
    from urllib import urlretrieve
    basepath = 'http://irsa.ipac.caltech.edu/data/Planck/release_1/all-sky-maps/maps/'
    file = 'HFI_SkyMap_217_2048_R1.10_nominal.fits'
    url = basepath + file
    savename = datadir + file
    urlretrieve(url, savename)

    
def download_redmapper():
    from urllib import urlretrieve
    #basepath = 'http://slac.stanford.edu/~erykoff/redmapper/catalogs/'
    #file = 'redmapper_dr8_public_v5.2_catalog.fits'
    basepath = 'http://www.slac.stanford.edu/~erykoff/catalogs/dr8/'
    file = 'dr8_run_redmapper_v5.10_lgt20_catalog.fit'
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
    download_planck()
    

def load_redmapper(hemi):
    d = fits.open(datadir+'dr8_run_redmapper_v5.10_lgt20_catalog.fit')[1].data
    if (hemi=='north'): wh=np.where(np.abs(d['ra']-180.)<100.)[0]
    if (hemi=='south'): wh=np.where(np.abs(d['ra']-180.)>=100.)[0]
    d = d[wh]
    ra = d['ra']
    dec = d['dec']
    lam = d['lambda_chisq']
    z_lam = d['z_lambda']
    cluster_id = d['mem_match_id']

    # let's go ahead and get the galactic coordinates.
    from astropy.coordinates import FK5
    from astropy import units as u
    coord = FK5(ra=ra, dec=dec, unit=(u.deg, u.deg))
    l_gal = coord.galactic.l.rad
    b_gal = coord.galactic.b.rad
    phi_gal = l_gal
    th_gal = np.pi/2.-b_gal
    return {'ra':ra, 'dec':dec, 'lam':lam, 'z_lam':z_lam, 'id':cluster_id, 
            'l_gal':l_gal, 'b_gal':b_gal, 'phi_gal':phi_gal, 'th_gal':th_gal}

    
