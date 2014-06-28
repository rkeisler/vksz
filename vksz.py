import numpy as np
import ipdb
import matplotlib.pylab as pl
pl.ion()
datadir = 'data/'
from astropy.io import fits
import cPickle as pickle
import healpy as hp
from cosmolopy import distance, fidcosmo, constants, perturbation
cosmo = fidcosmo  #probably want to put in Planck cosmology at some point.
mpc2km = constants.Mpc_km
# Unless otherwise noted, distances are in Mpc and velocities are in Mpc/s.
TCMB = 2.72548 #K
nside=2**11
lmax=4000


def make_many_corr():
    hemi='south'
    dd = counts_2d_2pt(False, False, hemi, quick=False)
    rd = counts_2d_2pt(True, False, hemi, quick=False)
    dr = counts_2d_2pt(False, True, hemi, quick=False)
    rr = counts_2d_2pt(True, True, hemi, quick=False)

    hemi='north'
    dd = counts_2d_2pt(False, False, hemi, quick=False)
    rd = counts_2d_2pt(True, False, hemi, quick=False)
    dr = counts_2d_2pt(False, True, hemi, quick=False)
    rr = counts_2d_2pt(True, True, hemi, quick=False)


    
def try_corr_2d_2pt(hemi='south'):
    quick=True
    dd = counts_2d_2pt(False, False, hemi, quick=quick)
    rd = counts_2d_2pt(True, False, hemi, quick=quick)
    dr = counts_2d_2pt(False, True, hemi, quick=quick)
    rr = counts_2d_2pt(True, True, hemi, quick=quick)
    corr = (dd - rd - dr + rr)/rr
    corr = corr[:,1:]
    logcorr = np.log10(np.abs(corr))
    
    pl.clf()
    pl.imshow(np.hstack([logcorr[:, ::-1], logcorr]), vmin=-3,vmax=2)
    pl.colorbar()
    ipdb.set_trace()

    
def counts_2d_2pt(randoms_sdss, randoms_rm, hemi, quick=True):
    letters = ['D','R']
    savename = datadir + 'counts_2d_2pt_%s_%s%s.pkl'%(hemi, letters[randoms_sdss], letters[randoms_rm])
    if quick: return pickle.load(open(savename,'r'))
    print '...getting SDSS...'
    pos_sdss = get_pos_sdss(hemi, randoms=randoms_sdss)
    print '...getting RM...'
    pos_rm = get_pos_rm(hemi, randoms=randoms_rm)
    lam = load_redmapper(hemi=hemi)['lam']
    counts = counts_2d_2pt_from_pos(pos_sdss, pos_rm, lam)
    pickle.dump(counts, open(savename,'w'))
    return counts


def get_lambda_bin(lam):
    output = np.zeros_like(lam, dtype=int)
    output[np.where( (lam>=20.) & (lam<30.)  )[0]] = 0
    output[np.where( (lam>=30.) & (lam<40.)  )[0]] = 1
    output[np.where( (lam>=40.) & (lam<999.) )[0]] = 2
    return output


def counts_2d_2pt_from_pos(pos_sdss, pos_rm, lam, rmax=100., reso=2.):
    # preliminaries
    nsdss = pos_sdss.shape[0]
    nrm = pos_rm.shape[0]

    # figure out which lambda bins each RM cluster goes into
    lam_bin = get_lambda_bin(lam)
    n_lam_bin = len(set(lam_bin))    
    
    # build a couple of KDTree's, one for SDSS, one for RM.
    from sklearn.neighbors import KDTree
    tree_sdss = KDTree(pos_sdss, leaf_size=30)

    # define grids for r_pi and r_sigma.
    rpigrid = np.arange(-rmax, rmax, reso)
    nrpigrid = len(rpigrid)
    rsigmagrid = np.arange(0, rmax, reso)
    nrsigmagrid = len(rsigmagrid)

    # find all BOSS galaxies within "rmax" Mpc of each RM clusters.
    print '...querying tree...'
    ind, dist = tree_sdss.query_radius(pos_rm, rmax, count_only=False, return_distance=True)
    print '...done querying tree...'

    # loop over clusters, calculate (r_pi, r_sigma) for all nearby BOSS galaxies
    # bin those counts.
    counts_rpi_rsigma = [np.zeros((nrpigrid+1, nrsigmagrid+1), dtype=np.float) for i in range(n_lam_bin)]
    for irm in range(nrm):
        print '%i/%i'%(irm, nrm)
        these_ind = ind[irm]
        if len(these_ind)==0: continue
        this_pos_rm = pos_rm[irm, :]
        these_pos_sdss = pos_sdss[these_ind, :]
        these_s = dist[irm]
        these_mu = dot_los2(this_pos_rm, these_pos_sdss)
        these_rpi = these_s*these_mu
        these_rsigma = these_s*np.sqrt((1.-these_mu**2.))
        ind_rpi = np.digitize(these_rpi, rpigrid)
        ind_rsigma = np.digitize(these_rsigma, rsigmagrid)
        this_lam_bin = lam_bin[irm]
        for this_ind_rpi, this_ind_rsigma in zip(ind_rpi, ind_rsigma):
                counts_rpi_rsigma[this_lam_bin][this_ind_rpi, this_ind_rsigma] += 1.

    # normalize
    # ok, really you'd want to normalize by nrm *per lambda bin*,
    # but i don't think it will make any material difference.
    for i in range(n_lam_bin): counts_rpi_rsigma[i] *= (1./nrm/nsdss)
    return counts_rpi_rsigma


def get_pos_sdss(hemi, randoms=False):
    if randoms: sdss = load_sdss_rand_both_catalogs(hemi)
    else: sdss = load_sdss_data_both_catalogs(hemi)
    grid = grid3d(hemi=hemi)        
    x_sdss, y_sdss, z_sdss = grid.xyz_from_radecz(sdss['ra'], sdss['dec'], sdss['z'], applyzcut=False)    
    pos_sdss = np.vstack([x_sdss, y_sdss, z_sdss]).T
    return pos_sdss


def get_pos_rm(hemi, randoms=False):
    # load redmapper catalog
    rm = load_redmapper(hemi=hemi)
    if randoms:
        # randomize redshifts
        ncl = len(rm['ra'])
        z_spec_orig = rm['z_spec'].copy()
        np.random.shuffle(rm['z_spec'])
        wh = np.where(rm['z_spec']==z_spec_orig)[0]
        rm['z_spec'][wh] = np.roll(rm['z_spec'],1000)[wh]
    # get XYZ positions (Mpc) of both datasets
    grid = grid3d(hemi=hemi)            
    x_rm, y_rm, z_rm = grid.xyz_from_radecz(rm['ra'], rm['dec'], rm['z_spec'], applyzcut=False)
    pos_rm = np.vstack([x_rm, y_rm, z_rm]).T
    return pos_rm




def main(hemi='south', kmax=0.1, rmax=50.):
    '''
    rm = get_linear_velocities(quick=False)    
    template = create_healpix_ksz_template(rm)
    amp_data = cross_template_with_planck(template, nrandom=0)
    amp_random = cross_template_with_planck(template, nrandom=100)
    pickle.dump((amp_data, amp_random), open(datadir+'amps_ksz_217_cl39_medrand_32MpcFWHM.pkl','w'))
    ipdb.set_trace()
    '''

    rm = get_pairwise_velocities(quick=False, kmax=kmax, rmax=rmax)
    rm_linear = get_linear_velocities(quick=True)
    assert(len(rm['weight'])==len(rm_linear['weight']))
    rm['weight'] = rm_linear['weight']

    #wh=np.where(rm_linear['weight']>.2)[0]; pl.clf();
    #pl.plot(rm['vlos'][wh], (rm_linear['vlos']/rm['weight'])[wh], '.')
    #pl.title('kmax=%0.2f, rmax=%i'%(kmax, rmax))
    #print np.corrcoef(rm['vlos'][wh], (rm_linear['vlos']/rm['weight'])[wh])[0,1] 
    #ipdb.set_trace()
    template = create_healpix_ksz_template(rm)
    amp_data = cross_template_with_planck(template, nrandom=0)
    amp_random = cross_template_with_planck(template, nrandom=100)
    savename = datadir+'amps_ksz_217_cl39_pairwise_kmax%0.2f_rmax%i.pkl'%(kmax, rmax)
    pickle.dump((amp_data, amp_random), open(savename,'w'))


def cross_template_with_planck(template, nrandom=0):
    #band='mb'
    band=217

    
    # get mask
    mask = load_planck_mask()
    mask_factor = np.mean(mask**2.)

    # get planck beam
    bl, l_bl = load_planck_bl(band)
    l_bl = l_bl[0:lmax+1]
    bl = bl[0:lmax+1]

    # get CL_PLANCK_THEORY
    l_planck, cl_planck = get_cl_theory(band)
    # store a version of the biased, beam-convolved spectrum.
    cl_planck_biased = cl_planck.copy()
    # "unbias" this spectrum, i.e. correct for Planck beam
    cl_planck /= (bl**2.)

    # Define how we'll weight the differnet multipoles.
    # Since we've already gone from 2d to 1d power spectrum, we
    # need to weight by nmodes=2L+1.
    weight = (2.*l_planck+1.)/cl_planck
    weight[0:10] = 0.
    
    # get template auto-spectrum
    cl_template = hp.anafast(template*mask, lmax=lmax)/mask_factor
    
    # estimate amplitude(s)
    amps = []
    if (nrandom==0):
        print ' '
        print 'data'
        planck = load_planck_data(band)
        cl_template_planck = hp.anafast(template*mask, map2=planck*mask, lmax=lmax)/mask_factor
        # correct by one power of planck beam function
        cl_template_planck /= bl
        amp = np.sum(cl_template_planck*weight)/np.sum(cl_template*weight)
        print '%0.3f'%amp
        print amp
        amps.append(amp)
    else:
        print ' '
        print 'randoms'
        # loop over nrandom
        for irandom in range(nrandom):
            print '%i/%i'%(irandom,nrandom)
            # generate a Planck map.
            # it's roundabout to generate a map from some CL's, then calculate
            # alm's to cross with the template.  why not directly generate alm's?
            # because i want to include the effect of multiplying by the real-space mask.
            planck = hp.synfast(cl_planck_biased, nside, lmax=lmax)
            cl_template_planck = hp.anafast(template*mask, map2=planck*mask, lmax=lmax)/mask_factor
            # correct by one power of planck beam function
            cl_template_planck /= bl
            amp = np.sum(cl_template_planck*weight)/np.sum(cl_template*weight)
            print '%0.3f'%amp
            print amp
            amps.append(amp)
            print 'current RMS:'
            print np.std(amps)


    # return list of amplitudes
    return amps


def load_planck_data(band):
    '''
    #tmpp, need to add these to download functions.
    planck = fits.open(datadir+'HFI_SkyMap_217_2048_R1.10_nominal.fits')[1].data['I_STOKES']
    #planck = fits.open(datadir+'HFI_SkyMap_143_2048_R1.10_nominal.fits')[1].data['I_STOKES']
    '''
    '''
    planck = fits.open(datadir+'COM_CompMap_CMB-smica_2048_R1.20.fits')[1].data['I']
    planck *= (1e-6) # convert from uK to K, SMICA only
    '''
    if band=='mb': return make_multiband_map(quick=True)
    planck = fits.open(datadir+'HFI_SkyMap_%i_2048_R1.10_nominal.fits'%band)[1].data['I_STOKES']    
    planck = hp.reorder(planck, n2r=True)
    planck[planck<(-1000e-6)]=0.
    planck[planck>(+1000e-6)]=0.    
    return planck


def load_planck_mask():
    '''
    gmask = fits.open(datadir+'HFI_Mask_GalPlane_2048_R1.10.fits')[1].data['GAL060']#tmpp, 40 vs 60%?
    pmask = np.ones_like(gmask, dtype=np.float)
    tmp = fits.open(datadir+'HFI_Mask_PointSrc_2048_R1.10.fits')[1].data
    for band in [100, 143, 217]:
    #for band in [217]:        #tmpp
        pmask *= tmp['F%i_05'%band]
    mask = gmask*pmask
    #mask = fits.open(datadir+'COM_CompMap_CMB-smica_2048_R1.20.fits')[1].data['VALMASK'] # tmpp, could try I_MASK
    '''
    #mask = fits.open(datadir+'COM_Mask_Likelihood_2048_R1.10.fits')[1].data['CL49']
    mask = fits.open(datadir+'COM_Mask_Likelihood_2048_R1.10.fits')[1].data['CL39']    
    #mask = fits.open(datadir+'COM_Mask_Likelihood_2048_R1.10.fits')[1].data['CL31']    
    mask = hp.reorder(mask, n2r=True)    
    return mask



def load_planck_bl(band):
    # tmpp, need to add these to download functions.
    if band=='mb':
        fwhm_arcmin = 4.85
        sigma_rad = fwhm_arcmin/60.*np.pi/180./2.355
        l_bl = np.arange(lmax+1)        
        bl = np.exp(-0.5*l_bl*(l_bl+1.)*sigma_rad**2.)
    else:
        x=np.loadtxt(datadir+'HFI_RIMO_R1.10.BEAMWF_%iX%i.txt'%(band,band))
        bl = x[:, 0]
        #bl = fits.open(datadir+'COM_CompMap_CMB-smica_2048_R1.20.fits')[4].data['beam_wf']
        l_bl = np.arange(len(bl))
    return bl, l_bl

    
def create_healpix_ksz_template(rm, n_theta_core=5.,
                                weight_min=0.1, beta=0.7):
    fill_free_electron_parameters(rm)
    # only add those above some weight threshold
    wh=np.where(rm['weight']>weight_min)[0]
    print 'USING %i CLUSTERS WITH WEIGHT>%0.2f'%(len(wh), weight_min)
    for k in rm.keys(): rm[k]=rm[k][wh]
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
        values_nearby = (1.+(theta_nearby/this_theta_c)**2.)**((1.-3.*beta)/2.)
        values_nearby *= rm['t0'][i]
        template[ind_nearby] += values_nearby
    return template


def create_healpix_tsz_template(rm, n_theta_core=5.,
                            weight_min=0.1, beta=0.7):
    fill_free_electron_parameters(rm)
    # only add those above some weight threshold
    wh=np.where(rm['weight']>weight_min)[0]
    print 'USING %i CLUSTERS WITH WEIGHT>%0.2f'%(len(wh), weight_min)    
    for k in rm.keys(): rm[k]=rm[k][wh]
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
        values_nearby = (1.+(theta_nearby/this_theta_c)**2.)**((1.-3.*beta)/2.)
        #values_nearby *= rm['t0'][i]
        template[ind_nearby] += values_nearby
    return template


    
def fill_free_electron_parameters(rm, tau20=0.001):
    # tau20 is the optical depth to Thomson scattering for a
    # CMB photon traversing through the center of a Lambda=20 cluster.
    ncl = len(rm['ra'])
    tau = []
    theta_c = []
    dang = distance.angular_diameter_distance(rm['z_spec'], **cosmo)
    for i in range(ncl):
        this_lambda = rm['lam'][i]
        
        this_tau = tau20*(this_lambda/20.)#tmpp, how should it scale w mass?        
        #this_tau = tau20*(this_lambda/20.)**(1./3.)#tmpp, how should it scale w mass?
        
        this_rc_mpc = 0.3 #tmpp.  should scale with some power of lambda.
        #this_rc_mpc = 0.3*(this_lambda/20.)**(1./3.)
        
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
    vlos, weight, grid = vlos_for_hemi(hemi)
    pl.clf(); pl.imshow(vlos[:,:,128]*mpc2km,vmin=-500,vmax=500)
    pl.colorbar()
    pl.title(hemi+', LOS velocities (km/s)')


def get_linear_velocities(quick=False):
    savename = datadir+'get_linear_velocities.pkl'
    if quick: return pickle.load(open(savename,'r'))
    rm_s = get_linear_velocities_one_hemi('south')
    rm_n = get_linear_velocities_one_hemi('north')
    rm={}
    for k in rm_s.keys():
        rm[k] = np.hstack([rm_s[k],rm_n[k]])
    pickle.dump(rm, open(savename,'w'))
    return rm


def get_pairwise_velocities(quick=False, kmax=0.1, rmax=50.):
    savename = datadir+'get_pairwise_velocities_kmax%0.2f_rmax%i.pkl'%(kmax, rmax)
    if quick: return pickle.load(open(savename,'r'))
    rm_s = get_pairwise_velocities_one_hemi('south', kmax=kmax, rmax=rmax)
    rm_n = get_pairwise_velocities_one_hemi('north', kmax=kmax, rmax=rmax)
    rm={}
    for k in rm_s.keys():
        rm[k] = np.hstack([rm_s[k],rm_n[k]])
    pickle.dump(rm, open(savename,'w'))
    return rm

    
    
def get_linear_velocities_one_hemi(hemi):
    vlos, weight, grid = vlos_for_hemi(hemi)
    rm = load_redmapper(hemi=hemi)
    ix, iy, iz = grid.voxel_indices_from_radecz(rm['ra'], rm['dec'], rm['z_spec'], applyzcut=False)
    rm['vlos'] = vlos[ix, iy, iz]
    rm['weight'] = weight[ix, iy, iz]
    return rm

    
def vlos_for_hemi(hemi):
    grid = grid3d(hemi=hemi)
    n_data = num_sdss_data_both_catalogs(hemi, grid)
    n_rand, weight = num_sdss_rand_both_catalogs(hemi, grid)
    n_data *= weight
    n_rand *= weight
    n_rand *= (1.*n_data.sum()/n_rand.sum())
    delta = num_to_delta(n_data, n_rand)
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

    
def num_sdss_rand_both_catalogs_noisy(hemi, grid):    
    d_rand = load_sdss_rand_both_catalogs(hemi)
    return grid.num_from_radecz(d_rand['ra'],d_rand['dec'], d_rand['z'])

def num_sdss_rand_both_catalogs(hemi, grid):
    d_rand = load_sdss_rand_both_catalogs(hemi)
    n_noisy = grid.num_from_radecz(d_rand['ra'],d_rand['dec'], d_rand['z'])
    # get the distance-from-observer 3d grid
    d_obs = grid.distance_from_observer()
    d_obs_max = d_obs[n_noisy>0].max()
    d_obs_1d = np.linspace(0, d_obs_max+1., 100)
    n_1d = np.zeros_like(d_obs_1d)
    delta_d_obs = d_obs_1d[1]-d_obs_1d[0]
    for i,this_d_obs in enumerate(d_obs_1d):
        wh=np.where((np.abs(d_obs-this_d_obs)<(0.5*delta_d_obs)) & (n_noisy>0))
        if len(wh[0])==0: continue
        n_1d[i] = np.median(n_noisy[wh])
    # now interpolate n_1d onto 3d grid
    from scipy import interpolate
    f = interpolate.interp1d(d_obs_1d, n_1d)
    n_median = np.zeros_like(n_noisy)
    wh_ok_interp = np.where((d_obs>np.min(d_obs_1d))&(d_obs<np.max(d_obs_1d)))    
    n_median[wh_ok_interp] = f(d_obs[wh_ok_interp])

    weight = np.zeros_like(n_median)
    weight[n_noisy>12]=1.
    n_median *= weight
    #pl.figure(1); pl.clf(); pl.imshow(n_noisy[:,:,128], vmin=0,vmax=n_median.max()); pl.colorbar()
    #pl.figure(2); pl.clf(); pl.imshow(n_median[:,:,128], vmin=0,vmax=n_median.max()); pl.colorbar()
    #ipdb.set_trace()
    return n_median, weight

    
def num_to_delta(n_data, n_rand, fwhm_sm=2.0, delta_max=3., linear_bias=2.0):
    from scipy.ndimage import gaussian_filter
    sigma_sm = fwhm_sm/2.355

    # smooth rand
    #if (fwhm_sm>1.):    
    #    n_rand = gaussian_filter(n_rand, sigma_sm)

    # smooth data
    # tmpp, replace with a filter that knows about LSS and shot noise.
    if (fwhm_sm>1.):
        n_data = gaussian_filter(n_data, sigma_sm)

    delta = (n_data-n_rand)/n_rand
    delta /= linear_bias
    delta[n_rand==0]=0.
    delta[delta>delta_max]=delta_max
    #weight = n_rand/np.max(n_rand)
    #return delta, weight
    return delta

    
class grid3d(object):
    def __init__(self, hemi='south', 
                 nx=2**8, ny=2**8, nz=2**8, reso_mpc=16.0, 
                 #nx=3*2**7, ny=3*2**7, nz=3*2**7, reso_mpc=10.667, 
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
        # x,y,z are in Mpc and should be within the bounds of self.x1d, etc.
        ind_x = np.digitize(x, self.x1d)
        ind_y = np.digitize(y, self.y1d)
        ind_z = np.digitize(z, self.z1d)
        return ind_x, ind_y, ind_z


    def num_from_xyz(self, x, y, z):
        # x,y,z are in Mpc and should be within the bounds of self.x1d, etc.
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
    

def load_redmapper(hemi=None):
    d = fits.open(datadir+'dr8_run_redmapper_v5.10_lgt20_catalog.fit')[1].data
    #d = fits.open(datadir+'dr8_run_redmapper_v5.10_lgt5_catalog.fit')[1].data  

    # if desired, use only one hemisphere.
    if (hemi!=None):
        if (hemi.lower()=='north'): wh_hemi=np.where(np.abs(d['ra']-180.)<100.)[0]
        if (hemi.lower()=='south'): wh_hemi=np.where(np.abs(d['ra']-180.)>=100.)[0]
        d = d[wh_hemi]

    #wh_zspec = np.where(d['bcg_spec_z']>0)[0]
    # tmpp, the following cut gets only clusters with spec-z's, and within a z range.
    # this is a manual duplicate of the z-range cut in grid3d, which i don't like.
    # but if i don't put it here, i get clusters that are outside of my grid3d.
    wh_zspec = np.where((d['bcg_spec_z']>0.1)&(d['bcg_spec_z']<0.55))[0]
    d = d[wh_zspec]

    ra = d['ra']
    dec = d['dec']
    lam = d['lambda_chisq']
    z_lam = d['z_lambda']
    z_spec = d['bcg_spec_z']
    cluster_id = d['mem_match_id']

    # let's go ahead and get the galactic coordinates.
    from astropy.coordinates import FK5
    from astropy import units as u
    coord = FK5(ra=ra, dec=dec, unit=(u.deg, u.deg))
    l_gal = coord.galactic.l.rad
    b_gal = coord.galactic.b.rad
    phi_gal = l_gal
    th_gal = np.pi/2.-b_gal
    return {'ra':ra, 'dec':dec, 'lam':lam, 'id':cluster_id, 
            'z_lam':z_lam, 'z_spec':z_spec, 
            'l_gal':l_gal, 'b_gal':b_gal, 'phi_gal':phi_gal, 'th_gal':th_gal}

    
def gnomview_template(template, rm):
    hp.gnomview(template*1e6,rot=(8.+np.median(rm['phi_gal'])*180./np.pi, np.median(rm['b_gal'])*180./np.pi, 0), xsize=1000, ysize=1000,min=-2,max=2,reso=0.5, unit='uK')


def make_cl_planck_data():
    savename = datadir+'cl_planck_217.pkl'
    # get mask
    mask = load_planck_mask()
    mask_factor = np.mean(mask**2.)
    # get planck data
    planck = load_planck_data()
    cl_planck = hp.anafast(mask*planck, lmax=4000)/mask_factor
    l_planck = np.arange(len(cl_planck))
    pickle.dump((l_planck, cl_planck), open(savename,'w'))
    
    
def get_cl_theory(band):

    # C31 mask
    #d3000_clustered = {100:0., 143:32., 217:50., 'mb':0.}  
    #d3000_poisson = {100:220., 143:75., 217:60., 'mb':0.}
    #uk_arcmin_noise = {100:100., 143:45., 217:63., 'mb':55.}

    # C39 mask
    d3000_clustered = {100:0., 143:32., 217:200., 'mb':0.}  
    d3000_poisson = {100:220., 143:75., 217:50., 'mb':0.}
    uk_arcmin_noise = {100:110., 143:50., 217:70., 'mb':63.}
    
    cl_theory = np.zeros(lmax+1)
    # load CMB
    tmp = np.loadtxt(datadir+'planck2013_TableIICol4_lensedCls.dat')
    l_theory = np.concatenate([np.array([0,1]),tmp[:,0]])
    dl_cmb = np.concatenate([np.array([0.,0.]), tmp[:,1]/1e12])
    l_theory = l_theory[0:lmax+1]
    dl_cmb = dl_cmb[0:lmax+1]    
    cl_cmb = dl_cmb/l_theory/(l_theory+1.)*2.*np.pi
    cl_cmb[0] = 0.
    cl_theory += cl_cmb

    # Poisson
    this_d3000_poisson = d3000_poisson[band]
    cl_poisson = this_d3000_poisson/3000./3001.*2.*np.pi / 1e12
    cl_theory += cl_poisson

    # Clustered
    this_dl_clustered = d3000_clustered[band]*(l_theory/3000.)**(0.6)/1e12
    cl_clustered = this_dl_clustered/l_theory/(l_theory+1.)*2.*np.pi
    cl_clustered[0] = 0.
    cl_theory += cl_clustered
    
    # multiply sky power by planck beam
    bl, l_bl = load_planck_bl(band)
    assert np.min(l_theory)==np.min(l_bl)
    assert len(l_theory)==len(l_bl)
    cl_theory *= (bl**2.)
        
    # add white noise
    this_uk_arcmin_noise = uk_arcmin_noise[band]
    cl_noise = (this_uk_arcmin_noise * 1./60.*np.pi/180.)**2. / 1e12
    cl_theory += cl_noise
    
    return l_theory, cl_theory    
            


def toy_test_real_vs_harmonic(amp=100., nside=2**5, nexp=1000):

    # generate the signal map
    npix = hp.nside2npix(nside)
    signal = np.zeros(npix)
    ind_hpix = hp.ang2pix(nside, 0., 0., nest=False)
    signal[ind_hpix] = 1.
    # get autospectrum of signal
    lmax_tmp = 4*nside-1
    cl_signal = hp.anafast(signal, lmax=lmax_tmp)
    l = np.arange(len(cl_signal))

    # create white CL's with unit variance
    nl = np.ones_like(l, dtype=np.float)
    nl /= (np.sum(nl*(2.*l+1.))/4./np.pi)
    # create harmonic space weights.  the factor of 
    # 2L+1 is because we'll be dealing with 1d spectra rather than 2d.
    weight = (2.*l+1.)/nl

    est_real = []
    est_harm = []
    for i in range(nexp):
        print '%i/%i'%(i,nexp)
        # make noise map
        noise = hp.synfast(nl, nside)
        data = amp*signal + noise
        # get real space estimate of signal amplitude
        est_real.append(data[ind_hpix])
        # get harmonic space estimate of signal amplitude
        amp_tmp = np.sum(hp.anafast(signal, map2=data, lmax=lmax_tmp)*weight) / np.sum(cl_signal*weight)
        est_harm.append(amp_tmp)
    est_real = np.array(est_real)
    est_harm = np.array(est_harm)

    print ' '
    print ' mean(est_real): %0.2f'%est_real.mean()
    print ' mean(est_harm): %0.2f'%est_harm.mean()
    print '  var(est_real): %0.2f'%est_real.var()
    print '  var(est_harm): %0.2f'%est_harm.var()
    print '  VarRatio:   %0.2f'%(est_harm.var()/est_real.var())
    print '  std(est_real): %0.2f'%est_real.std()
    print '  std(est_harm): %0.2f'%est_harm.std()
    print '  StdRatio:   %0.2f'%(est_harm.std()/est_real.std())
    print ' '

    nbins=15
    pl.clf()
    pl.hist(est_real,bins=nbins, alpha=0.5)
    pl.hist(est_harm,bins=nbins, alpha=0.5)
    pl.legend(['real space','harmonic space'])
    pl.title('AMP=%0.3f, NEXP=%i, VarRatio=%0.3f'%(amp, nexp, est_harm.var()/est_real.var()))
    ipdb.set_trace()
        
        
def study_redmapper_lrg_3d(hemi='north'):
    # create 3d grid object
    grid = grid3d(hemi=hemi)
    
    # load SDSS data
    sdss = load_sdss_data_both_catalogs(hemi)
    
    # load redmapper catalog
    rm = load_redmapper(hemi=hemi)
    
    # get XYZ positions (Mpc) of both datasets
    x_sdss, y_sdss, z_sdss = grid.xyz_from_radecz(sdss['ra'], sdss['dec'], sdss['z'], applyzcut=False)
    x_rm, y_rm, z_rm = grid.xyz_from_radecz(rm['ra'], rm['dec'], rm['z_spec'], applyzcut=False)
    pos_sdss = np.vstack([x_sdss, y_sdss, z_sdss]).T
    pos_rm = np.vstack([x_rm, y_rm, z_rm]).T

    # build a couple of KDTree's, one for SDSS, one for RM.
    from sklearn.neighbors import KDTree
    tree_sdss = KDTree(pos_sdss, leaf_size=30)
    tree_rm = KDTree(pos_rm, leaf_size=30)

    lrg_counts = tree_sdss.query_radius(pos_rm, 100., count_only=True)
    pl.clf()
    pl.hist(lrg_counts, bins=50)
    
    
    ipdb.set_trace()


def study_redmapper_2d():
    # I just want to know the typical angular separation for RM clusters.
    # I'm going to do this in a lazy way.
    hemi = 'north'
    rm = load_redmapper(hemi=hemi)
    ra = rm['ra']
    dec = rm['dec']
    ncl = len(ra)
    dist = np.zeros((ncl, ncl))
    for i in range(ncl):
        this_ra = ra[i]
        this_dec = dec[i]
        dra = this_ra-ra
        ddec = this_dec-dec
        dxdec = dra*np.cos(this_dec*np.pi/180.)
        dd = np.sqrt(dxdec**2. + ddec**2.)
        dist[i,:] = dd
        dist[i,i] = 99999999.
    d_near_arcmin = dist.min(0)*60.
    pl.clf(); pl.hist(d_near_arcmin, bins=100)
    pl.title('Distance to Nearest Neighbor for RM clusters')
    pl.xlabel('Distance (arcmin)')
    pl.ylabel('N')
    fwhm_planck_217 = 5.5 # arcmin
    sigma = fwhm_planck_217/2.355
    frac_2sigma = 1.*len(np.where(d_near_arcmin>2.*sigma)[0])/len(d_near_arcmin)
    frac_3sigma = 1.*len(np.where(d_near_arcmin>3.*sigma)[0])/len(d_near_arcmin)
    print '%0.3f percent of RM clusters are separated by 2-sigma_planck_beam'%(100.*frac_2sigma)
    print '%0.3f percent of RM clusters are separated by 3-sigma_planck_beam'%(100.*frac_3sigma)    
    ipdb.set_trace()


def get_pairwise_velocities_one_hemi(hemi, kmax=0.1, rmax=50.):
    # create 3d grid object
    grid = grid3d(hemi=hemi)
    
    # load SDSS data
    sdss = load_sdss_data_both_catalogs(hemi)
    
    # load redmapper catalog
    rm = load_redmapper(hemi=hemi)
    
    # get XYZ positions (Mpc) of both datasets
    x_sdss, y_sdss, z_sdss = grid.xyz_from_radecz(sdss['ra'], sdss['dec'], sdss['z'], applyzcut=False)
    x_rm, y_rm, z_rm = grid.xyz_from_radecz(rm['ra'], rm['dec'], rm['z_spec'], applyzcut=False)
    pos_sdss = np.vstack([x_sdss, y_sdss, z_sdss]).T
    pos_rm = np.vstack([x_rm, y_rm, z_rm]).T

    # build a KDTree for SDSS LRG's.
    from sklearn.neighbors import KDTree
    tree_sdss = KDTree(pos_sdss, leaf_size=30)
    # find those RM clusters that have some number of LRG's within X Mpc.
    #rmax = 300. # Mpc
    lrg_counts = tree_sdss.query_radius(pos_rm, rmax, count_only=True)
    ind, dist = tree_sdss.query_radius(pos_rm, rmax, count_only=False, return_distance=True)    
    min_counts = np.percentile(lrg_counts, 10)
    #min_counts = 500.
    #wh_use = np.where(lrg_counts>min_counts)[0]
    #for k in rm.keys(): rm[k] = rm[k][wh_use]
    #lrg_counts = lrg_counts[wh_use]
    #ind = ind[wh_use]
    #dist = dist[wh_use]
    #pos_rm = pos_rm[wh_use, :]
        
    # loop over RM clusters, get vlos
    ncl = len(rm['ra'])
    vlos = np.zeros(ncl)    
    rmin = 5.#Mpc, tmpp, worth exploring
    #r_pivot = 10.
    #r_decay = 10.

    redshift_grid = np.arange(0.05, 0.7, 0.01)
    rfine = np.arange(rmin-1, rmax+1,1.)
    # create a dictionary containing interpoltor objects, keyed on redshift
    corr_delta_vel_dict = {}
    from scipy import interpolate
    for redshift in redshift_grid:
        corr_delta_vel_dict[redshift] = interpolate.interp1d(rfine, corr_delta_vel(rfine, z=redshift, kmax=kmax))


    #distance_weight = 
    print '*********** using kmax=%0.2f, rmax=%i'%(kmax, rmax)
    for i in range(ncl):
        print i,ncl
        if (lrg_counts[i]<min_counts): continue
        wh_not_too_close = np.where(dist[i]>rmin)[0]        
        these_dist = dist[i][wh_not_too_close]
        these_ind = ind[i][wh_not_too_close]
        # get 3d positions
        these_pos_sdss = pos_sdss[these_ind, :]
        this_pos_rm = pos_rm[i, :]

        # dot with line of sight
        these_dot_los = dot_los(this_pos_rm, these_pos_sdss)
        this_redshift = rm['z_spec'][i]
        closest_redshift = redshift_grid[np.argmin(np.abs(redshift_grid-this_redshift))]
        this_corr_delta_vel = corr_delta_vel_dict[closest_redshift]
        these_vel = this_corr_delta_vel(these_dist)
        #ipdb.set_trace()
        #these_vel = corr_delta_vel(these_dist, z=this_redshift, kmax=kmax)
        #these_vel = np.exp(-(these_dist-r_pivot)/r_decay)
        #these_vel = np.exp(-0.5*(these_dist/r_decay)**2.)
        these_vlos = these_vel*these_dot_los
        this_vlos = np.sum(these_vlos) #tmpp, sum or mean?
        #indsort=np.argsort(these_dist)
        #pl.clf(); pl.plot(these_dist[indsort], np.cumsum(these_vlos[indsort]),'.')
        #ipdb.set_trace()
        vlos[i] = this_vlos
    rm['vlos'] = vlos
    rm['weight'] = np.ones(ncl)
    return rm


def dot_los(this_pos, these_pos):
    r_diff = these_pos-this_pos
    #tmpp, note that the following differs slightly
    # from definition given in Keisler&Schmidt.
    # should be very similar, but need to check.
    r_avg = 0.5*(this_pos+these_pos) 
    r_diff_hat = r_diff / np.sqrt((r_diff**2.).sum(1))[:, np.newaxis]
    r_avg_hat = r_avg / np.sqrt((r_avg**2.).sum(1))[:, np.newaxis]
    these_dot_los = (r_diff_hat*r_avg_hat).sum(1)
    return these_dot_los


def dot_los2(this_pos, these_pos):
    r_diff = these_pos-this_pos
    #tmpp, note that the following differs slightly
    # from definition given in Keisler&Schmidt.
    # should be very similar, but need to check.
    r_avg = this_pos[np.newaxis,:]
    r_diff_hat = r_diff / np.sqrt((r_diff**2.).sum(1))[:, np.newaxis]
    r_avg_hat = r_avg / np.sqrt((r_avg**2.).sum(1))[:, np.newaxis]
    these_dot_los = (r_diff_hat*r_avg_hat).sum(1)
    return these_dot_los

    
def study_multiband_planck(quick=True):
    savename = datadir+'cl_multiband.pkl'
    bands = [100, 143, 217, 'mb']
    if quick: cl = pickle.load(open(savename,'r'))
    else:
        cl = {}
        mask = load_planck_mask()
        mask_factor = np.mean(mask**2.)
        for band in bands:
            this_map = load_planck_data(band)
            this_cl = hp.anafast(this_map*mask, lmax=lmax)/mask_factor
            cl[band] = this_cl
        pickle.dump(cl, open(savename,'w'))


    cl_theory = {}
    pl.clf()
    
    for band in bands:
        l_theory, cl_theory[band] = get_cl_theory(band)
        this_cl = cl[band]
        pl.plot(this_cl/cl_theory[band])
        
    pl.legend(bands)
    pl.plot([0,4000],[1,1],'k--')
    pl.ylim(.7,1.3)
    pl.ylabel('data/theory')

    

def make_multiband_map(quick=True):
    savename = datadir+'map_mb.pkl'
    if quick: return pickle.load(open(savename,'r'))
    
    mask = load_planck_mask()
    mask_factor = np.mean(mask**2.)    
    bands=[100, 143, 217]

    # first calculate the band-dependent, ell-dependent weights
    weight = {}
    cl_theory = {}
    total_weight = 0.
    for band in bands:
        # load biased, beam-convolved spectrum
        l_theory, this_cl_theory = get_cl_theory(band)
        # debias (correct for beam)
        bl, l_bl = load_planck_bl(band)
        this_cl_theory /= (bl**2.)
        cl_theory[band] = this_cl_theory
        weight[band] = 1./this_cl_theory
        total_weight += 1./this_cl_theory
    for k in weight.keys(): weight[k]/=total_weight

    alm_multiband = 0.
    for band in bands:
        # load map        
        this_map = load_planck_data(band)
        
        # get alm's
        this_cl, this_alm = hp.anafast(this_map*mask, lmax=lmax, alm=True)
        # you might be tempted to correct alm by sqrt(mask_factor), but we're going to
        # transfer back to map space anyway.  we'll account for the mask later.

        # debias these alm's (i.e. correct for one power of beam)
        bl, l_bl = load_planck_bl(band)
        assert len(bl)==(lmax+1)
        this_alm = hp.sphtfunc.almxfl(this_alm, 1./bl, inplace=False)
        # multiply by ell-dependent weights
        this_alm = hp.sphtfunc.almxfl(this_alm, weight[band], inplace=False)
        alm_multiband += this_alm

    # apply a nominal multi-band beam, which we'll take to be a FWHM=5' gaussian
    bl_mb, l_bl = load_planck_bl('mb')
    alm_multiband = hp.sphtfunc.almxfl(alm_multiband, bl_mb, inplace=False)
    
    # transfer back to real space
    map_mb = hp.sphtfunc.alm2map(alm_multiband, nside, lmax=lmax)

    # save
    pickle.dump(map_mb, open(savename,'w'))
    return map_mb


def view_planck_likelihood_masks():
    tmp = fits.open(datadir+'COM_Mask_Likelihood_2048_R1.10.fits')[1].data
    for k in ['CL31','CL39','CL49']:
        hp.mollview(hp.reorder(tmp[k], n2r=True), title=k)

def compare_unbiased_spectra():
    bands = [100, 143, 217, 'mb']
    pl.clf()
    leg=[]

    l_theory, cl_mb = get_cl_theory('mb')
    bl, l_bl = load_planck_bl('mb')
    cl_mb /= (bl**2.)
    
    for band in bands:
        l_theory, this_cl_theory = get_cl_theory(band)
        bl, l_bl = load_planck_bl(band)
        this_cl_theory /= (bl**2.)
        #pl.plot(l_theory, this_cl_theory/cl_mb, lw=2)
        pl.semilogy(l_theory, this_cl_theory, lw=2)        
        leg.append(band)
    pl.ylim(8e-16,1e-14)
    #pl.ylim(.1,2)
    pl.legend(leg)
        

def plot_optimal_delta_filter():
    # see Ho&Spergel 0903.2845
    #nbar = 3e-4 / (0.7**3.)
    nbar = 3e-4 / (0.7**3.) / 10.    
    k = np.arange(1e-3,1,1e-3)
    zmean = 0.3
    from cosmolopy import perturbation
    pk = perturbation.power_spectrum(k, zmean, **cosmo)
    bias = 2.0
    w = pk*bias**2. / (pk*bias**2. + 1./nbar)
    pl.clf(); pl.plot(k, w)
    reso_mpc = 16.
    fwhm = 1.2*reso_mpc
    ipdb.set_trace()

def study_sdss_density(hemi='south'):
    grid = grid3d(hemi=hemi)
    n_data = num_sdss_data_both_catalogs(hemi, grid)
    n_rand, weight = num_sdss_rand_both_catalogs(hemi, grid)
    n_rand *= ((n_data*weight).sum() / (n_rand*weight).sum())
    delta = (n_data - n_rand) / n_rand
    delta[weight==0]=0.
    fdelta = np.fft.fftn(delta*weight)
    power = np.abs(fdelta)**2.
    ks = get_wavenumbers(delta.shape, grid.reso_mpc)
    kmag = ks[3]
    kbin = np.arange(0,0.06,0.002)
    ind = np.digitize(kmag.ravel(), kbin)
    power_ravel = power.ravel()
    power_bin = np.zeros_like(kbin)
    for i in range(len(kbin)):
        print i
        wh = np.where(ind==i)[0]
        power_bin[i] = power_ravel[wh].mean()
    #pl.clf()
    #pl.plot(kbin, power_bin)
    from cosmolopy import perturbation
    pk = perturbation.power_spectrum(kbin, 0.4, **cosmo)
    pl.clf(); pl.plot(kbin, power_bin/pk, 'b')
    pl.plot(kbin, power_bin/pk, 'bo')    
    pl.xlabel('k (1/Mpc)',fontsize=16)
    pl.ylabel('P(k) ratio, DATA/THEORY [arb. norm.]',fontsize=16)
    ipdb.set_trace()


    
def spherical_bessel_j1(x):
    return np.sin(x)/x**2. - np.cos(x)/x


def corr_delta_vel(r, z=0.4, kmin=1e-3, kmax=0.2):
    # r is in Mpc
    a = 1./(1.+z)
    hz = distance.hubble_z(z, **cosmo)
    fgrowth = perturbation.fgrowth(z, cosmo['omega_M_0'])
    k = np.arange(kmin,kmax,kmin)
    dk = k[1]-k[0]
    corr = []
    pk = perturbation.power_spectrum(k, z, **cosmo)
    for this_r in r:
        this_corr = a*hz*fgrowth/2./np.pi**2. * np.sum(dk*k*pk*spherical_bessel_j1(k*this_r))
        corr.append(this_corr)
    return np.array(corr)

def plot_many_corr_delta_vel():
    pl.clf()
    leg = []
    for kmax in [0.05, 0.1, 0.2, 0.5, 1., 2., 5.]:
        plot_corr_delta_vel(kmin=1e-3, kmax=kmax, doclf=False)
        leg.append('kmax=%0.2f'%kmax)
    pl.legend(leg)

def plot_corr_delta_vel(kmin=1e-3, kmax=0.2, doclf=True):
    r = np.arange(1.,200,1)
    corr = corr_delta_vel(r, kmin=kmin, kmax=kmax)
    #corr /= np.max(corr)    
    if doclf: pl.clf()
    pl.plot(r, corr)
    fs = 18
    pl.xlabel('r (Mpc)', fontsize=fs-2)
    pl.ylabel(r'$\xi_{\delta v}(r)$', fontsize=fs)
