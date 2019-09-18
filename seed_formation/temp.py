import scipy.stats as stats
import astropy.io.fits as fits
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
def select_eboss_sgc_good(fn='./eBOSS_ELG_full_ALLdata-vDR16.fits'):
    '''
    eboss elg sgc with good redshift
    input: filename
    output: array of objects w/ sgc
    '''
    dat = fits.getdata(fn)
    sel = (dat['z_ok']==1 )&((dat['chunk']=='eboss21')|(dat['chunk']=='eboss22'))&(dat['Z']>0)&(dat['Z']<2)&(dat['WEIGHT_CP']*dat['WEIGHT_NOZ']<20)
    return np.array(dat[sel]['Z']),np.array(dat[sel]['WEIGHT_CP']*dat[sel]['WEIGHT_NOZ'])

def select_eboss_sgc_good_all(fn='./eBOSS_ELG_full_ALLdata-vDR16.fits'):
    dat = fits.getdata(fn)
    sel = (dat['z_ok']==1 )&((dat['chunk']=='eboss21')|(dat['chunk']=='eboss22'))&(dat['Z']>0)&(dat['Z']<2)&(dat['WEIGHT_CP']*dat['WEIGHT_NOZ']<20)
    return sel
def gmm_gen(dat, n_components=10):
    '''
    generate gmm parameters with a set of data
    '''
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", tol=0.001)
    gmm = gmm.fit(X=np.expand_dims(dat, 1))
    print(gmm.weights_)
    print(gmm.means_)
    print(gmm.covariances_)
    np.savetxt('./eboss_weights.txt',gmm.weights_)
    np.savetxt('./eboss_means.txt',gmm.means_)
    np.savetxt('./eboss_covariances.txt',gmm.covariances_)
    return gmm

def plot(fn = './eBOSS_ELG_full_ALLdata-vDR16.fits'):
    '''
    check how GMM model works by making plots
    '''
    dat,weight = select_eboss_sgc_good()
    gmm=gmm_gen(dat)
    sums=None
    weights = np.loadtxt('eboss_weights.txt')
    mean = np.loadtxt('eboss_means.txt')
    cov = np.loadtxt('eboss_covariances.txt')
    faxis = np.arange(0,2,0.005)
    #import pdb;pdb.set_trace()
    plt.hist(np.array(dat),normed=True,weights=np.array(weight),bins=150,alpha=0.5,color='red',label='eboss sgc')
    for i in range(len(mean)):
       if sums is None:
           sums=weights[i]*stats.norm.pdf(faxis,mean[i],np.sqrt(cov[i])).ravel()
       else:
           sums+=weights[i]*stats.norm.pdf(faxis,mean[i],np.sqrt(cov[i])).ravel()
    plt.plot(faxis,sums,label = 'GMM',color = 'cyan')
    #import pdb;pdb.set_trace()
    plt.hist(gmm.sample(1000000)[0],bins=100,linewidth=3,histtype='step',normed=True,label='sampling')
    plt.legend()
    plt.show()

def get_decals_sources(dat,idx):
    '''
    try to look up source in tractor catalog, to locate the rhalf
    '''
    #import pdb;pdb.set_trace()
    topdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr3/tractor/'
    tractor_all = fits.getdata(topdir + dat['brickname'][idx][:3]+'/'+'tractor-'+dat['brickname'][idx]+'.fits')
    try:
        tractor = tractor_all[dat['decals_objid'][idx]]
    except:
        return None,None
    try:
        assert(dat['decals_objid'][idx]==tractor['objid'])
    except:
        print('objid unequal')
        return None,None
    try:
       assert((tractor['ra']-dat['ra'][idx]>-1./3600.)&(tractor['ra']-dat['ra'][idx]<1./3600.))
    except:
        #print(dat['decals_uniqid'][idx])
        #print(tractor['ra']-dat['ra'][idx])
        #print(idx)
        #print(tractor['ra'],tractor['dec'],22.5 - 2.5 * np.log10(tractor['decam_flux'][1] / tractor['decam_mw_transmission'][1]),tractor['decam_flux'][1],tractor['decam_mw_transmission'][1])
        #print(dat['ra'][idx],dat['dec'][idx],dat['g'][idx])
        return None,None
    try:
        assert((tractor['dec']-dat['dec'][idx]>-2./3600.)&(tractor['dec']-dat['dec'][idx]<2./3600.))
    except:
        #print(dat['decals_uniqid'][idx])
        #print(tractor['dec']-dat['dec'][idx])
        #print(idx)
        #print(tractor['ra']-dat['ra'][idx])
        #print(idx)
        #print(tractor['ra'],tractor['dec'],22.5 - 2.5 * np.log10(tractor['decam_flux'][1] / tractor['decam_mw_transmission'][1]))
        #print(dat['ra'][idx],dat['dec'][idx],dat['g'][idx])
        return None,None
    return tractor['shapeexp_r'],tractor['shapedev_r']

def get_decals_sources_manually(dat,idx):
    '''
    try to look up source in tractor catalog, to locate the rhalf, find it manually
    '''

    topdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr3/tractor/'
    tractor = fits.getdata(topdir + dat['brickname'][idx][:3]+'/'+'tractor-'+dat['brickname'][idx]+'.fits')
    ra = dat['ra'][idx]
    dec = dat['dec'][idx]
    sel = (tractor['ra']-ra<1.5/3600)&(tractor['ra']-ra>-1.5/3600)&(tractor['dec']-dec>-1.5/3600)&(tractor['dec']-dec<1.5/3600)
    if sel.sum()==0:
        print('did not find matched source successfully, set rhalf to 0.5, same as simp')
        return 0.45,0.45
    else:
        sel_min = ((tractor[sel]['ra']-ra)**2+(tractor[sel]['dec']-dec)**2).min()
        sel_id = np.where((tractor[sel]['ra']-ra)**2+(tractor[sel]['dec']-dec)**2==sel_min)
        #import pdb;pdb.set_trace()
        return tractor[sel][sel_id[0][0]]['shapeexp_r'],tractor[sel][sel_id[0][0]]['shapedev_r']
fn = './eBOSS_ELG_full_ALLdata-vDR16.fits'
sel = select_eboss_sgc_good_all(fn)
DAT = fits.getdata(fn)[sel]
def get_decals_sources_wrapper(idx):
    #fn = './eBOSS_ELG_full_ALLdata-vDR16.fits'
    #sel = select_eboss_sgc_good_all(fn)
    dat = DAT
    exp,dev = get_decals_sources(dat,idx)
    if exp is None:
        exp,dev = get_decals_sources_manually(dat,idx)
    return exp,dev

def tss_spectra_generation(fn = './eBOSS_ELG_full_ALLdata-vDR16.fits'):
    '''
    make eboss elg spectra with a eboss elg file, cut to sgc, good z
    write the output spectra to a csv file, format required by obiwan
    '''
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    sel = select_eboss_sgc_good_all(fn)
    dat = fits.getdata(fn)[sel]
    exp_r = []
    dev_r = []
    from multiprocessing import Pool
    p = Pool(64)
    outputs = p.map(get_decals_sources_wrapper,np.arange(len(dat)))
    #import pdb;pdb.set_trace()
    outputs = np.array(outputs)
    exp_r = np.clip(outputs[:,0],0.1,5)
    dev_r = np.clip(outputs[:,1],0.1,5)
    print(exp_r)
    print(dev_r)
    '''
    for i in range(64):
        for j in range(len(outputs[ji])):
            exp_r.append(outputs[i][0])
            dev_r.append(outputs[i][1])
    '''
    '''
    for i in range(len(dat)):
       if i%1000==0:
           print(i)
       exp,dev = get_decals_sources(dat, i)
       if exp is None:
           exp,dev = get_decals_sources_manually(dat, i)
       exp_r.append(exp)
       dev_r.append(dev)

    exp_r = np.arange(exp)
    dev_r = np.arange(dev)
    '''
    #r_half = decals_dat[idx2]['r_half']
    profile = dat['type']
    EXP = ((profile=='EXP')|(profile=='PSF')|(profile=='SIMP'))
    DEV = ~EXP
    #print(len(dat),len(dat[idx1]))
    dat_exp = dat[EXP]
    dat_dev = dat[DEV]

    all_data_exp = {'tractor_id':dat_exp['EBOSS_TARGET_ID'],'g':dat_exp['g'],'r':dat_exp['g'] - dat_exp['gr'],'z':dat_exp['g']-dat_exp['gr']-dat_exp['rz'],'fwhm_or_rhalf':exp_r[EXP],'redshift':dat_exp['Z']}
    tss_exp = pd.DataFrame(data = all_data_exp)
    tss_exp.to_csv('eboss_elg_tsspectra_EXP.csv')
    all_data_dev = {'tractor_id':dat_dev['EBOSS_TARGET_ID'],'g':dat_dev['g'],'r':dat_dev['g'] - dat_dev['gr'],'z':dat_dev['g']-dat_dev['gr']-dat_dev['rz'],'fwhm_or_rhalf':dev_r[DEV],'redshift':dat_dev['Z']}
    tss_dev = pd.DataFrame(data = all_data_dev)
    tss_dev.to_csv('eboss_elg_tsspectra_DEV.csv')
tss_spectra_generation()
