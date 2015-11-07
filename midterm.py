import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import kde
import astropy.coordinates as coord
import astropy.units as u
import astropy.io.ascii as ascii
import operator

#Some handy unit conversions
#days to seconds
daytosec=86400.0
#AU to meters
autom=149597870700.0
#Degrees to radians
degtorad=np.pi/180.0
#Mass of the sun in kg
msun=1.989e30
#Mass of Jupiter in kg
mjup=1.898e27
#Radius of Sun to meters
rsun=6.963e8
#Radius of Jupiter to meters
rjup=6.9911e7
#mas to degrees
mastodeg=1/(3.6e6)
#gravitational constant in SI
g=6.67408e-11

#HD 209458 Mass = 1.148 Msun

m1=1.148*msun

#HD 209458 Text file columns; HJD in days, RV in m s^-1, RV_unc in m s^-1
#Real Orbital Parameters from exoplanet.eu Wang & Ford 2011: P=3.52472, Msini=0.69 Mj, T0=2453344.76824

#Read in RV data
rvdata = ascii.read("/Users/ram/Dropbox/ut/2015fa/ast381/midterm/hd209458_rv.txt",delimiter='\s')

hjd=rvdata['col1']
rv=rvdata['col2']
rverr=rvdata['col3']

#HJD ranges from 2451341.1209 - 2453370.6986; let t0 vary between 1341.1209 and 3370.6986 and add 2450000

#Make initial guesses
t0=2453344.75#2453344.76824
m2sini0=0.75#0.69
p0=3.525#3.52472

#Number of jumps
nj=250000

#Empty arrays for the parameters and Chi^2 values
t=np.zeros(nj)
m2sini=np.zeros(nj)
p=np.zeros(nj)
chi2=np.zeros(nj)
#Count when a parameter is varied and when it is accepted
par_a=np.zeros(3)
par_s=np.zeros(3)
npar=np.zeros(3)

t[0]=t0
m2sini[0]=m2sini0
p[0]=p0

#Find an initial value for Chi^2 based on initial guesses
ph0_arr=((hjd-t[0])%p[0])/p[0]
#Do not want possible RM Effect RV anomaly to skew fit
fit_ph0=np.where((ph0_arr < 0.975) & (ph0_arr > 0.025))
k0=((2.0*np.pi*g/(p[0]*daytosec))**(1.0/3.0))*m2sini[0]*mjup/(m1**(2.0/3.0))
rv0=-1.0*k0*np.sin(2*np.pi*ph0_arr[fit_ph0])
chi2[0]=np.sum((rv0-rv[fit_ph0])**(2.0)/rverr[fit_ph0]**2.0)

#Jump sizes - tweaked these until ~30% acceptance rate was achieved
tj=(np.max(hjd)-np.min(hjd))*6e-6
mj=m2sini0*3e-2
pj=p0*1e-5

for i0 in range(1,nj):
	t[i0]=t[i0-1]
	m2sini[i0]=m2sini[i0-1]
	p[i0]=p[i0-1]
	chi2[i0]=chi2[i0-1]
	#randomly choose a parameter to vary
	test=len(npar)*np.random.uniform()
	if test < 1.0:
		t[i0]=t[i0-1]+np.random.randn()*tj
		par_s[0]=par_s[0]+1
	if 1.0 <= test < 2.0:
		m2sini[i0]=m2sini[i0-1]+np.random.randn()*mj
		par_s[1]=par_s[1]+1
	if 2.0 <= test < 3.0:
		p[i0]=p[i0-1]+np.random.randn()*pj
		par_s[2]=par_s[2]+1
	npar[0]=t[i0]
	npar[1]=m2sini[i0]
	npar[2]=p[i0]
	#Don't want the time of transit to be past the available data - if jumps there, reject the proposition
	#Don't want negative mass - if jumps there reject the proposition
	#Don't want a negative period - if jumps there refect the proposition
	if (npar[0] > np.max(hjd)) or (npar[1] <= 0.0) or (npar[2] <=0.0):
		t[i0]=t[i0-1]
		m2sini[i0]=m2sini[i0-1]
		p[i0]=p[i0-1]
		chi2[i0]=chi2[i0-1]
	else:
		#Phase data based on this period and time of transit
		mc_phase=((hjd-npar[0])%npar[2])/npar[2]
		#Since there are signs of the RM Effect, will not fit data points within 0.025 phase of transit
		fit_ph=np.where((mc_phase < 0.975) & (mc_phase > 0.025))
		#Calculate velocity semi-amplitude
		mc_k=((2.0*np.pi*g/(npar[2]*daytosec))**(1.0/3.0))*npar[1]*mjup/(m1**(2.0/3.0))
		#Calculate expected RVs based on these parameters
		mc_rv=-1.0*mc_k*np.sin(2*np.pi*mc_phase[fit_ph])
		#Calculated Chi^2 of observed data and model
		chi2[i0]=np.sum((mc_rv-rv[fit_ph])**(2.0)/rverr[fit_ph]**2.0)
		#Decide to accept or reject this proposition (based on Ford 2005 procedure)
		if chi2[i0] <= chi2[i0-1]:
			print i0,chi2[i0],t[i0],m2sini[i0],p[i0],'ACCEPTED!'
			par_a[np.floor(test)]=par_a[np.floor(test)]+1
		if chi2[i0] > chi2[i0-1]:
			prob=np.random.uniform()
			if prob <= np.exp(-1.0*(chi2[i0]-chi2[i0-1])/2.0):
				print i0,chi2[i0],t[i0],m2sini[i0],p[i0],'ACCEPTED!'
				par_a[np.floor(test)]=par_a[np.floor(test)]+1
			else:
				t[i0]=t[i0-1]
				m2sini[i0]=m2sini[i0-1]
				p[i0]=p[i0-1]
				chi2[i0]=chi2[i0-1]

print 'Acceptance Rates: ',100.0*par_a/par_s

#Burn-in occurs after ~200 jumps - will not incorporate those jumps for final parameter numbers

tfns_arr=t[200:]
mfns_arr=m2sini[200:]
pfns_arr=p[200:]
cfns_arr=chi2[200:]

tfin_arr=np.sort(t[200:])
mfin_arr=np.sort(m2sini[200:])
pfin_arr=np.sort(p[200:])

mvp,x_mvp,y_mvp=np.histogram2d(pfns_arr,mfns_arr,bins=10,range=[[np.min(pfns_arr),np.max(pfns_arr)],[np.min(mfns_arr),np.max(mfns_arr)]])
mvt,x_mvt,y_mvt=np.histogram2d(tfns_arr,mfns_arr,bins=10,range=[[np.min(tfns_arr),np.max(tfns_arr)],[np.min(mfns_arr),np.max(mfns_arr)]])
pvt,x_pvt,y_pvt=np.histogram2d(tfns_arr,pfns_arr,bins=10,range=[[np.min(tfns_arr),np.max(tfns_arr)],[np.min(pfns_arr),np.max(pfns_arr)]])

mvp_rav=np.ravel(mvp)
mvt_rav=np.ravel(mvt)
pvt_rav=np.ravel(pvt)

mvp_u,mvp_uni=np.unique(mvp_rav,return_index=True)
mvt_u,mvt_uni=np.unique(mvt_rav,return_index=True)
pvt_u,pvt_uni=np.unique(pvt_rav,return_index=True)

mvp_unq=mvp_rav[mvp_uni]
mvt_unq=mvt_rav[mvt_uni]
pvt_unq=pvt_rav[pvt_uni]

conf_mvp=np.zeros(len(mvp_unq))
conf_mvt=np.zeros(len(mvt_unq))
conf_pvt=np.zeros(len(pvt_unq))

mvp_sor=np.sort(mvp_unq)
mvt_sor=np.sort(mvt_unq)
pvt_sor=np.sort(pvt_unq)

for i1 in range(len(mvp_unq)):
	conf_mvp[i1]=np.sum(mvp_unq[i1:])/(nj-200)

for i1 in range(len(mvt_unq)):
	conf_mvt[i1]=np.sum(mvt_unq[i1:])/(nj-200)

for i1 in range(len(pvt_unq)):
	conf_pvt[i1]=np.sum(pvt_unq[i1:])/(nj-200)

mvp_1sig_ind=(np.abs(conf_mvp - 0.6827)).argmin()
mvp_2sig_ind=(np.abs(conf_mvp - 0.9545)).argmin()
mvp_3sig_ind=(np.abs(conf_mvp - 0.9973)).argmin()

mvt_1sig_ind=(np.abs(conf_mvt - 0.6827)).argmin()
mvt_2sig_ind=(np.abs(conf_mvt - 0.9545)).argmin()
mvt_3sig_ind=(np.abs(conf_mvt - 0.9973)).argmin()

pvt_1sig_ind=(np.abs(conf_pvt - 0.6827)).argmin()
pvt_2sig_ind=(np.abs(conf_pvt - 0.9545)).argmin()
pvt_3sig_ind=(np.abs(conf_pvt - 0.9973)).argmin()

mvp_levels=[mvp_unq[mvp_1sig_ind],mvp_unq[mvp_2sig_ind],mvp_unq[mvp_3sig_ind]]
mvt_levels=[mvt_unq[mvt_1sig_ind],mvt_unq[mvt_2sig_ind],mvt_unq[mvt_3sig_ind]]
pvt_levels=[pvt_unq[pvt_1sig_ind],pvt_unq[pvt_2sig_ind],pvt_unq[pvt_3sig_ind]]

maxis=np.linspace(np.min(mfin_arr),np.max(mfin_arr),10)
paxis=np.linspace(np.min(pfin_arr),np.max(pfin_arr),10)
taxis=np.linspace(np.min(tfin_arr),np.max(tfin_arr),10)

tmed=np.median(tfin_arr)
mmed=np.median(mfin_arr)
pmed=np.median(pfin_arr)

tm_ind=len(tfin_arr)/2
tu_ind=tm_ind+np.floor(0.34*len(tfin_arr)/2)
tl_ind=tm_ind-np.floor(0.34*len(tfin_arr)/2)
tu=tfin_arr[tu_ind]
tl=tfin_arr[tl_ind]

mm_ind=len(mfin_arr)/2
mu_ind=mm_ind+np.floor(0.34*len(mfin_arr)/2)
ml_ind=mm_ind-np.floor(0.34*len(mfin_arr)/2)
mu=mfin_arr[mu_ind]
ml=mfin_arr[ml_ind]

pm_ind=len(pfin_arr)/2
pu_ind=pm_ind+np.floor(0.34*len(pfin_arr)/2)
pl_ind=pm_ind-np.floor(0.34*len(pfin_arr)/2)
pu=pfin_arr[pu_ind]
pl=pfin_arr[pl_ind]

print tl,tmed,tu,tl-tmed,tu-tmed
print ml,mmed,mu,ml-mmed,mu-mmed
print pl,pmed,pu,pl-pmed,pu-pmed

#Plot best fit model
kfin=((2.0*np.pi*g/(pmed*daytosec))**(1.0/3.0))*mmed*mjup/(m1**(2.0/3.0))
phase_fin=((hjd-tmed)%pmed)/pmed

full_ph=np.arange(0,1001,dtype=float)/1000.0
rvcurve=-kfin*np.sin(2*np.pi*full_ph)

#trials=np.arange(1,nj+1,dtype=float)
#plt.plot(trials,chi2,'.')
#plt.savefig("chi2.eps")
#plt.clf()

plt.plot(full_ph,rvcurve)
plt.errorbar(phase_fin,rv,yerr=rverr,fmt='.')
plt.title("Best-fit Radial Velocity Curve for HD 209458")
plt.xlabel("Phase")
plt.ylabel("Radial Velocity (m/s)")
plt.savefig("rv.eps")
plt.clf()

plt.plot(pfns_arr,mfns_arr,'.')
plt.savefig("mvp.eps")
plt.clf()

plt.plot(tfns_arr,mfns_arr,'.')
plt.savefig("mvt.eps")
plt.clf()

plt.plot(tfns_arr,pfns_arr,'.')
plt.savefig("pvt.eps")
plt.clf()

cs_mvp=plt.contour(paxis,maxis,mvp,levels=mvp_levels)
plt.contour(paxis,maxis,mvp)
plt.clabel(cs_mvp,inline=1)
plt.title("Contour Plot of Msin(i) vs. Period")
plt.xlabel("Period (Days)")
plt.ylabel("Msin(i) (Mjup)")
plt.savefig("mvp_contour.eps")
plt.clf()

cs_mvt=plt.contour(taxis,maxis,mvt,levels=mvt_levels)
plt.contour(taxis,maxis,mvt)
plt.clabel(cs_mvt,inline=1)
plt.title("Contour Plot of Msin(i) vs. Transit Time")
plt.xlabel("Time of Transit (HJD)")
plt.ylabel("Msin(i) (Mjup)")
plt.savefig("mvt_contour.eps")
plt.clf()

cs_pvt=plt.contour(taxis,paxis,pvt,levels=pvt_levels)
plt.contour(taxis,paxis,pvt)
plt.clabel(cs_pvt,inline=1)
plt.title("Contour Plot of Period vs. Transit Time")
plt.xlabel("Time of Transit (HJD)")
plt.ylabel("Period (Days)")
#plt.plot(pfns_arr,mfns_arr,'.')
plt.savefig("pvt_contour.eps")
plt.clf()