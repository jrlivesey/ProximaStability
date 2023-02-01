/***********************************************************************
 *                                                                     *
 * PROX_HILL_STAB.C                                                    *
 *                                                                     *
 * Authors: Rory Barnes (rory@astro.washington.edu)                    *
 *          Joseph Livesey (jlivesey@uw.edu)                           *
 *                                                                     *
 * To compile: gcc -o proxhillstab prox_hill_stab.c -lm                *
 *                                                                     *
 * Based on the Hill stability code employed in Barnes & Greenberg     *
 * (2006). This code has been modified to compute the Hill stability   *
 * of many possible configurations of the inner Proxima Centauri       *
 * system (planets d & b).                                             *
 *                                                                     *
 * Original:                                                           *
 * https://github.com/RoryBarnes/HillStability/blob/master/hill_stab.c *
 *                                                                     *
 ***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>
#define dot(a, b)   (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
#define unif(a, b)  rand()/((double) RAND_MAX) * (b - a) + a

#define PI     3.14159265
#define BIGG	 6.672e-8
#define AUCM	 1.49598e13
#define MSUN	 1.98892e33
#define MJUP   1.8987e30
#define MEARTH 5.9722e24

typedef struct {
  double a,e,i,lasc,aper,mean_an;
} ELEMS;

char *sLower(char cString[]) {
  int iPos;
  for (iPos=0;cString[iPos];iPos++)
    cString[iPos] = tolower(cString[iPos]);

  return cString;
}



void hel_bar(double **hel,double **bar,double *ms,double *m,int n) {
   int i,p;

   for(i=1;i<=3;i++)
      bar[0][i] = 0;
   for(p=1;p<=n;p++) {
      for(i=1;i<=3;i++)
         bar[0][i] -= m[p]/ms[n]*hel[p][i];
   }
   for(p=1;p<=n;p++) {
      for(i=1;i<=3;i++)
         bar[p][i] = hel[p][i]+bar[0][i];
   }
}



void set_vals(ELEMS *p1, ELEMS *p2, double *m, int *origin, bool ecc_only) {
  double obs_inc;
  /* This function was added for the Proxima paper. Sets random inputs instead
  reading them from an input file. */

  if (ecc_only) {
    /* Varying eccentricity only */
    *origin  = 1;                     // body-centric coordinate system
    obs_inc  = 133.; // Sky plane inclination (range 3–33 deg, centered on Proxima c inclination: 18º)
    p1->a    = 0.029;               // Planet d semi-major axis
    p1->e    = unif(0., 0.9);                  // Planet d eccentricity
    p1->aper = 0.;          // Planet d argument of pericenter (any)
    p1->i    = 0.;           // Planet d inclination (range 0–15 deg)
    p1->lasc = 0.;          // Planet d longitude of ascending node (any)
    p1->mean_an = 0.;
    p2->a    = 0.049;               // Planet b semi-major axis
    p2->e    = unif(0., 0.9);                  // Planet b eccentricity
    p2->aper = 180.;          // Planet b argument of pericenter (any)
    p2->i    = 0.;           // Planet b inclination (range 0–15 deg)
    p2->lasc = 0.;          // Planet b longitude of ascending node (any)
    p2->mean_an = 180.;
    m[0]     = 0.12;                  // Star mass (in solar masses)
    m[1]     = 0.26 / sin(obs_inc);   // Proxima d mass (in earth masses, M*sin(i) = 0.26 ± 0.05)
    m[2]     = 1.07 / sin(obs_inc);   // Proxima b mass (in earth masses, M*sin(i) = 1.07 ± 0.06)
  } else {
    /* Full parameter sweep */
    *origin  = 1;                     // body-centric coordinate system
    obs_inc  = (rand() % 30 + 3) * M_PI/180; // Sky plane inclination (range 3–33 deg, centered on Proxima c inclination: 18º)
    p1->a    = 0.029;               // Planet d semi-major axis
    p1->e    = unif(0., 0.9);                  // Planet d eccentricity
    p1->aper = unif(0., 360.);          // Planet d argument of pericenter (any)
    p1->i    = unif(0., 15.);           // Planet d inclination (range 0–15 deg)
    p1->lasc = unif(0., 360.);          // Planet d longitude of ascending node (any)
    p1->mean_an = unif(0., 360.);
    p2->a    = 0.049;               // Planet b semi-major axis
    p2->e    = unif(0., 0.9);                  // Planet b eccentricity
    p2->aper = unif(0., 360.);          // Planet b argument of pericenter (any)
    p2->i    = unif(0., 15.);           // Planet b inclination (range 0–15 deg)
    p2->lasc = unif(0., 360.);          // Planet b longitude of ascending node (any)
    p2->mean_an = unif(0., 360.);
    m[0]     = 0.12;                  // Star mass (in solar masses)
    m[1]     = unif(0.21, 0.31) / sin(obs_inc);   // Proxima d mass (in earth masses, M*sin(i) = 0.26 ± 0.05)
    m[2]     = unif(1.01, 1.13) / sin(obs_inc);   // Proxima b mass (in earth masses, M*sin(i) = 1.07 ± 0.06)
  }

  /* Varying inclination only */
  // *origin  = 1;                     // body-centric coordinate system
  // obs_inc  = 133.; // Sky plane inclination (range 3–33 deg, centered on Proxima c inclination: 18º)
  // p1->a    = 0.029;               // Planet d semi-major axis
  // p1->e    = 0.7;                  // Planet d eccentricity
  // p1->aper = 0.;          // Planet d argument of pericenter (any)
  // p1->i    = unif(0., PI);           // Planet d inclination (range 0–15 deg)
  // p1->lasc = 0.;          // Planet d longitude of ascending node (any)
  // p1->mean_an = 0.;
  // p2->a    = 0.049;               // Planet b semi-major axis
  // p2->e    = 0.7;                  // Planet b eccentricity
  // p2->aper = 180.;          // Planet b argument of pericenter (any)
  // p2->i    = unif(0., PI);           // Planet b inclination (range 0–15 deg)
  // p2->lasc = 0.;          // Planet b longitude of ascending node (any)
  // p2->mean_an = 180.;
  // m[0]     = 0.12;                  // Star mass (in solar masses)
  // m[1]     = 0.26 / sin(obs_inc);   // Proxima d mass (in earth masses, M*sin(i) = 0.26 ± 0.05)
  // m[2]     = 1.07 / sin(obs_inc);   // Proxima b mass (in earth masses, M*sin(i) = 1.07 ± 0.06)

  /* Unit conversions */
  p1->a    *= AUCM;
  p1->aper *= M_PI/180;
  p1->i    *= M_PI/180;
  p1->lasc *= M_PI/180;
  p2->a    *= AUCM;
  p2->aper *= M_PI/180;
  p2->i    *= M_PI/180;
  p2->lasc *= M_PI/180;
  m[0]     *= MSUN;
  m[1]     *= MEARTH;
  m[2]     *= MEARTH;
}



void cartes(double *x, double *v, ELEMS elem_ptr,double mu) {
  double a,e,m,cosi,sini,cos_lasc,sin_lasc,cos_aper,sin_aper;
  double es,ec,w,wp,wpp,wppp,ecc,dx,lo,up,next;int iter;
  double sin_ecc,cos_ecc,l1,m1,n1,l2,m2,n2;
  double xi,eta,vel_scl;

  a = elem_ptr.a;
  e = elem_ptr.e;
  m = elem_ptr.mean_an;
  cosi = cos(elem_ptr.i);
  sini = sin(elem_ptr.i);
  cos_lasc = cos(elem_ptr.lasc);
  sin_lasc = sin(elem_ptr.lasc);
  cos_aper = cos(elem_ptr.aper);
  sin_aper = sin(elem_ptr.aper);

  /*
   * Reduce mean anomoly to [0, 2*PI)
   */
  m -= ((int)(m/(2*M_PI)))*2*M_PI;
  /*
   * Solve kepler's equation.
   */
  if (sin(m)>0)
    ecc = m+0.85*e;
  else
    ecc = m-0.85*e;
  lo = -2*M_PI;
  up = 2*M_PI;
  for(iter=1;iter<=32;iter++) {
    es = e*sin(ecc);
    ec = e*cos(ecc);
    w = ecc-es-m;
    wp = 1-ec;
    wpp = es;
    wppp = ec;
    if (w>0)
      up = ecc;
    else
      lo = ecc;
    dx = -w/wp;
    dx = -w/(wp+dx*wpp/2);
    dx = -w/(wp+dx*wpp/2+dx*dx*wppp/6);
    next = ecc+dx;
    if (ecc==next)
      break;
    if ((next>lo) && (next<up))
      ecc= next;
    else ecc= (lo+up)/2;
    if((ecc==lo) || (ecc==up))
      break;
    if (iter>30)
      printf("%4d %23.20f %e\n",iter,ecc,up-lo);
  }
  if(iter>32) {
    fprintf(stderr,"ERROR: Kepler solultion failed.\n");
    exit(1);
  }

  cos_ecc = cos(ecc);
  sin_ecc = sin(ecc);

  l1 = cos_lasc*cos_aper-sin_lasc*sin_aper*cosi;
  m1 = sin_lasc*cos_aper+cos_lasc*sin_aper*cosi;
  n1 = sin_aper*sini;
  l2 = -cos_lasc*sin_aper-sin_lasc*cos_aper*cosi;
  m2 = -sin_lasc*sin_aper+cos_lasc*cos_aper*cosi;
  n2 = cos_aper*sini;

  xi = a*(cos_ecc-e);
  eta = a*sqrt(1-e*e)*sin_ecc;
  x[0] = l1*xi+l2*eta;
  x[1] = m1*xi+m2*eta;
  x[2] = n1*xi+n2*eta;
  vel_scl = sqrt((mu*a)/dot(x,x));
  xi = -vel_scl*sin_ecc;
  eta = vel_scl*sqrt(1-e*e)*cos_ecc;
  v[0] = l1*xi+l2*eta;
  v[1] = m1*xi+m2*eta;
  v[2] = n1*xi+n2*eta;
}



double GetExact(double **r,double **v,double *m) {
  double c,h,ke;
  double br[3][3],bv[3][3];
  double c1[3][3];
  double ctot[3];
  double p_a,crit;
  double mm,mtot;
  double r01,r02,r12;
  int i,p;

  for (i=0;i<3;i++) {
    for (p=0;p<3;p++) {
      br[p][i]=0;
      bv[p][i]=0;
    }
  }

  mtot=m[0]+m[1]+m[2];
  mm=m[0]*m[1] + m[0]*m[2] + m[1]*m[2];
  r01=sqrt(pow((r[0][0]-r[1][0]),2) + pow((r[0][1]-r[1][1]),2) +
        pow((r[0][2]-r[1][2]),2));
  r12=sqrt(pow((r[1][0]-r[2][0]),2) + pow((r[1][1]-r[2][1]),2) +
        pow((r[1][2]-r[2][2]),2));
  r02=sqrt(pow((r[0][0]-r[2][0]),2) + pow((r[0][1]-r[2][1]),2) +
        pow((r[0][2]-r[2][2]),2));

  /* Convert to barycentric coordinates */
  for (p=1;p<=2;p++) {
    for (i=0;i<3;i++) {
       br[0][i] -= m[p]/mtot*r[p][i];
       bv[0][i] -= m[p]/mtot*v[p][i];
    }
  }
  for (p=1;p<=2;p++) {
    for (i=0;i<3;i++) {
       br[p][i] = r[p][i]+br[0][i];
       bv[p][i] = v[p][i]+bv[0][i];
    }
  }
  /* Total energy and angular momentum */
  ke=0;
  for (p=0;p<3;p++) {
    c1[p][0]=m[p]*(br[p][1]*bv[p][2]-br[p][2]*bv[p][1]);
    c1[p][1]=m[p]*(br[p][2]*bv[p][0]-br[p][0]*bv[p][2]);
    c1[p][2]=m[p]*(br[p][0]*bv[p][1]-br[p][1]*bv[p][0]);
    for (i=0;i<3;i++)
      ke += 0.5*m[p]*v[p][i]*v[p][i];
  }

  for (i=0;i<3;i++) {
      ctot[i]=0;
      for (p=0;p<3;p++)
	  ctot[i] += c1[p][i];
  }

  h=ke-BIGG*(m[0]*m[1]/r01 + m[0]*m[2]/r02 + m[1]*m[2]/r12);
  c=sqrt(ctot[0]*ctot[0] + ctot[1]*ctot[1] + ctot[2]*ctot[2]);

  p_a = -2*c*c*h*mtot/(BIGG*BIGG*pow(mm,3));

  /* Note that M+B and G93 use m3 as the central mass, but we use m[0] */
  crit = 1 + pow(3,(4.0/3))*(m[1]*m[2])/
       (pow(m[0],(2.0/3))*(pow((m[1]+m[2]),(4.0/3))))
       - (m[1]*m[2]*(11*m[1]+7*m[2]))/(3*m[0]*(m[1]+m[2])*(m[1]+m[2]));

  return p_a/crit;
}



double GetApprox(ELEMS p1,ELEMS p2,double *m) {
  double mu[2],zeta,gamma[2],lambda,dTotMass;
  double p_a,crit;

  dTotMass = m[0] + m[1] + m[2];
  gamma[0] = sqrt(1 - p1.e*p1.e);
  gamma[1] = sqrt(1 - p2.e*p2.e);
  lambda = sqrt(p2.a/p1.a);
  mu[0] = m[1]/m[0];
  mu[1] = m[2]/m[0];
  zeta = mu[0]+mu[1];

  p_a = pow(zeta,-3)*(mu[0]+mu[1]/(lambda*lambda))*pow((mu[0]*gamma[0] + mu[1]*gamma[1]*lambda),2);
  crit = 1 + pow(3,(4./3))*mu[0]*mu[1]/pow(zeta,4./3);

  return p_a/crit;
}



void write_output(char filename[], int line_num, char exact[],
                  char approx[], char d_mass[], char b_mass[],
                  char d_semi[], char b_semi[], char d_ecc[],
                  char b_ecc[], char d_aper[], char b_aper[],
                  char d_inc[], char b_inc[], char d_lasc[],
                  char b_lasc[], char d_mean_an[], char b_mean_an[]) {
  /* This function was added for the Proxima paper. Outputs exact and
  approximate Hill stability & orbital elements to a data file. */
  FILE *fp;

  if (line_num == 0)
    fp = fopen(filename, "w+");
  else
    fp = fopen(filename, "a");
  fprintf(fp, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n", exact, approx,
          d_mass, b_mass, d_semi, b_semi, d_ecc, b_ecc, d_aper, b_aper, d_inc,
          b_inc, d_lasc, b_lasc, d_mean_an, b_mean_an);
  fclose(fp);
}



void calc_hill_stab(char filename[], bool ecc_only) {
  double **x, **v, *m;
  double **bax, **bav;
  double *ms;
  double ratio, e1, e2;
  ELEMS p1, p2;
  double exact, approx;
  char sExact[100], sApprox[100], d_mass[100], b_mass[100], d_semi[100],
       b_semi[100], d_ecc[100], b_ecc[100], d_aper[100], b_aper[100], d_inc[100],
       b_inc[100], d_lasc[100], b_lasc[100], d_mean_an[100], b_mean_an[100];
  int line_num;
  int i, j, origin;
  /* This function was re-factored for the Proxima paper. It performs the
  stability calculation and writes to a file, with the input parameters and
  target file determined by the value of "random". If random == 1, random
  orbital elements are used, and the output is written to proxhillstab.txt. If
  random == 0, the eccentricity space of Fig. 1 is used, and the output is
  written to proxhillstabecc.txt. */

  m=malloc(3*sizeof(double));
  x=malloc(3*sizeof(double*));
  v=malloc(3*sizeof(double*));
  bax=malloc(3*sizeof(double*));
  bav=malloc(3*sizeof(double*));
  ms=malloc(3*sizeof(double));
  for (i=0;i<3;i++) {
    x[i]=malloc(3*sizeof(double));
    v[i]=malloc(3*sizeof(double));
    bax[i]=malloc(3*sizeof(double));
    bav[i]=malloc(3*sizeof(double));
  }

  line_num = 0;
  for (j=0; j<1000000; j++) {
    // fprintf(stdout, "%i\n", line_num);
    set_vals(&p1, &p2, m, &origin, ecc_only);
    /* bodycentric coordinates */
    for (i=0;i<3;i++) {
      x[0][i]=0.0;
      v[0][i]=0.0;
    }
    cartes(x[1],v[1],p1,m[0]*BIGG);
    cartes(x[2],v[2],p2,m[0]*BIGG);
    if (origin) {
      ms[0]=m[0];
      for (i=1;i<3;i++)
        ms[i] = ms[i-1] + m[i];
      hel_bar(x,bax,ms,m,2);
    }
    exact=GetExact(x,v,m);
    approx=GetApprox(p1,p2,m);

    if (exact < 1e-4 || exact > 1e4)
      sprintf(sExact, "%.5e", exact);
    else
      sprintf(sExact, "%.5lf", exact);

    if (approx < 1e-4 || approx > 1e4)
      sprintf(sApprox, "%.5e", approx);
    else
      sprintf(sApprox, "%.5lf", approx);

    sprintf(d_mass, "%.5lf", m[1]);
    sprintf(b_mass, "%.5lf", m[2]);
    sprintf(d_semi, "%.5lf", p1.a);
    sprintf(b_semi, "%.5lf", p2.a);
    sprintf(d_ecc, "%.5lf", p1.e);
    sprintf(b_ecc, "%.5lf", p2.e);
    sprintf(d_aper, "%.5lf", p1.aper);
    sprintf(b_aper, "%.5lf", p2.aper);
    sprintf(d_inc, "%.5lf", p1.i);
    sprintf(b_inc, "%.5lf", p2.i);
    sprintf(d_lasc, "%.5lf", p1.lasc);
    sprintf(b_lasc, "%.5lf", p2.lasc);
    sprintf(d_mean_an, "%.5lf", p1.mean_an);
    sprintf(b_mean_an, "%.5lf", p2.mean_an);

    write_output(filename, line_num, sExact, sApprox, d_mass, b_mass, d_semi,
                 b_semi, d_ecc, b_ecc, d_aper, b_aper, d_inc, b_inc, d_lasc,
                 b_lasc, d_mean_an, b_mean_an);
    line_num += 1;
  }
}



int main(int argc, char *argv[]) {
  if (argc != 1) {
    fprintf(stderr,"Usage: %s\n", argv[0]);
    exit(1);
  }

  calc_hill_stab("proxhillstab.txt", 0);
  calc_hill_stab("proxhillstabecc.txt", 1);

  return 0;
}
