/******************************************************************************
 The computer software and associated documentation called STAMP hereinafter
 referred to as the WORK which is more particularly identified and described in 
 the LICENSE.  Conditions and restrictions for use of
 this package are also in the LICENSE.

 The WORK is only available to licensed institutions.

 The WORK was developed by: 
	Robert B. Russell and Geoffrey J. Barton

 Of current contact addresses:

 Robert B. Russell (RBR)             Geoffrey J. Barton (GJB)
 Bioinformatics                      EMBL-European Bioinformatics Institute
 SmithKline Beecham Pharmaceuticals  Wellcome Trust Genome Campus
 New Frontiers Science Park (North)  Hinxton, Cambridge, CB10 1SD U.K.
 Harlow, Essex, CM19 5AW, U.K.       
 Tel: +44 1279 622 884               Tel: +44 1223 494 414
 FAX: +44 1279 622 200               FAX: +44 1223 494 468
 e-mail: russelr1@mh.uk.sbphrd.com   e-mail geoff@ebi.ac.uk
                                     WWW: http://barton.ebi.ac.uk/

   The WORK is Copyright (1997,1998,1999) Robert B. Russell & Geoffrey J. Barton
	
	
	

 All use of the WORK must cite: 
 R.B. Russell and G.J. Barton, "Multiple Protein Sequence Alignment From Tertiary
  Structure Comparison: Assignment of Global and Residue Confidence Levels",
  PROTEINS: Structure, Function, and Genetics, 14:309--323 (1992).
*****************************************************************************/
/******************************************************************
---------------------------------------------------------

     Note that XA is ATOMS1 and XB is ATOMS2   G.J.B.

     SUBROUTINE TO FIT THE COORD SET ATOMS1(3,N) TO THE SET ATOMS2(3,N)
     IN THE SENSE OF:
            XA= R*XB +V
     R IS A UNITARY 3.3 RIGHT HANDED ROTATION MATRIX
     AND V IS THE OFFSET VECTOR. THIS IS AN EXACT SOLUTION

    THIS SUBROUTINE IS A COMBINATION OF MCLACHLAN'S AND KABSCH'S
    TECHNIQUES. SEE
     KABSCH, W. ACTA CRYST A34, 827,1978
     MCLACHAN, A.D., J. MOL. BIO. NNN, NNNN 1978
     WRITTEN BY S.J. REMINGTON 11/78.

     THIS SUBROUTINE USES THE IBM SSP EIGENVALUE ROUTINE 'EIGEN'

    N.B. CHANGED BY R.B.R (OCTOBER 1990) FOR USE IN THE PROGRAM
     STRUCALIGN.  SEE STRUCALIGN.F FOR DETAILS

    CHANGED BY R.B.R. (JANUARY 1991) FOR USE IN THE PROGRAM
       RIGIDBODY. 
-----------------------------------------------------------------
	Translated into C by the program F2C by RBR January 1993 

******************************************************************/

/* qkfit.f -- translated by f2c (version of 11 May 1990  14:38:39).
   You must link the resulting object file with the libraries:
	-lF77 -lI77 -lm -lc   (in that order)
*/

#include <stdio.h>
#include <f2c.h>

/* Table of constant values */

static integer c__3 = 3;
static integer c__0 = 0;

/* Subroutine */ 
int qkfit(doublereal *umat, doublereal *rtsum, doublereal *r, integer *entry_) {

    /* Initialized data */

    static doublereal eps = 1e-5;
    static doublereal pi = 3.14159265358979;
    static doublereal one = 1.;
    static doublereal two = 2.;
    static doublereal three = 3.;
    static doublereal half = .5;
    static doublereal third = .333333333;
    static doublereal forthr = 1.333333333;
    static struct {
       doublereal fill_1[1];
       doublereal e_2[2];
       doublereal fill_3[2];
       doublereal e_4;
       doublereal fill_5[3];
	} equiv_6 = { {0}, {0.0}, {0.0}, 0, {0.0} };


    /* System generated locals */
    integer i_1;
    static doublereal equiv_7[9];

    /* Builtin functions */
    double sqrt(), d_sign(), atan(), cos();

    /* Local variables */
    static doublereal diff;
    static integer isig;
    static doublereal detu, root[3];
#define a ((doublereal *)&equiv_6)
#define b (equiv_7)
    static integer i, j, k;
    static doublereal s, t;
    extern /* Subroutine */ int eigen_();
    static doublereal digav, theta, argsq, b1, b2;
    extern /* Subroutine */ int esort_();
    static doublereal cos3th, cc, b13, dd, b23;
    static integer ia;
    static doublereal b33, qq, rt;
#define usqmat ((doublereal *)&equiv_6)
#define aam ((doublereal *)&equiv_6)
#define bam ((doublereal *)&equiv_6 + 4)
#define cam ((doublereal *)&equiv_6 + 8)
#define fam ((doublereal *)&equiv_6 + 7)
#define gam ((doublereal *)&equiv_6 + 6)
#define ham ((doublereal *)&equiv_6 + 3)
    static doublereal du11, du21, du31;
#define utr (equiv_7)





    /* Parameter adjustments */
    r -= 4;
    umat -= 4;

    /* Function Body */
    isig = 1;
    if (*entry_ == 1) {
	goto L200;
    }
/*     CALC DET OF UMAT */

    du11 = umat[8] * umat[12] - umat[11] * umat[9];
    du21 = umat[11] * umat[6] - umat[5] * umat[12];
    du31 = umat[5] * umat[9] - umat[8] * umat[6];
    detu = umat[4] * du11 + umat[7] * du21 + umat[10] * du31;
    if (detu < 0.) {
	isig = -1;
    }

/*     FORM USQMAT AS POSITIVE SEMI DEFINITE MATRIX */

    for (j = 1; j <= 3; ++j) {
	i_1 = j;
	for (i = 1; i <= i_1; ++i) {
	    usqmat[i + j * 3 - 4] = umat[i * 3 + 1] * umat[j * 3 + 1] + umat[
		    i * 3 + 2] * umat[j * 3 + 2] + umat[i * 3 + 3] * umat[j * 
		    3 + 3];
/* L105: */
	}
/* L110: */
    }
/* %    WRITE(6,999) USQMAT */

/*     REDUCE AVG OF DIAGONAL TERMS TO ZERO */

    digav = (*aam + *bam + *cam) * third;
/* %    WRITE(6,999) DIGAV */
    *aam -= digav;
    *bam -= digav;
    *cam -= digav;

/*     SETUP COEFFS OF SECULAR EQUATION OF MATRIX WITH TRACE ZERO */

    cc = *fam * *fam + *gam * *gam + *ham * *ham - *aam * *bam - *bam * *cam 
	    - *cam * *aam;
    dd = *aam * *bam * *cam + two * (*fam * *gam * *ham) - *aam * *fam * *fam 
	    - *bam * *gam * *gam - *cam * *ham * *ham;

/*     THE SECULAR EQN IS Y**3-CC*Y-DD=0  AND DD IS DET(USQMAT) */
/*     REDUCE THIS TO THE FORM COS**3-(3/4)COS- */
/*     (1/4)COS3THETA = 0 */
/*     WITH SOLUTIONS COSTHETA.  SO Y=QQ*COSTHETA */

    if (cc <= eps) {
	goto L115;
    }
    qq = sqrt(forthr * cc);
    cos3th = three * dd / (cc * qq);
    if (abs(cos3th) > one) {
/*	cos3th = d_sign(&one, &cos3th);  */
/*      Change suggested by Andrew Torda with many thanks, etc. eliminates the need for the FORTRAN libraries */
	cos3th = (cos3th > 0 ? one:-one); 

    }

/*     FUNCTION ARCOS */

    if (cos3th != 0.) {
	goto L1200;
    }
/* L1100: */
    theta = (float)1.570796327;
    goto L1400;
L1200:
    argsq = cos3th * cos3th;
    theta = atan(sqrt((float)1. - argsq) / cos3th);
    if (cos3th < 0.) {
	theta = pi - theta;
    }
L1400:

/*     ROOTS IN ORDER OF SIZE GO 1,2,3 1 LARGEST */

    theta *= third;
    root[0] = qq * cos(theta);
    diff = half * sqrt(three * (qq * qq - root[0] * root[0]));
    root[1] = -root[0] * half + diff;
    root[2] = -root[0] * half - diff;
    goto L120;
L115:

/*     SPECIAL FOR TRIPLY DEGENERATE */

    root[0] = (float)0.;
    root[1] = (float)0.;
    root[2] = (float)0.;
L120:
/*     ADD ON DIGAV AND TAKE SQRT */
    for (j = 1; j <= 3; ++j) {
	rt = root[j - 1] + digav;
	if (rt < eps) {
	    rt = (float)0.;
	}
	root[j - 1] = sqrt(rt);
/* L125: */
    }
/* %    WRITE(6,999) ROOT */
/*     IF DETU IS <0 CHANGE SIGN OF ROOT(3) */
    if (isig == -1) {
	root[2] = -root[2];
    }
    *rtsum = root[0] + root[1] + root[2];
/* %    WRITE(6,999) RTSUM */
    return 0;

/*     THIS IS THE FANCY PART */

L200:

/*     FORM USQ = (UT).U    (IN UPPER TRIANGULAR SYMMETRIC STORAGE MODE) 
*/

    for (i = 1; i <= 3; ++i) {
	for (j = i; j <= 3; ++j) {
/* SMJS Changed to be like Robs version */
	    t = (doublereal)0.;
	    for (k = 1; k <= 3; ++k) {
		t += umat[k + i * 3] * umat[k + j * 3];
/* L210: */
	    }
	    ia = i + (j * j - j) / 2;
	    utr[ia - 1] = t;
/* L220: */
	}
    }
/* %    WRITE(6,999) UTR */

/*     CALCULATE EIGENVALUES AND VECTORS */

    eigen_(utr, a, &c__3, &c__0);
    esort_(utr, a, &c__3, &c__0);
/* %    WRITE(6,999) UTR */

    root[0] = utr[0];
    root[1] = utr[2];
    root[2] = utr[5];
/* %    WRITE(6,999) ROOT */
/* %    WRITE(6,999) A */

/*     SET A3 = A1 CROSS A2 */
/*     ROOTS ARE IN ORDER R(1) >= R(2) >= R(3) >= 0 */

    a[6] = a[1] * a[5] - a[2] * a[4];
    a[7] = a[2] * a[3] - a[0] * a[5];
    a[8] = a[0] * a[4] - a[1] * a[3];
/* %    WRITE(6,999) A */

/*     VECTOR SET B=U.A */

    for (i = 1; i <= 3; ++i) {
	for (j = 1; j <= 3; ++j) {
/* SMJS Changed to be like Robs version */
	    t = (doublereal)0.;
	    for (k = 1; k <= 3; ++k) {
/* L230: */
		t += umat[j + k * 3] * a[k + i * 3 - 4];
	    }
	    b[j + i * 3 - 4] = t;
/* L240: */
	}
    }

/*      NORMALIZE B1 AND B2 AND CALCULATE B3 = B1 CROSS B2 */

    b1 = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    b2 = sqrt(b[3] * b[3] + b[4] * b[4] + b[5] * b[5]);
    for (i = 1; i <= 3; ++i) {
	b[i - 1] /= b1;
/* L250: */
	b[i + 2] /= b2;
    }

/*      CHECK FOR LEFT HANDED ROTATION */

    b13 = b[1] * b[5] - b[2] * b[4];
    b23 = b[2] * b[3] - b[0] * b[5];
    b33 = b[0] * b[4] - b[1] * b[3];

    s = b13 * b[6] + b23 * b[7] + b33 * b[8];
    if (s < 0.) {
	isig = -1;
    }
    b[6] = b13;
    b[7] = b23;
    b[8] = b33;
/* %    WRITE(6,999) B */

/*     CALCULATE ROTATION MATRIX R */

    for (i = 1; i <= 3; ++i) {
	for (j = 1; j <= 3; ++j) {
/* SMJS Changed to be like Robs version */
	    t = (doublereal)0.;
	    for (k = 1; k <= 3; ++k) {
/* L260: */
		t += b[i + k * 3 - 4] * a[j + k * 3 - 4];
	    }
	    r[i + j * 3] = t;
/* L270: */
	}
    }

/*     RMS ERROR */

    for (i = 1; i <= 3; ++i) {
	if (root[i - 1] < 0.) {
	    root[i - 1] = (float)0.;
	}
	root[i - 1] = sqrt(root[i - 1]);
/* L280: */
    }

/*     CHANGE SIGN OF EVAL #3 IF LEFT HANDED */

    if (isig < 0) {
	root[2] = -root[2];
    }
    *rtsum = root[2] + root[1] + root[0];
    return 0;
} /* qkfit_ */

#undef utr
#undef ham
#undef gam
#undef fam
#undef cam
#undef bam
#undef aam
#undef usqmat
#undef b
#undef a


/*---- SUBROUTINE TO COMPUTE EIGENVALUES & EIGENVECTORS OF A REAL SYMMETRIC*/
/* ---- MATRIX, STOLEN FROM IBM SSP MANUAL (SEE P165) */
/* ---- DESCRIPTION OF PARAMETERS - */
/* ---- A - ORIGINAL MATRIX STORED COLUMNWISE AS UPPER TRIANGLE ONLY, */
/* ---- I.E. "STORAGE MODE" = 1.  EIGENVALUES ARE WRITTEN INTO DIAGONAL */
/* ---- ELEMENTS OF A  I.E.  A(1)  A(3)  A(6)  FOR A 3*3 MATRIX. */
/* ---- R - RESULTANT MATRIX OF EIGENVECTORS STORED COLUMNWISE IN SAME */
/* ---- ORDER AS EIGENVALUES. */
/* ---- N - ORDER OF MATRICES A & R. */
/* ---- MV = 0 TO COMPUTE EIGENVALUES & EIGENVECTORS. */
/* Subroutine */ int eigen_(a, r, n, mv)
doublereal *a, *r;
integer *n, *mv;
{
    /* System generated locals */
    integer i_1, i_2;
    doublereal d_1;

    /* Builtin functions */
    double sqrt();

    /* Local variables */
    static doublereal cosx, sinx, cosx2, sinx2;
    static integer i, j, l, m;
    static doublereal x, y, range, anorm, sincs, anrmx;
    static integer ia, ij, il, im, ll, lm, iq, mm, lq, mq, ind, ilq, imq, ilr,
	     imr;
    static doublereal thr;

/* ---- DOUBLE PRECISION A,R,ANORM,ANRMX,THR,X,Y,SINX,SINX2,COSX, */
/* 	1COSX2,SINCS,RANGE */
/*---- FOR DOUBLE PRECISION, SQRT IN STATEMENTS 40,68,75&78 MUST BE DSQRT,
*/
/* ---- ABS IN 62 MUST BE DABS AND 1.E-6 IN 5 MUST BE 1.D-12 . */
    /* Parameter adjustments */
    --r;
    --a;

    /* Function Body */
/* L5: */
    range = (float)1e-6;
    if (*mv - 1 != 0) {
	goto L10;
    } else {
	goto L25;
    }
L10:
    iq = -(*n);
    i_1 = *n;
    for (j = 1; j <= i_1; ++j) {
	iq += *n;
	i_2 = *n;
	for (i = 1; i <= i_2; ++i) {
	    ij = iq + i;
	    r[ij] = (float)0.;
	    if (i - j != 0) {
		goto L20;
	    } else {
		goto L15;
	    }
L15:
	    r[ij] = (float)1.;
L20:
	    ;
	}
    }
/* ---- INITIAL AND FINAL NORMS (ANORM & ANRMX) */
L25:
    anorm = (float)0.;
    i_2 = *n;
    for (i = 1; i <= i_2; ++i) {
	i_1 = *n;
	for (j = i; j <= i_1; ++j) {
	    if (i - j != 0) {
		goto L30;
	    } else {
		goto L35;
	    }
L30:
	    ia = i + (j * j - j) / 2;
/* Computing 2nd power */
	    d_1 = a[ia];
	    anorm += d_1 * d_1;
L35:
	    ;
	}
    }
    if (anorm <= 0.) {
	goto L165;
    } else {
	goto L40;
    }
L40:
    anorm = sqrt(anorm * (float)2.);
    anrmx = anorm * range / *n;
/* ---- INITIALIZE INDICATORS AND COMPUTE THRESHOLD */
    ind = 0;
    thr = anorm;
L45:
    thr /= *n;
L50:
    l = 1;
L55:
    m = l + 1;
/* ---- COMPUTE SIN & COS */
L60:
    mq = (m * m - m) / 2;
    lq = (l * l - l) / 2;
    lm = l + mq;
/* L62: */
    if ((d_1 = a[lm], abs(d_1)) - thr >= 0.) {
	goto L65;
    } else {
	goto L130;
    }
L65:
    ind = 1;
    ll = l + lq;
    mm = m + mq;
    x = (a[ll] - a[mm]) * (float).5;
/* L68: */
/* Computing 2nd power */
    d_1 = a[lm];
    y = -a[lm] / sqrt(d_1 * d_1 + x * x);
    if (x >= 0.) {
	goto L75;
    } else {
	goto L70;
    }
L70:
    y = -y;
L75:
    sinx = y / sqrt((sqrt((float)1. - y * y) + (float)1.) * (float)2.);
/* Computing 2nd power */
    d_1 = sinx;
    sinx2 = d_1 * d_1;
/* L78: */
    cosx = sqrt((float)1. - sinx2);
/* Computing 2nd power */
    d_1 = cosx;
    cosx2 = d_1 * d_1;
    sincs = sinx * cosx;
/* ---- ROTATE L & M COLUMNS */
    ilq = *n * (l - 1);
    imq = *n * (m - 1);
    i_1 = *n;
    for (i = 1; i <= i_1; ++i) {
	iq = (i * i - i) / 2;
	if (i - l != 0) {
	    goto L80;
	} else {
	    goto L115;
	}
L80:
	if ((i_2 = i - m) < 0) {
	    goto L85;
	} else if (i_2 == 0) {
	    goto L115;
	} else {
	    goto L90;
	}
L85:
	im = i + mq;
	goto L95;
L90:
	im = m + iq;
L95:
	if (i - l >= 0) {
	    goto L105;
	} else {
	    goto L100;
	}
L100:
	il = i + lq;
	goto L110;
L105:
	il = l + iq;
L110:
	x = a[il] * cosx - a[im] * sinx;
	a[im] = a[il] * sinx + a[im] * cosx;
	a[il] = x;
L115:
	if (*mv - 1 != 0) {
	    goto L120;
	} else {
	    goto L125;
	}
L120:
	ilr = ilq + i;
	imr = imq + i;
	x = r[ilr] * cosx - r[imr] * sinx;
	r[imr] = r[ilr] * sinx + r[imr] * cosx;
	r[ilr] = x;
L125:
	;
    }
    x = a[lm] * (float)2. * sincs;
    y = a[ll] * cosx2 + a[mm] * sinx2 - x;
    x = a[ll] * sinx2 + a[mm] * cosx2 + x;
    a[lm] = (a[ll] - a[mm]) * sincs + a[lm] * (cosx2 - sinx2);
    a[ll] = y;
    a[mm] = x;
/* ---- TESTS FOR COMPLETION */
/* ---- TEST FOR M = LAST COLUMN */
L130:
    if (m - *n != 0) {
	goto L135;
    } else {
	goto L140;
    }
L135:
    ++m;
    goto L60;
/* ---- TEST FOR L =PENULTIMATE COLUMN */
L140:
    if (l - (*n - 1) != 0) {
	goto L145;
    } else {
	goto L150;
    }
L145:
    ++l;
    goto L55;
L150:
    if (ind - 1 != 0) {
	goto L160;
    } else {
	goto L155;
    }
L155:
    ind = 0;
    goto L50;
/* ---- COMPARE THRESHOLD WITH FINAL NORM */
L160:
    if (thr - anrmx <= 0.) {
	goto L165;
    } else {
	goto L45;
    }
L165:
    return 0;
/*---- SORT EIGENVALUES AND EIGENVECTORS IN DESCENDING ORDER OF EIGENVALUE
S*/
} /* eigen_ */

/* Subroutine */ int esort_(a, r, n, mv)
doublereal *a, *r;
integer *n, *mv;
{
    /* System generated locals */
    integer i_1, i_2, i_3;

    /* Local variables */
    static integer i, j, k;
    static doublereal x;
    static integer ll, iq, jq, mm, ilr, imr;

    /* Parameter adjustments */
    --r;
    --a;

    /* Function Body */
    iq = -(*n);
    i_1 = *n;
    for (i = 1; i <= i_1; ++i) {
	iq += *n;
	ll = i + (i * i - i) / 2;
	jq = *n * (i - 2);
	i_2 = *n;
	for (j = i; j <= i_2; ++j) {
	    jq += *n;
	    mm = j + (j * j - j) / 2;
	    if (a[ll] - a[mm] >= 0.) {
		goto L185;
	    } else {
		goto L170;
	    }
L170:
	    x = a[ll];
	    a[ll] = a[mm];
	    a[mm] = x;
	    if (*mv - 1 != 0) {
		goto L175;
	    } else {
		goto L185;
	    }
L175:
	    i_3 = *n;
	    for (k = 1; k <= i_3; ++k) {
		ilr = iq + k;
		imr = jq + k;
		x = r[ilr];
		r[ilr] = r[imr];
/* L180: */
		r[imr] = x;
	    }
L185:
	    ;
	}
    }
    return 0;
} /* esort_ */

