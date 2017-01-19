/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/* $Id: pub3dfft.c,v 1.2 2005/07/20 15:37:39 johns Exp $ */

#include <math.h>
#include "pub3dfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Subroutine */ int passb(int *nac, int *ido, int *ip, int *
	l1, int *idl1, float *cc, float *c1, float *c2, 
	float *ch, float *ch2, float *wa)
{
    /* System generated locals */
    int ch_dim1, ch_dim2, ch_offset, cc_dim1, cc_dim2, cc_offset, c1_dim1,
	     c1_dim2, c1_offset, c2_dim1, c2_offset, ch2_dim1, ch2_offset, 
	    i_1, i_2, i_3;

    /* Local variables */
    static int idij, idlj, idot, ipph, i, j, k, l, jc, lc, ik, nt, idj, 
	    idl, inc, idp;
    static float wai, war;
    static int ipp2;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_dim2 = *ip;
    cc_offset = cc_dim1 * (cc_dim2 + 1) + 1;
    cc -= cc_offset;
    c1_dim1 = *ido;
    c1_dim2 = *l1;
    c1_offset = c1_dim1 * (c1_dim2 + 1) + 1;
    c1 -= c1_offset;
    c2_dim1 = *idl1;
    c2_offset = c2_dim1 + 1;
    c2 -= c2_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    ch2_dim1 = *idl1;
    ch2_offset = ch2_dim1 + 1;
    ch2 -= ch2_offset;
    --wa;

    /* Function Body */
    idot = *ido / 2;
    nt = *ip * *idl1;
    ipp2 = *ip + 2;
    ipph = (*ip + 1) / 2;
    idp = *ip * *ido;

    if (*ido < *l1) {
	goto L106;
    }
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	jc = ipp2 - j;
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    i_3 = *ido;
	    for (i = 1; i <= i_3; ++i) {
		ch[i + (k + j * ch_dim2) * ch_dim1] = cc[i + (j + k * cc_dim2)
			 * cc_dim1] + cc[i + (jc + k * cc_dim2) * cc_dim1];
		ch[i + (k + jc * ch_dim2) * ch_dim1] = cc[i + (j + k * 
			cc_dim2) * cc_dim1] - cc[i + (jc + k * cc_dim2) * 
			cc_dim1];
/* L101: */
	    }
/* L102: */
	}
/* L103: */
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 1; i <= i_2; ++i) {
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * cc_dim2 + 1) * 
		    cc_dim1];
/* L104: */
	}
/* L105: */
    }
    goto L112;
L106:
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	jc = ipp2 - j;
	i_2 = *ido;
	for (i = 1; i <= i_2; ++i) {
	    i_3 = *l1;
	    for (k = 1; k <= i_3; ++k) {
		ch[i + (k + j * ch_dim2) * ch_dim1] = cc[i + (j + k * cc_dim2)
			 * cc_dim1] + cc[i + (jc + k * cc_dim2) * cc_dim1];
		ch[i + (k + jc * ch_dim2) * ch_dim1] = cc[i + (j + k * 
			cc_dim2) * cc_dim1] - cc[i + (jc + k * cc_dim2) * 
			cc_dim1];
/* L107: */
	    }
/* L108: */
	}
/* L109: */
    }
    i_1 = *ido;
    for (i = 1; i <= i_1; ++i) {
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * cc_dim2 + 1) * 
		    cc_dim1];
/* L110: */
	}
/* L111: */
    }
L112:
    idl = 2 - *ido;
    inc = 0;
    i_1 = ipph;
    for (l = 2; l <= i_1; ++l) {
	lc = ipp2 - l;
	idl += *ido;
	i_2 = *idl1;
	for (ik = 1; ik <= i_2; ++ik) {
	    c2[ik + l * c2_dim1] = ch2[ik + ch2_dim1] + wa[idl - 1] * ch2[ik 
		    + (ch2_dim1 << 1)];
	    c2[ik + lc * c2_dim1] = wa[idl] * ch2[ik + *ip * ch2_dim1];
/* L113: */
	}
	idlj = idl;
	inc += *ido;
	i_2 = ipph;
	for (j = 3; j <= i_2; ++j) {
	    jc = ipp2 - j;
	    idlj += inc;
	    if (idlj > idp) {
		idlj -= idp;
	    }
	    war = wa[idlj - 1];
	    wai = wa[idlj];
	    i_3 = *idl1;
	    for (ik = 1; ik <= i_3; ++ik) {
		c2[ik + l * c2_dim1] += war * ch2[ik + j * ch2_dim1];
		c2[ik + lc * c2_dim1] += wai * ch2[ik + jc * ch2_dim1];
/* L114: */
	    }
/* L115: */
	}
/* L116: */
    }
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	i_2 = *idl1;
	for (ik = 1; ik <= i_2; ++ik) {
	    ch2[ik + ch2_dim1] += ch2[ik + j * ch2_dim1];
/* L117: */
	}
/* L118: */
    }
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	jc = ipp2 - j;
	i_2 = *idl1;
	for (ik = 2; ik <= i_2; ik += 2) {
	    ch2[ik - 1 + j * ch2_dim1] = c2[ik - 1 + j * c2_dim1] - c2[ik + 
		    jc * c2_dim1];
	    ch2[ik - 1 + jc * ch2_dim1] = c2[ik - 1 + j * c2_dim1] + c2[ik + 
		    jc * c2_dim1];
	    ch2[ik + j * ch2_dim1] = c2[ik + j * c2_dim1] + c2[ik - 1 + jc * 
		    c2_dim1];
	    ch2[ik + jc * ch2_dim1] = c2[ik + j * c2_dim1] - c2[ik - 1 + jc * 
		    c2_dim1];
/* L119: */
	}
/* L120: */
    }
    *nac = 1;
    if (*ido == 2) {
	return 0;
    }
    *nac = 0;
    i_1 = *idl1;
    for (ik = 1; ik <= i_1; ++ik) {
	c2[ik + c2_dim1] = ch2[ik + ch2_dim1];
/* L121: */
    }
    i_1 = *ip;
    for (j = 2; j <= i_1; ++j) {
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    c1[(k + j * c1_dim2) * c1_dim1 + 1] = ch[(k + j * ch_dim2) * 
		    ch_dim1 + 1];
	    c1[(k + j * c1_dim2) * c1_dim1 + 2] = ch[(k + j * ch_dim2) * 
		    ch_dim1 + 2];
/* L122: */
	}
/* L123: */
    }
    if (idot > *l1) {
	goto L127;
    }
    idij = 0;
    i_1 = *ip;
    for (j = 2; j <= i_1; ++j) {
	idij += 2;
	i_2 = *ido;
	for (i = 4; i <= i_2; i += 2) {
	    idij += 2;
	    i_3 = *l1;
	    for (k = 1; k <= i_3; ++k) {
		c1[i - 1 + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i 
			- 1 + (k + j * ch_dim2) * ch_dim1] - wa[idij] * ch[i 
			+ (k + j * ch_dim2) * ch_dim1];
		c1[i + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i + (
			k + j * ch_dim2) * ch_dim1] + wa[idij] * ch[i - 1 + (
			k + j * ch_dim2) * ch_dim1];
/* L124: */
	    }
/* L125: */
	}
/* L126: */
    }
    return 0;
L127:
    idj = 2 - *ido;
    i_1 = *ip;
    for (j = 2; j <= i_1; ++j) {
	idj += *ido;
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    idij = idj;
	    i_3 = *ido;
	    for (i = 4; i <= i_3; i += 2) {
		idij += 2;
		c1[i - 1 + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i 
			- 1 + (k + j * ch_dim2) * ch_dim1] - wa[idij] * ch[i 
			+ (k + j * ch_dim2) * ch_dim1];
		c1[i + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i + (
			k + j * ch_dim2) * ch_dim1] + wa[idij] * ch[i - 1 + (
			k + j * ch_dim2) * ch_dim1];
/* L128: */
	    }
/* L129: */
	}
/* L130: */
    }
    return 0;
} /* passb_ */


/* Subroutine */ int passb2(int *ido, int *l1, float *cc, 
	float *ch, float *wa1)
{
    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ti2, tr2;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = cc_dim1 * 3 + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;

    /* Function Body */
    if (*ido > 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[((k << 1) + 1) * cc_dim1 + 1] + 
		cc[((k << 1) + 2) * cc_dim1 + 1];
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cc[((k << 1) + 1) * cc_dim1 
		+ 1] - cc[((k << 1) + 2) * cc_dim1 + 1];
	ch[(k + ch_dim2) * ch_dim1 + 2] = cc[((k << 1) + 1) * cc_dim1 + 2] + 
		cc[((k << 1) + 2) * cc_dim1 + 2];
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = cc[((k << 1) + 1) * cc_dim1 
		+ 2] - cc[((k << 1) + 2) * cc_dim1 + 2];
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = cc[i - 1 + ((k << 1) + 1) * 
		    cc_dim1] + cc[i - 1 + ((k << 1) + 2) * cc_dim1];
	    tr2 = cc[i - 1 + ((k << 1) + 1) * cc_dim1] - cc[i - 1 + ((k << 1) 
		    + 2) * cc_dim1];
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + ((k << 1) + 1) * cc_dim1]
		     + cc[i + ((k << 1) + 2) * cc_dim1];
	    ti2 = cc[i + ((k << 1) + 1) * cc_dim1] - cc[i + ((k << 1) + 2) * 
		    cc_dim1];
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * ti2 + wa1[i]
		     * tr2;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * tr2 - 
		    wa1[i] * ti2;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passb2_ */


/* Subroutine */ int passb3(int *ido, int *l1, float *cc, 
	float *ch, float *wa1, float *wa2)
{
    /* Initialized data */

    static float taur = -.5;
    static float taui = .866025403784439;

    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = (cc_dim1 << 2) + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;
    --wa2;

    /* Function Body */
    if (*ido != 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	tr2 = cc[(k * 3 + 2) * cc_dim1 + 1] + cc[(k * 3 + 3) * cc_dim1 + 1];
	cr2 = cc[(k * 3 + 1) * cc_dim1 + 1] + taur * tr2;
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[(k * 3 + 1) * cc_dim1 + 1] + tr2;

	ti2 = cc[(k * 3 + 2) * cc_dim1 + 2] + cc[(k * 3 + 3) * cc_dim1 + 2];
	ci2 = cc[(k * 3 + 1) * cc_dim1 + 2] + taur * ti2;
	ch[(k + ch_dim2) * ch_dim1 + 2] = cc[(k * 3 + 1) * cc_dim1 + 2] + ti2;

	cr3 = taui * (cc[(k * 3 + 2) * cc_dim1 + 1] - cc[(k * 3 + 3) * 
		cc_dim1 + 1]);
	ci3 = taui * (cc[(k * 3 + 2) * cc_dim1 + 2] - cc[(k * 3 + 3) * 
		cc_dim1 + 2]);
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cr2 - ci3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = cr2 + ci3;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = ci2 + cr3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 2] = ci2 - cr3;
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    tr2 = cc[i - 1 + (k * 3 + 2) * cc_dim1] + cc[i - 1 + (k * 3 + 3) *
		     cc_dim1];
	    cr2 = cc[i - 1 + (k * 3 + 1) * cc_dim1] + taur * tr2;
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = cc[i - 1 + (k * 3 + 1) * 
		    cc_dim1] + tr2;
	    ti2 = cc[i + (k * 3 + 2) * cc_dim1] + cc[i + (k * 3 + 3) * 
		    cc_dim1];
	    ci2 = cc[i + (k * 3 + 1) * cc_dim1] + taur * ti2;
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * 3 + 1) * cc_dim1] + 
		    ti2;
	    cr3 = taui * (cc[i - 1 + (k * 3 + 2) * cc_dim1] - cc[i - 1 + (k * 
		    3 + 3) * cc_dim1]);
	    ci3 = taui * (cc[i + (k * 3 + 2) * cc_dim1] - cc[i + (k * 3 + 3) *
		     cc_dim1]);
	    dr2 = cr2 - ci3;
	    dr3 = cr2 + ci3;
	    di2 = ci2 + cr3;
	    di3 = ci2 - cr3;
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * di2 + wa1[i]
		     * dr2;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * dr2 - 
		    wa1[i] * di2;
	    ch[i + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * di3 + wa2[i] * 
		    dr3;
	    ch[i - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * dr3 - wa2[
		    i] * di3;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passb3_ */

/* Subroutine */ int passb4(int *ido, int *l1, float *cc, 
	float *ch, float *wa1, float *wa2, float *wa3)
{
    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, 
	    tr2, tr3, tr4;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = cc_dim1 * 5 + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;
    --wa2;
    --wa3;

    /* Function Body */
    if (*ido != 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	ti1 = cc[((k << 2) + 1) * cc_dim1 + 2] - cc[((k << 2) + 3) * cc_dim1 
		+ 2];
	ti2 = cc[((k << 2) + 1) * cc_dim1 + 2] + cc[((k << 2) + 3) * cc_dim1 
		+ 2];
	tr4 = cc[((k << 2) + 4) * cc_dim1 + 2] - cc[((k << 2) + 2) * cc_dim1 
		+ 2];
	ti3 = cc[((k << 2) + 2) * cc_dim1 + 2] + cc[((k << 2) + 4) * cc_dim1 
		+ 2];
	tr1 = cc[((k << 2) + 1) * cc_dim1 + 1] - cc[((k << 2) + 3) * cc_dim1 
		+ 1];
	tr2 = cc[((k << 2) + 1) * cc_dim1 + 1] + cc[((k << 2) + 3) * cc_dim1 
		+ 1];
	ti4 = cc[((k << 2) + 2) * cc_dim1 + 1] - cc[((k << 2) + 4) * cc_dim1 
		+ 1];
	tr3 = cc[((k << 2) + 2) * cc_dim1 + 1] + cc[((k << 2) + 4) * cc_dim1 
		+ 1];
	ch[(k + ch_dim2) * ch_dim1 + 1] = tr2 + tr3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = tr2 - tr3;
	ch[(k + ch_dim2) * ch_dim1 + 2] = ti2 + ti3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 2] = ti2 - ti3;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = tr1 + tr4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 1] = tr1 - tr4;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = ti1 + ti4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 2] = ti1 - ti4;
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    ti1 = cc[i + ((k << 2) + 1) * cc_dim1] - cc[i + ((k << 2) + 3) * 
		    cc_dim1];
	    ti2 = cc[i + ((k << 2) + 1) * cc_dim1] + cc[i + ((k << 2) + 3) * 
		    cc_dim1];
	    ti3 = cc[i + ((k << 2) + 2) * cc_dim1] + cc[i + ((k << 2) + 4) * 
		    cc_dim1];
	    tr4 = cc[i + ((k << 2) + 4) * cc_dim1] - cc[i + ((k << 2) + 2) * 
		    cc_dim1];
	    tr1 = cc[i - 1 + ((k << 2) + 1) * cc_dim1] - cc[i - 1 + ((k << 2) 
		    + 3) * cc_dim1];
	    tr2 = cc[i - 1 + ((k << 2) + 1) * cc_dim1] + cc[i - 1 + ((k << 2) 
		    + 3) * cc_dim1];
	    ti4 = cc[i - 1 + ((k << 2) + 2) * cc_dim1] - cc[i - 1 + ((k << 2) 
		    + 4) * cc_dim1];
	    tr3 = cc[i - 1 + ((k << 2) + 2) * cc_dim1] + cc[i - 1 + ((k << 2) 
		    + 4) * cc_dim1];
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = tr2 + tr3;
	    cr3 = tr2 - tr3;
	    ch[i + (k + ch_dim2) * ch_dim1] = ti2 + ti3;
	    ci3 = ti2 - ti3;
	    cr2 = tr1 + tr4;
	    cr4 = tr1 - tr4;
	    ci2 = ti1 + ti4;
	    ci4 = ti1 - ti4;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * cr2 - 
		    wa1[i] * ci2;
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * ci2 + wa1[i]
		     * cr2;
	    ch[i - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * cr3 - wa2[
		    i] * ci3;
	    ch[i + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * ci3 + wa2[i] * 
		    cr3;
	    ch[i - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * cr4 - 
		    wa3[i] * ci4;
	    ch[i + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * ci4 + wa3[i]
		     * cr4;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passb4_ */

/* Subroutine */ int passb5(int *ido, int *l1, float *cc, 
	float *ch, float *wa1, float *wa2, float *wa3, 
	float *wa4)
{
    /* Initialized data */

    static float tr11 = .309016994374947;
    static float ti11 = .951056516295154;
    static float tr12 = -.809016994374947;
    static float ti12 = .587785252292473;

    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, 
	    cr4, ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = cc_dim1 * 6 + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;
    --wa2;
    --wa3;
    --wa4;

    /* Function Body */
    if (*ido != 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	ti5 = cc[(k * 5 + 2) * cc_dim1 + 2] - cc[(k * 5 + 5) * cc_dim1 + 2];
	ti2 = cc[(k * 5 + 2) * cc_dim1 + 2] + cc[(k * 5 + 5) * cc_dim1 + 2];
	ti4 = cc[(k * 5 + 3) * cc_dim1 + 2] - cc[(k * 5 + 4) * cc_dim1 + 2];
	ti3 = cc[(k * 5 + 3) * cc_dim1 + 2] + cc[(k * 5 + 4) * cc_dim1 + 2];
	tr5 = cc[(k * 5 + 2) * cc_dim1 + 1] - cc[(k * 5 + 5) * cc_dim1 + 1];
	tr2 = cc[(k * 5 + 2) * cc_dim1 + 1] + cc[(k * 5 + 5) * cc_dim1 + 1];
	tr4 = cc[(k * 5 + 3) * cc_dim1 + 1] - cc[(k * 5 + 4) * cc_dim1 + 1];
	tr3 = cc[(k * 5 + 3) * cc_dim1 + 1] + cc[(k * 5 + 4) * cc_dim1 + 1];
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[(k * 5 + 1) * cc_dim1 + 1] + tr2 
		+ tr3;
	ch[(k + ch_dim2) * ch_dim1 + 2] = cc[(k * 5 + 1) * cc_dim1 + 2] + ti2 
		+ ti3;
	cr2 = cc[(k * 5 + 1) * cc_dim1 + 1] + tr11 * tr2 + tr12 * tr3;
	ci2 = cc[(k * 5 + 1) * cc_dim1 + 2] + tr11 * ti2 + tr12 * ti3;
	cr3 = cc[(k * 5 + 1) * cc_dim1 + 1] + tr12 * tr2 + tr11 * tr3;
	ci3 = cc[(k * 5 + 1) * cc_dim1 + 2] + tr12 * ti2 + tr11 * ti3;
	cr5 = ti11 * tr5 + ti12 * tr4;
	ci5 = ti11 * ti5 + ti12 * ti4;
	cr4 = ti12 * tr5 - ti11 * tr4;
	ci4 = ti12 * ti5 - ti11 * ti4;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cr2 - ci5;
	ch[(k + ch_dim2 * 5) * ch_dim1 + 1] = cr2 + ci5;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = ci2 + cr5;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 2] = ci3 + cr4;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = cr3 - ci4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 1] = cr3 + ci4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 2] = ci3 - cr4;
	ch[(k + ch_dim2 * 5) * ch_dim1 + 2] = ci2 - cr5;
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    ti5 = cc[i + (k * 5 + 2) * cc_dim1] - cc[i + (k * 5 + 5) * 
		    cc_dim1];
	    ti2 = cc[i + (k * 5 + 2) * cc_dim1] + cc[i + (k * 5 + 5) * 
		    cc_dim1];
	    ti4 = cc[i + (k * 5 + 3) * cc_dim1] - cc[i + (k * 5 + 4) * 
		    cc_dim1];
	    ti3 = cc[i + (k * 5 + 3) * cc_dim1] + cc[i + (k * 5 + 4) * 
		    cc_dim1];
	    tr5 = cc[i - 1 + (k * 5 + 2) * cc_dim1] - cc[i - 1 + (k * 5 + 5) *
		     cc_dim1];
	    tr2 = cc[i - 1 + (k * 5 + 2) * cc_dim1] + cc[i - 1 + (k * 5 + 5) *
		     cc_dim1];
	    tr4 = cc[i - 1 + (k * 5 + 3) * cc_dim1] - cc[i - 1 + (k * 5 + 4) *
		     cc_dim1];
	    tr3 = cc[i - 1 + (k * 5 + 3) * cc_dim1] + cc[i - 1 + (k * 5 + 4) *
		     cc_dim1];
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = cc[i - 1 + (k * 5 + 1) * 
		    cc_dim1] + tr2 + tr3;
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * 5 + 1) * cc_dim1] + 
		    ti2 + ti3;
	    cr2 = cc[i - 1 + (k * 5 + 1) * cc_dim1] + tr11 * tr2 + tr12 * tr3;

	    ci2 = cc[i + (k * 5 + 1) * cc_dim1] + tr11 * ti2 + tr12 * ti3;
	    cr3 = cc[i - 1 + (k * 5 + 1) * cc_dim1] + tr12 * tr2 + tr11 * tr3;

	    ci3 = cc[i + (k * 5 + 1) * cc_dim1] + tr12 * ti2 + tr11 * ti3;
	    cr5 = ti11 * tr5 + ti12 * tr4;
	    ci5 = ti11 * ti5 + ti12 * ti4;
	    cr4 = ti12 * tr5 - ti11 * tr4;
	    ci4 = ti12 * ti5 - ti11 * ti4;
	    dr3 = cr3 - ci4;
	    dr4 = cr3 + ci4;
	    di3 = ci3 + cr4;
	    di4 = ci3 - cr4;
	    dr5 = cr2 + ci5;
	    dr2 = cr2 - ci5;
	    di5 = ci2 - cr5;
	    di2 = ci2 + cr5;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * dr2 - 
		    wa1[i] * di2;
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * di2 + wa1[i]
		     * dr2;
	    ch[i - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * dr3 - wa2[
		    i] * di3;
	    ch[i + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * di3 + wa2[i] * 
		    dr3;
	    ch[i - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * dr4 - 
		    wa3[i] * di4;
	    ch[i + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * di4 + wa3[i]
		     * dr4;
	    ch[i - 1 + (k + ch_dim2 * 5) * ch_dim1] = wa4[i - 1] * dr5 - wa4[
		    i] * di5;
	    ch[i + (k + ch_dim2 * 5) * ch_dim1] = wa4[i - 1] * di5 + wa4[i] * 
		    dr5;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passb5_ */

/* Subroutine */ int passf(int *nac, int *ido, int *ip, int *
	l1, int *idl1, float *cc, float *c1, float *c2, 
	float *ch, float *ch2, float *wa)
{
    /* System generated locals */
    int ch_dim1, ch_dim2, ch_offset, cc_dim1, cc_dim2, cc_offset, c1_dim1,
	     c1_dim2, c1_offset, c2_dim1, c2_offset, ch2_dim1, ch2_offset, 
	    i_1, i_2, i_3;

    /* Local variables */
    static int idij, idlj, idot, ipph, i, j, k, l, jc, lc, ik, nt, idj, 
	    idl, inc, idp;
    static float wai, war;
    static int ipp2;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_dim2 = *ip;
    cc_offset = cc_dim1 * (cc_dim2 + 1) + 1;
    cc -= cc_offset;
    c1_dim1 = *ido;
    c1_dim2 = *l1;
    c1_offset = c1_dim1 * (c1_dim2 + 1) + 1;
    c1 -= c1_offset;
    c2_dim1 = *idl1;
    c2_offset = c2_dim1 + 1;
    c2 -= c2_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    ch2_dim1 = *idl1;
    ch2_offset = ch2_dim1 + 1;
    ch2 -= ch2_offset;
    --wa;

    /* Function Body */
    idot = *ido / 2;
    nt = *ip * *idl1;
    ipp2 = *ip + 2;
    ipph = (*ip + 1) / 2;
    idp = *ip * *ido;

    if (*ido < *l1) {
	goto L106;
    }
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	jc = ipp2 - j;
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    i_3 = *ido;
	    for (i = 1; i <= i_3; ++i) {
		ch[i + (k + j * ch_dim2) * ch_dim1] = cc[i + (j + k * cc_dim2)
			 * cc_dim1] + cc[i + (jc + k * cc_dim2) * cc_dim1];
		ch[i + (k + jc * ch_dim2) * ch_dim1] = cc[i + (j + k * 
			cc_dim2) * cc_dim1] - cc[i + (jc + k * cc_dim2) * 
			cc_dim1];
/* L101: */
	    }
/* L102: */
	}
/* L103: */
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 1; i <= i_2; ++i) {
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * cc_dim2 + 1) * 
		    cc_dim1];
/* L104: */
	}
/* L105: */
    }
    goto L112;
L106:
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	jc = ipp2 - j;
	i_2 = *ido;
	for (i = 1; i <= i_2; ++i) {
	    i_3 = *l1;
	    for (k = 1; k <= i_3; ++k) {
		ch[i + (k + j * ch_dim2) * ch_dim1] = cc[i + (j + k * cc_dim2)
			 * cc_dim1] + cc[i + (jc + k * cc_dim2) * cc_dim1];
		ch[i + (k + jc * ch_dim2) * ch_dim1] = cc[i + (j + k * 
			cc_dim2) * cc_dim1] - cc[i + (jc + k * cc_dim2) * 
			cc_dim1];
/* L107: */
	    }
/* L108: */
	}
/* L109: */
    }
    i_1 = *ido;
    for (i = 1; i <= i_1; ++i) {
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * cc_dim2 + 1) * 
		    cc_dim1];
/* L110: */
	}
/* L111: */
    }
L112:
    idl = 2 - *ido;
    inc = 0;
    i_1 = ipph;
    for (l = 2; l <= i_1; ++l) {
	lc = ipp2 - l;
	idl += *ido;
	i_2 = *idl1;
	for (ik = 1; ik <= i_2; ++ik) {
	    c2[ik + l * c2_dim1] = ch2[ik + ch2_dim1] + wa[idl - 1] * ch2[ik 
		    + (ch2_dim1 << 1)];
	    c2[ik + lc * c2_dim1] = -wa[idl] * ch2[ik + *ip * ch2_dim1];
/* L113: */
	}
	idlj = idl;
	inc += *ido;
	i_2 = ipph;
	for (j = 3; j <= i_2; ++j) {
	    jc = ipp2 - j;
	    idlj += inc;
	    if (idlj > idp) {
		idlj -= idp;
	    }
	    war = wa[idlj - 1];
	    wai = wa[idlj];
	    i_3 = *idl1;
	    for (ik = 1; ik <= i_3; ++ik) {
		c2[ik + l * c2_dim1] += war * ch2[ik + j * ch2_dim1];
		c2[ik + lc * c2_dim1] -= wai * ch2[ik + jc * ch2_dim1];
/* L114: */
	    }
/* L115: */
	}
/* L116: */
    }
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	i_2 = *idl1;
	for (ik = 1; ik <= i_2; ++ik) {
	    ch2[ik + ch2_dim1] += ch2[ik + j * ch2_dim1];
/* L117: */
	}
/* L118: */
    }
    i_1 = ipph;
    for (j = 2; j <= i_1; ++j) {
	jc = ipp2 - j;
	i_2 = *idl1;
	for (ik = 2; ik <= i_2; ik += 2) {
	    ch2[ik - 1 + j * ch2_dim1] = c2[ik - 1 + j * c2_dim1] - c2[ik + 
		    jc * c2_dim1];
	    ch2[ik - 1 + jc * ch2_dim1] = c2[ik - 1 + j * c2_dim1] + c2[ik + 
		    jc * c2_dim1];
	    ch2[ik + j * ch2_dim1] = c2[ik + j * c2_dim1] + c2[ik - 1 + jc * 
		    c2_dim1];
	    ch2[ik + jc * ch2_dim1] = c2[ik + j * c2_dim1] - c2[ik - 1 + jc * 
		    c2_dim1];
/* L119: */
	}
/* L120: */
    }
    *nac = 1;
    if (*ido == 2) {
	return 0;
    }
    *nac = 0;
    i_1 = *idl1;
    for (ik = 1; ik <= i_1; ++ik) {
	c2[ik + c2_dim1] = ch2[ik + ch2_dim1];
/* L121: */
    }
    i_1 = *ip;
    for (j = 2; j <= i_1; ++j) {
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    c1[(k + j * c1_dim2) * c1_dim1 + 1] = ch[(k + j * ch_dim2) * 
		    ch_dim1 + 1];
	    c1[(k + j * c1_dim2) * c1_dim1 + 2] = ch[(k + j * ch_dim2) * 
		    ch_dim1 + 2];
/* L122: */
	}
/* L123: */
    }
    if (idot > *l1) {
	goto L127;
    }
    idij = 0;
    i_1 = *ip;
    for (j = 2; j <= i_1; ++j) {
	idij += 2;
	i_2 = *ido;
	for (i = 4; i <= i_2; i += 2) {
	    idij += 2;
	    i_3 = *l1;
	    for (k = 1; k <= i_3; ++k) {
		c1[i - 1 + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i 
			- 1 + (k + j * ch_dim2) * ch_dim1] + wa[idij] * ch[i 
			+ (k + j * ch_dim2) * ch_dim1];
		c1[i + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i + (
			k + j * ch_dim2) * ch_dim1] - wa[idij] * ch[i - 1 + (
			k + j * ch_dim2) * ch_dim1];
/* L124: */
	    }
/* L125: */
	}
/* L126: */
    }
    return 0;
L127:
    idj = 2 - *ido;
    i_1 = *ip;
    for (j = 2; j <= i_1; ++j) {
	idj += *ido;
	i_2 = *l1;
	for (k = 1; k <= i_2; ++k) {
	    idij = idj;
	    i_3 = *ido;
	    for (i = 4; i <= i_3; i += 2) {
		idij += 2;
		c1[i - 1 + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i 
			- 1 + (k + j * ch_dim2) * ch_dim1] + wa[idij] * ch[i 
			+ (k + j * ch_dim2) * ch_dim1];
		c1[i + (k + j * c1_dim2) * c1_dim1] = wa[idij - 1] * ch[i + (
			k + j * ch_dim2) * ch_dim1] - wa[idij] * ch[i - 1 + (
			k + j * ch_dim2) * ch_dim1];
/* L128: */
	    }
/* L129: */
	}
/* L130: */
    }
    return 0;
} /* passf_ */

/* Subroutine */ int passf2(int *ido, int *l1, float *cc, 
	float *ch, float *wa1)
{
    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ti2, tr2;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = cc_dim1 * 3 + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;

    /* Function Body */
    if (*ido > 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[((k << 1) + 1) * cc_dim1 + 1] + 
		cc[((k << 1) + 2) * cc_dim1 + 1];
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cc[((k << 1) + 1) * cc_dim1 
		+ 1] - cc[((k << 1) + 2) * cc_dim1 + 1];
	ch[(k + ch_dim2) * ch_dim1 + 2] = cc[((k << 1) + 1) * cc_dim1 + 2] + 
		cc[((k << 1) + 2) * cc_dim1 + 2];
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = cc[((k << 1) + 1) * cc_dim1 
		+ 2] - cc[((k << 1) + 2) * cc_dim1 + 2];
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = cc[i - 1 + ((k << 1) + 1) * 
		    cc_dim1] + cc[i - 1 + ((k << 1) + 2) * cc_dim1];
	    tr2 = cc[i - 1 + ((k << 1) + 1) * cc_dim1] - cc[i - 1 + ((k << 1) 
		    + 2) * cc_dim1];
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + ((k << 1) + 1) * cc_dim1]
		     + cc[i + ((k << 1) + 2) * cc_dim1];
	    ti2 = cc[i + ((k << 1) + 1) * cc_dim1] - cc[i + ((k << 1) + 2) * 
		    cc_dim1];
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * ti2 - wa1[i]
		     * tr2;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * tr2 + 
		    wa1[i] * ti2;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passf2_ */

/* Subroutine */ int passf3(int *ido, int *l1, float *cc, 
	float *ch, float *wa1, float *wa2)
{
    /* Initialized data */

    static float taur = -.5;
    static float taui = -.866025403784439;

    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = (cc_dim1 << 2) + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;
    --wa2;

    /* Function Body */
    if (*ido != 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	tr2 = cc[(k * 3 + 2) * cc_dim1 + 1] + cc[(k * 3 + 3) * cc_dim1 + 1];
	cr2 = cc[(k * 3 + 1) * cc_dim1 + 1] + taur * tr2;
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[(k * 3 + 1) * cc_dim1 + 1] + tr2;

	ti2 = cc[(k * 3 + 2) * cc_dim1 + 2] + cc[(k * 3 + 3) * cc_dim1 + 2];
	ci2 = cc[(k * 3 + 1) * cc_dim1 + 2] + taur * ti2;
	ch[(k + ch_dim2) * ch_dim1 + 2] = cc[(k * 3 + 1) * cc_dim1 + 2] + ti2;

	cr3 = taui * (cc[(k * 3 + 2) * cc_dim1 + 1] - cc[(k * 3 + 3) * 
		cc_dim1 + 1]);
	ci3 = taui * (cc[(k * 3 + 2) * cc_dim1 + 2] - cc[(k * 3 + 3) * 
		cc_dim1 + 2]);
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cr2 - ci3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = cr2 + ci3;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = ci2 + cr3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 2] = ci2 - cr3;
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    tr2 = cc[i - 1 + (k * 3 + 2) * cc_dim1] + cc[i - 1 + (k * 3 + 3) *
		     cc_dim1];
	    cr2 = cc[i - 1 + (k * 3 + 1) * cc_dim1] + taur * tr2;
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = cc[i - 1 + (k * 3 + 1) * 
		    cc_dim1] + tr2;
	    ti2 = cc[i + (k * 3 + 2) * cc_dim1] + cc[i + (k * 3 + 3) * 
		    cc_dim1];
	    ci2 = cc[i + (k * 3 + 1) * cc_dim1] + taur * ti2;
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * 3 + 1) * cc_dim1] + 
		    ti2;
	    cr3 = taui * (cc[i - 1 + (k * 3 + 2) * cc_dim1] - cc[i - 1 + (k * 
		    3 + 3) * cc_dim1]);
	    ci3 = taui * (cc[i + (k * 3 + 2) * cc_dim1] - cc[i + (k * 3 + 3) *
		     cc_dim1]);
	    dr2 = cr2 - ci3;
	    dr3 = cr2 + ci3;
	    di2 = ci2 + cr3;
	    di3 = ci2 - cr3;
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * di2 - wa1[i]
		     * dr2;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * dr2 + 
		    wa1[i] * di2;
	    ch[i + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * di3 - wa2[i] * 
		    dr3;
	    ch[i - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * dr3 + wa2[
		    i] * di3;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passf3_ */

/* Subroutine */ int passf4(int *ido, int *l1, float *cc, 
	float *ch, float *wa1, float *wa2, float *wa3)
{
    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, 
	    tr2, tr3, tr4;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = cc_dim1 * 5 + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;
    --wa2;
    --wa3;

    /* Function Body */
    if (*ido != 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	ti1 = cc[((k << 2) + 1) * cc_dim1 + 2] - cc[((k << 2) + 3) * cc_dim1 
		+ 2];
	ti2 = cc[((k << 2) + 1) * cc_dim1 + 2] + cc[((k << 2) + 3) * cc_dim1 
		+ 2];
	tr4 = cc[((k << 2) + 2) * cc_dim1 + 2] - cc[((k << 2) + 4) * cc_dim1 
		+ 2];
	ti3 = cc[((k << 2) + 2) * cc_dim1 + 2] + cc[((k << 2) + 4) * cc_dim1 
		+ 2];
	tr1 = cc[((k << 2) + 1) * cc_dim1 + 1] - cc[((k << 2) + 3) * cc_dim1 
		+ 1];
	tr2 = cc[((k << 2) + 1) * cc_dim1 + 1] + cc[((k << 2) + 3) * cc_dim1 
		+ 1];
	ti4 = cc[((k << 2) + 4) * cc_dim1 + 1] - cc[((k << 2) + 2) * cc_dim1 
		+ 1];
	tr3 = cc[((k << 2) + 2) * cc_dim1 + 1] + cc[((k << 2) + 4) * cc_dim1 
		+ 1];
	ch[(k + ch_dim2) * ch_dim1 + 1] = tr2 + tr3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = tr2 - tr3;
	ch[(k + ch_dim2) * ch_dim1 + 2] = ti2 + ti3;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 2] = ti2 - ti3;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = tr1 + tr4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 1] = tr1 - tr4;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = ti1 + ti4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 2] = ti1 - ti4;
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    ti1 = cc[i + ((k << 2) + 1) * cc_dim1] - cc[i + ((k << 2) + 3) * 
		    cc_dim1];
	    ti2 = cc[i + ((k << 2) + 1) * cc_dim1] + cc[i + ((k << 2) + 3) * 
		    cc_dim1];
	    ti3 = cc[i + ((k << 2) + 2) * cc_dim1] + cc[i + ((k << 2) + 4) * 
		    cc_dim1];
	    tr4 = cc[i + ((k << 2) + 2) * cc_dim1] - cc[i + ((k << 2) + 4) * 
		    cc_dim1];
	    tr1 = cc[i - 1 + ((k << 2) + 1) * cc_dim1] - cc[i - 1 + ((k << 2) 
		    + 3) * cc_dim1];
	    tr2 = cc[i - 1 + ((k << 2) + 1) * cc_dim1] + cc[i - 1 + ((k << 2) 
		    + 3) * cc_dim1];
	    ti4 = cc[i - 1 + ((k << 2) + 4) * cc_dim1] - cc[i - 1 + ((k << 2) 
		    + 2) * cc_dim1];
	    tr3 = cc[i - 1 + ((k << 2) + 2) * cc_dim1] + cc[i - 1 + ((k << 2) 
		    + 4) * cc_dim1];
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = tr2 + tr3;
	    cr3 = tr2 - tr3;
	    ch[i + (k + ch_dim2) * ch_dim1] = ti2 + ti3;
	    ci3 = ti2 - ti3;
	    cr2 = tr1 + tr4;
	    cr4 = tr1 - tr4;
	    ci2 = ti1 + ti4;
	    ci4 = ti1 - ti4;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * cr2 + 
		    wa1[i] * ci2;
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * ci2 - wa1[i]
		     * cr2;
	    ch[i - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * cr3 + wa2[
		    i] * ci3;
	    ch[i + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * ci3 - wa2[i] * 
		    cr3;
	    ch[i - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * cr4 + 
		    wa3[i] * ci4;
	    ch[i + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * ci4 - wa3[i]
		     * cr4;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passf4_ */

/* Subroutine */ int passf5(int *ido, int *l1, float *cc, 
	float *ch, float *wa1, float *wa2, float *wa3, 
	float *wa4)
{
    /* Initialized data */

    static float tr11 = .309016994374947;
    static float ti11 = -.951056516295154;
    static float tr12 = -.809016994374947;
    static float ti12 = -.587785252292473;

    /* System generated locals */
    int cc_dim1, cc_offset, ch_dim1, ch_dim2, ch_offset, i_1, i_2;

    /* Local variables */
    static int i, k;
    static float ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, 
	    cr4, ti2, ti3, ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;

    /* Parameter adjustments */
    cc_dim1 = *ido;
    cc_offset = cc_dim1 * 6 + 1;
    cc -= cc_offset;
    ch_dim1 = *ido;
    ch_dim2 = *l1;
    ch_offset = ch_dim1 * (ch_dim2 + 1) + 1;
    ch -= ch_offset;
    --wa1;
    --wa2;
    --wa3;
    --wa4;

    /* Function Body */
    if (*ido != 2) {
	goto L102;
    }
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	ti5 = cc[(k * 5 + 2) * cc_dim1 + 2] - cc[(k * 5 + 5) * cc_dim1 + 2];
	ti2 = cc[(k * 5 + 2) * cc_dim1 + 2] + cc[(k * 5 + 5) * cc_dim1 + 2];
	ti4 = cc[(k * 5 + 3) * cc_dim1 + 2] - cc[(k * 5 + 4) * cc_dim1 + 2];
	ti3 = cc[(k * 5 + 3) * cc_dim1 + 2] + cc[(k * 5 + 4) * cc_dim1 + 2];
	tr5 = cc[(k * 5 + 2) * cc_dim1 + 1] - cc[(k * 5 + 5) * cc_dim1 + 1];
	tr2 = cc[(k * 5 + 2) * cc_dim1 + 1] + cc[(k * 5 + 5) * cc_dim1 + 1];
	tr4 = cc[(k * 5 + 3) * cc_dim1 + 1] - cc[(k * 5 + 4) * cc_dim1 + 1];
	tr3 = cc[(k * 5 + 3) * cc_dim1 + 1] + cc[(k * 5 + 4) * cc_dim1 + 1];
	ch[(k + ch_dim2) * ch_dim1 + 1] = cc[(k * 5 + 1) * cc_dim1 + 1] + tr2 
		+ tr3;
	ch[(k + ch_dim2) * ch_dim1 + 2] = cc[(k * 5 + 1) * cc_dim1 + 2] + ti2 
		+ ti3;
	cr2 = cc[(k * 5 + 1) * cc_dim1 + 1] + tr11 * tr2 + tr12 * tr3;
	ci2 = cc[(k * 5 + 1) * cc_dim1 + 2] + tr11 * ti2 + tr12 * ti3;
	cr3 = cc[(k * 5 + 1) * cc_dim1 + 1] + tr12 * tr2 + tr11 * tr3;
	ci3 = cc[(k * 5 + 1) * cc_dim1 + 2] + tr12 * ti2 + tr11 * ti3;
	cr5 = ti11 * tr5 + ti12 * tr4;
	ci5 = ti11 * ti5 + ti12 * ti4;
	cr4 = ti12 * tr5 - ti11 * tr4;
	ci4 = ti12 * ti5 - ti11 * ti4;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 1] = cr2 - ci5;
	ch[(k + ch_dim2 * 5) * ch_dim1 + 1] = cr2 + ci5;
	ch[(k + (ch_dim2 << 1)) * ch_dim1 + 2] = ci2 + cr5;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 2] = ci3 + cr4;
	ch[(k + ch_dim2 * 3) * ch_dim1 + 1] = cr3 - ci4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 1] = cr3 + ci4;
	ch[(k + (ch_dim2 << 2)) * ch_dim1 + 2] = ci3 - cr4;
	ch[(k + ch_dim2 * 5) * ch_dim1 + 2] = ci2 - cr5;
/* L101: */
    }
    return 0;
L102:
    i_1 = *l1;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *ido;
	for (i = 2; i <= i_2; i += 2) {
	    ti5 = cc[i + (k * 5 + 2) * cc_dim1] - cc[i + (k * 5 + 5) * 
		    cc_dim1];
	    ti2 = cc[i + (k * 5 + 2) * cc_dim1] + cc[i + (k * 5 + 5) * 
		    cc_dim1];
	    ti4 = cc[i + (k * 5 + 3) * cc_dim1] - cc[i + (k * 5 + 4) * 
		    cc_dim1];
	    ti3 = cc[i + (k * 5 + 3) * cc_dim1] + cc[i + (k * 5 + 4) * 
		    cc_dim1];
	    tr5 = cc[i - 1 + (k * 5 + 2) * cc_dim1] - cc[i - 1 + (k * 5 + 5) *
		     cc_dim1];
	    tr2 = cc[i - 1 + (k * 5 + 2) * cc_dim1] + cc[i - 1 + (k * 5 + 5) *
		     cc_dim1];
	    tr4 = cc[i - 1 + (k * 5 + 3) * cc_dim1] - cc[i - 1 + (k * 5 + 4) *
		     cc_dim1];
	    tr3 = cc[i - 1 + (k * 5 + 3) * cc_dim1] + cc[i - 1 + (k * 5 + 4) *
		     cc_dim1];
	    ch[i - 1 + (k + ch_dim2) * ch_dim1] = cc[i - 1 + (k * 5 + 1) * 
		    cc_dim1] + tr2 + tr3;
	    ch[i + (k + ch_dim2) * ch_dim1] = cc[i + (k * 5 + 1) * cc_dim1] + 
		    ti2 + ti3;
	    cr2 = cc[i - 1 + (k * 5 + 1) * cc_dim1] + tr11 * tr2 + tr12 * tr3;

	    ci2 = cc[i + (k * 5 + 1) * cc_dim1] + tr11 * ti2 + tr12 * ti3;
	    cr3 = cc[i - 1 + (k * 5 + 1) * cc_dim1] + tr12 * tr2 + tr11 * tr3;

	    ci3 = cc[i + (k * 5 + 1) * cc_dim1] + tr12 * ti2 + tr11 * ti3;
	    cr5 = ti11 * tr5 + ti12 * tr4;
	    ci5 = ti11 * ti5 + ti12 * ti4;
	    cr4 = ti12 * tr5 - ti11 * tr4;
	    ci4 = ti12 * ti5 - ti11 * ti4;
	    dr3 = cr3 - ci4;
	    dr4 = cr3 + ci4;
	    di3 = ci3 + cr4;
	    di4 = ci3 - cr4;
	    dr5 = cr2 + ci5;
	    dr2 = cr2 - ci5;
	    di5 = ci2 - cr5;
	    di2 = ci2 + cr5;
	    ch[i - 1 + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * dr2 + 
		    wa1[i] * di2;
	    ch[i + (k + (ch_dim2 << 1)) * ch_dim1] = wa1[i - 1] * di2 - wa1[i]
		     * dr2;
	    ch[i - 1 + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * dr3 + wa2[
		    i] * di3;
	    ch[i + (k + ch_dim2 * 3) * ch_dim1] = wa2[i - 1] * di3 - wa2[i] * 
		    dr3;
	    ch[i - 1 + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * dr4 + 
		    wa3[i] * di4;
	    ch[i + (k + (ch_dim2 << 2)) * ch_dim1] = wa3[i - 1] * di4 - wa3[i]
		     * dr4;
	    ch[i - 1 + (k + ch_dim2 * 5) * ch_dim1] = wa4[i - 1] * dr5 + wa4[
		    i] * di5;
	    ch[i + (k + ch_dim2 * 5) * ch_dim1] = wa4[i - 1] * di5 - wa4[i] * 
		    dr5;
/* L103: */
	}
/* L104: */
    }
    return 0;
} /* passf5_ */

/* Subroutine */ int passb4(int *ido, int *l1, float *cc,
	float *ch, float *wa1, float *wa2, float *wa3);

/* Subroutine */ int cfftb1(int *n, float *c, float *ch, 
	float *wa, int *ifac)
{
    /* System generated locals */
    int i_1;

    /* Local variables */
    static int idot, i;
    static int k1, l1, l2, n2;
    static int na, nf, ip, iw, ix2, ix3, ix4, nac, ido, idl1;

    /* Parameter adjustments */
    --c;
    --ch;
    --wa;
    --ifac;

    /* Function Body */
    nf = ifac[2];
    na = 0;
    l1 = 1;
    iw = 1;
    i_1 = nf;
    for (k1 = 1; k1 <= i_1; ++k1) {
	ip = ifac[k1 + 2];
	l2 = ip * l1;
	ido = *n / l2;
	idot = ido + ido;
	idl1 = idot * l1;
	if (ip != 4) {
	    goto L103;
	}
	ix2 = iw + idot;
	ix3 = ix2 + idot;
	if (na != 0) {
	    goto L101;
	}
	passb4(&idot, &l1, &c[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3]);
	goto L102;
L101:
	passb4(&idot, &l1, &ch[1], &c[1], &wa[iw], &wa[ix2], &wa[ix3]);
L102:
	na = 1 - na;
	goto L115;
L103:
	if (ip != 2) {
	    goto L106;
	}
	if (na != 0) {
	    goto L104;
	}
	passb2(&idot, &l1, &c[1], &ch[1], &wa[iw]);
	goto L105;
L104:
	passb2(&idot, &l1, &ch[1], &c[1], &wa[iw]);
L105:
	na = 1 - na;
	goto L115;
L106:
	if (ip != 3) {
	    goto L109;
	}
	ix2 = iw + idot;
	if (na != 0) {
	    goto L107;
	}
	passb3(&idot, &l1, &c[1], &ch[1], &wa[iw], &wa[ix2]);
	goto L108;
L107:
	passb3(&idot, &l1, &ch[1], &c[1], &wa[iw], &wa[ix2]);
L108:
	na = 1 - na;
	goto L115;
L109:
	if (ip != 5) {
	    goto L112;
	}
	ix2 = iw + idot;
	ix3 = ix2 + idot;
	ix4 = ix3 + idot;
	if (na != 0) {
	    goto L110;
	}
	passb5(&idot, &l1, &c[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
	goto L111;
L110:
	passb5(&idot, &l1, &ch[1], &c[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
L111:
	na = 1 - na;
	goto L115;
L112:
	if (na != 0) {
	    goto L113;
	}
	passb(&nac, &idot, &ip, &l1, &idl1, &c[1], &c[1], &c[1], &ch[1], &ch[
		1], &wa[iw]);
	goto L114;
L113:
	passb(&nac, &idot, &ip, &l1, &idl1, &ch[1], &ch[1], &ch[1], &c[1], &
		c[1], &wa[iw]);
L114:
	if (nac != 0) {
	    na = 1 - na;
	}
L115:
	l1 = l2;
	iw += (ip - 1) * idot;
/* L116: */
    }
    if (na == 0) {
	return 0;
    }
    n2 = *n + *n;
    i_1 = n2;
    for (i = 1; i <= i_1; ++i) {
	c[i] = ch[i];
/* L117: */
    }
    return 0;
} /* cfftb1_ */


/* Subroutine */ int cfftb(int *n, float *c, float *wsave)
{
    static int iw1, iw2;

    /* Parameter adjustments */
    --c;
    --wsave;

    /* Function Body */
    if (*n == 1) {
	return 0;
    }
    iw1 = *n + *n + 1;
    iw2 = iw1 + *n + *n;
    cfftb1(n, &c[1], &wsave[1], &wsave[iw1], (int*) &wsave[iw2]);
    return 0;
} /* cfftb_ */


/* Subroutine */ int cfftf1(int *n, float *c, float *ch, 
	float *wa, int *ifac)
{
    /* System generated locals */
    int i_1;

    /* Local variables */
    static int idot, i;
    static int k1, l1, l2, n2;
    static int na, nf, ip, iw, ix2, ix3, ix4, nac, ido, idl1;

    /* Parameter adjustments */
    --c;
    --ch;
    --wa;
    --ifac;

    /* Function Body */
    nf = ifac[2];
    na = 0;
    l1 = 1;
    iw = 1;
    i_1 = nf;
    for (k1 = 1; k1 <= i_1; ++k1) {
	ip = ifac[k1 + 2];
	l2 = ip * l1;
	ido = *n / l2;
	idot = ido + ido;
	idl1 = idot * l1;
	if (ip != 4) {
	    goto L103;
	}
	ix2 = iw + idot;
	ix3 = ix2 + idot;
	if (na != 0) {
	    goto L101;
	}
	passf4(&idot, &l1, &c[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3]);
	goto L102;
L101:
	passf4(&idot, &l1, &ch[1], &c[1], &wa[iw], &wa[ix2], &wa[ix3]);
L102:
	na = 1 - na;
	goto L115;
L103:
	if (ip != 2) {
	    goto L106;
	}
	if (na != 0) {
	    goto L104;
	}
	passf2(&idot, &l1, &c[1], &ch[1], &wa[iw]);
	goto L105;
L104:
	passf2(&idot, &l1, &ch[1], &c[1], &wa[iw]);
L105:
	na = 1 - na;
	goto L115;
L106:
	if (ip != 3) {
	    goto L109;
	}
	ix2 = iw + idot;
	if (na != 0) {
	    goto L107;
	}
	passf3(&idot, &l1, &c[1], &ch[1], &wa[iw], &wa[ix2]);
	goto L108;
L107:
	passf3(&idot, &l1, &ch[1], &c[1], &wa[iw], &wa[ix2]);
L108:
	na = 1 - na;
	goto L115;
L109:
	if (ip != 5) {
	    goto L112;
	}
	ix2 = iw + idot;
	ix3 = ix2 + idot;
	ix4 = ix3 + idot;
	if (na != 0) {
	    goto L110;
	}
	passf5(&idot, &l1, &c[1], &ch[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
	goto L111;
L110:
	passf5(&idot, &l1, &ch[1], &c[1], &wa[iw], &wa[ix2], &wa[ix3], &wa[
		ix4]);
L111:
	na = 1 - na;
	goto L115;
L112:
	if (na != 0) {
	    goto L113;
	}
	passf(&nac, &idot, &ip, &l1, &idl1, &c[1], &c[1], &c[1], &ch[1], &ch[
		1], &wa[iw]);
	goto L114;
L113:
	passf(&nac, &idot, &ip, &l1, &idl1, &ch[1], &ch[1], &ch[1], &c[1], &
		c[1], &wa[iw]);
L114:
	if (nac != 0) {
	    na = 1 - na;
	}
L115:
	l1 = l2;
	iw += (ip - 1) * idot;
/* L116: */
    }
    if (na == 0) {
	return 0;
    }
    n2 = *n + *n;
    i_1 = n2;
    for (i = 1; i <= i_1; ++i) {
	c[i] = ch[i];
/* L117: */
    }
    return 0;
} /* cfftf1_ */


/* Subroutine */ int cfftf(int *n, float *c, float *wsave)
{
    static int iw1, iw2;

    /* Parameter adjustments */
    --c;
    --wsave;

    /* Function Body */
    if (*n == 1) {
	return 0;
    }
    iw1 = *n + *n + 1;
    iw2 = iw1 + *n + *n;
    cfftf1(n, &c[1], &wsave[1], &wsave[iw1], (int*) &wsave[iw2]);
    return 0;
} /* cfftf_ */


/* Subroutine */ int cffti1(int *n, float *wa, int *ifac)
{
    /* Initialized data */

    static int ntryh[4] = { 3,4,2,5 };

    /* System generated locals */
    int i_1, i_2, i_3;

    /* Local variables */
    static float argh;
    static int idot, ntry, i, j;
    static float argld;
    static int i1, k1, l1, l2, ib;
    static float fi;
    static int ld, ii, nf, ip, nl, nq, nr;
    static float arg;
    static int ido, ipm;
    static float tpi;

    /* Parameter adjustments */
    --wa;
    --ifac;

    /* Function Body */
    nl = *n;
    nf = 0;
    j = 0;
L101:
    ++j;
    if (j - 4 <= 0) {
	goto L102;
    } else {
	goto L103;
    }
L102:
    ntry = ntryh[j - 1];
    goto L104;
L103:
    ntry += 2;
L104:
    nq = nl / ntry;
    nr = nl - ntry * nq;
    if (nr != 0) {
	goto L101;
    } else {
	goto L105;
    }
L105:
    ++nf;
    ifac[nf + 2] = ntry;
    nl = nq;
    if (ntry != 2) {
	goto L107;
    }
    if (nf == 1) {
	goto L107;
    }
    i_1 = nf;
    for (i = 2; i <= i_1; ++i) {
	ib = nf - i + 2;
	ifac[ib + 2] = ifac[ib + 1];
/* L106: */
    }
    ifac[3] = 2;
L107:
    if (nl != 1) {
	goto L104;
    }
    ifac[1] = *n;
    ifac[2] = nf;
    tpi = 6.28318530717959;
    argh = tpi / (float) (*n);
    i = 2;
    l1 = 1;
    i_1 = nf;
    for (k1 = 1; k1 <= i_1; ++k1) {
	ip = ifac[k1 + 2];
	ld = 0;
	l2 = l1 * ip;
	ido = *n / l2;
	idot = ido + ido + 2;
	ipm = ip - 1;
	i_2 = ipm;
	for (j = 1; j <= i_2; ++j) {
	    i1 = i;
	    wa[i - 1] = 1.;
	    wa[i] = 0.;
	    ld += l1;
	    fi = 0.;
	    argld = (float) ld * argh;
	    i_3 = idot;
	    for (ii = 4; ii <= i_3; ii += 2) {
		i += 2;
		fi += 1.;
		arg = fi * argld;
		wa[i - 1] = cos(arg);
		wa[i] = sin(arg);
/* L108: */
	    }
	    if (ip <= 5) {
		goto L109;
	    }
	    wa[i1 - 1] = wa[i - 1];
	    wa[i1] = wa[i];
L109:
	;}
	l1 = l2;
/* L110: */
    }
    return 0;
} /* cffti1_ */

/* Subroutine */ int cffti(int *n, float *wsave)
{
    static int iw1, iw2;

    /* Parameter adjustments */
    --wsave;

    /* Function Body */
    if (*n == 1) {
	return 0;
    }
    iw1 = *n + *n + 1;
    iw2 = iw1 + *n + *n;
    cffti1(n, &wsave[iw1], (int*) &wsave[iw2]);
    return 0;
} /* cffti_ */


/* typedef struct { float r, i; } floatcomplex; */

/****************************************************************************
**/

/* 	3D (slow) Fourier Transform */
/*   this 1d->3d code is brute force approach */
/*   the 1d code is a float precision version of fftpack from netlib */
/*   due to Paul N Swartztrauber at NCAR Boulder Coloraso */

/****************************************************************************
**/
/* Subroutine */ int pubz3di(int *n1, int *n2, int *n3, 
	float *table, int *ntable)
{
    /* System generated locals */
    int table_dim1, table_offset;

    /* Local variables */

    /* Parameter adjustments */
    table_dim1 = *ntable;
    table_offset = table_dim1 + 1;
    table -= table_offset;

    /* Function Body */
/* ntable should be 4*max(n1,n2,n3) +15 */
    cffti(n1, &table[table_dim1 + 1]);
    cffti(n2, &table[(table_dim1 << 1) + 1]);
    cffti(n3, &table[table_dim1 * 3 + 1]);
    return 0;
} /* pubz3di_ */

/****************************************************************************
**/
/* Subroutine */ int pubz3d(int *isign, int *n1, int *n2, 
	int *n3, floatcomplex *w, int *ld1, int *ld2, float 
	*table, int *ntable, floatcomplex *work /*, int * DUMMY1 nwork */)
{
    /* System generated locals */
    int w_dim1, w_dim2, w_offset, table_dim1, table_offset, i_1, i_2, i_3,
	     i_4, i_5;

    /* Local variables */
    static int i, j, k;

    /* Parameter adjustments */
    w_dim1 = *ld1;
    w_dim2 = *ld2;
    w_offset = w_dim1 * (w_dim2 + 1) + 1;
    w -= w_offset;
    table_dim1 = *ntable;
    table_offset = table_dim1 + 1;
    table -= table_offset;
    --work;

    /* Function Body */
/* ntable should be 4*max(n1,n2,n3) +15 */
/* nwork should be max(n1,n2,n3) */

/*   transform along X  first ... */

    i_1 = *n3;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *n2;
	for (j = 1; j <= i_2; ++j) {
	    i_3 = *n1;
	    for (i = 1; i <= i_3; ++i) {
		i_4 = i;
		i_5 = i + (j + k * w_dim2) * w_dim1;
		work[i_4].r = w[i_5].r, work[i_4].i = w[i_5].i;
/* L70: */
	    }
	    if (*isign == -1) {
		cfftf(n1, (float*) &work[1], &table[table_dim1 + 1]);
	    }
	    if (*isign == 1) {
		cfftb(n1, (float*) &work[1], &table[table_dim1 + 1]);
	    }
	    i_3 = *n1;
	    for (i = 1; i <= i_3; ++i) {
		i_4 = i + (j + k * w_dim2) * w_dim1;
		i_5 = i;
		w[i_4].r = work[i_5].r, w[i_4].i = work[i_5].i;
/* L80: */
	    }
/* L90: */
	}
/* L100: */
    }

/*   transform along Y then ... */

    i_1 = *n3;
    for (k = 1; k <= i_1; ++k) {
	i_2 = *n1;
	for (i = 1; i <= i_2; ++i) {
	    i_3 = *n2;
	    for (j = 1; j <= i_3; ++j) {
		i_4 = j;
		i_5 = i + (j + k * w_dim2) * w_dim1;
		work[i_4].r = w[i_5].r, work[i_4].i = w[i_5].i;
/* L170: */
	    }
	    if (*isign == -1) {
		cfftf(n2, (float*) &work[1], &table[(table_dim1 << 1) + 1]);
	    }
	    if (*isign == 1) {
		cfftb(n2, (float*) &work[1], &table[(table_dim1 << 1) + 1]);
	    }
	    i_3 = *n2;
	    for (j = 1; j <= i_3; ++j) {
		i_4 = i + (j + k * w_dim2) * w_dim1;
		i_5 = j;
		w[i_4].r = work[i_5].r, w[i_4].i = work[i_5].i;
/* L180: */
	    }
/* L190: */
	}
/* L200: */
    }

/*   transform along Z finally ... */

    i_1 = *n1;
    for (i = 1; i <= i_1; ++i) {
	i_2 = *n2;
	for (j = 1; j <= i_2; ++j) {
	    i_3 = *n3;
	    for (k = 1; k <= i_3; ++k) {
		i_4 = k;
		i_5 = i + (j + k * w_dim2) * w_dim1;
		work[i_4].r = w[i_5].r, work[i_4].i = w[i_5].i;
/* L270: */
	    }
	    if (*isign == -1) {
		cfftf(n3, (float*) &work[1], &table[table_dim1 * 3 + 1]);
	    }
	    if (*isign == 1) {
		cfftb(n3, (float*) &work[1], &table[table_dim1 * 3 + 1]);
	    }
	    i_3 = *n3;
	    for (k = 1; k <= i_3; ++k) {
		i_4 = i + (j + k * w_dim2) * w_dim1;
		i_5 = k;
		w[i_4].r = work[i_5].r, w[i_4].i = work[i_5].i;
/* L280: */
	    }
/* L290: */
	}
/* L300: */
    }
    return 0;
} /* pubz3d_ */

/****************************************************************************/

int pubd3di(int n1, int n2, int n3, float *table, int ntable) {

  int n1over2;

  n1over2 = n1 / 2;
  return pubz3di(&n1over2,&n2,&n3,table,&ntable);

} /* pubd3di */

/****************************************************************************/
/* real to complex fft */

int pubdz3d(int isign, int n1, int n2,
   int n3, float *w, int ld1, int ld2, float
   *table, int ntable, float *work) {

  int n1over2, ld1over2, rval;
  int i, j, j2, k, k2, i1, i2, imax;
  float *data, *data2;
  float TwoPiOverN, tmp1r, tmp1i, tmp2r, tmp2i;

  /* complex transform */
  n1over2 = n1 / 2;
  ld1over2 = ld1 / 2;
  rval = pubz3d(&isign, &n1over2, &n2, &n3, (floatcomplex*) w,
         &ld1over2, &ld2, table, &ntable, (floatcomplex*) work);

  /* rearrange data */
  TwoPiOverN = isign * 2.0 * M_PI / n1;
  imax = n1/4+1;
  for ( i=0; i<imax; ++i ) {
    work[2*i] = cos(i * TwoPiOverN);
    work[2*i+1] = sin(i * TwoPiOverN);
  }
  for ( k=0; k<n3; ++k ) {
    for ( j=0; j<n2; ++j ) {
      data = w + ld1*(ld2*k + j);
      data[n1] = data[0];
      data[n1+1] = data[1];
    }
  }
  for ( k=0; k<n3; ++k ) {
    k2 = k?(n3-k):0;
    for ( j=0; j<n2; ++j ) {
      j2 = j?(n2-j):0;
      data = w + ld1*(ld2*k + j);
      data2 = w + ld1*(ld2*k2 + j2);
      imax = n1/4;
      if ( (n1/2) & 1 ) imax += 1;
      else {
        if ( (2*j<n2) || (2*j==n2 && 2*k<=n3) ) imax +=1;
        if ( j==0 && 2*k>n3 ) imax -=1;
      }
      for ( i=0; i<imax; ++i ) {
	i1 = 2*i;  i2 = n1-i1;
	tmp1r = data[i1] - data2[i2];
	tmp1i = data[i1+1] + data2[i2+1];
	tmp2r = tmp1r * work[i1+1] + tmp1i * work[i1];
	tmp2i = tmp1i * work[i1+1] - tmp1r * work[i1];
	tmp1r = data[i1] + data2[i2];
	tmp1i = data[i1+1] - data2[i2+1];
	data[i1] = 0.5 * ( tmp1r + tmp2r );
	data[i1+1] = 0.5 * ( tmp1i + tmp2i );
	data2[i2] = 0.5 * ( tmp1r - tmp2r );
	data2[i2+1] = 0.5 * ( tmp2i - tmp1i );
      }
    }
  }

  return rval;

} /* pubdz3d */

/****************************************************************************/
/* complex to real fft */

int pubzd3d(int isign, int n1, int n2,
   int n3, float *w, int ld1, int ld2, float
   *table, int ntable, float *work) {

  int n1over2, ld1over2;
  int i, j, j2, k, k2, i1, i2, imax;
  float *data, *data2;
  float TwoPiOverN, tmp1r, tmp1i, tmp2r, tmp2i;

  /* rearrange data */
  TwoPiOverN = isign * 2.0 * M_PI / n1;
  imax = n1/4+1;
  for ( i=0; i<imax; ++i ) {
    work[2*i] = -cos(i * TwoPiOverN);
    work[2*i+1] = -sin(i * TwoPiOverN);
  }
  for ( k=0; k<n3; ++k ) {
    k2 = k?(n3-k):0;
    for ( j=0; j<n2; ++j ) {
      j2 = j?(n2-j):0;
      data = w + ld1*(ld2*k + j);
      data2 = w + ld1*(ld2*k2 + j2);
      imax = n1/4;
      if ( (n1/2) & 1 ) imax += 1;
      else {
        if ( (2*j<n2) || (2*j==n2 && 2*k<=n3) ) imax +=1;
        if ( j==0 && 2*k>n3 ) imax -=1;
      }
      for ( i=0; i<imax; ++i ) {
	i1 = 2*i;  i2 = n1-i1;
	tmp1r = data[i1] - data2[i2];
	tmp1i = data[i1+1] + data2[i2+1];
	tmp2r = tmp1r * work[i1+1] + tmp1i * work[i1];
	tmp2i = tmp1i * work[i1+1] - tmp1r * work[i1];
	tmp1r = data[i1] + data2[i2];
	tmp1i = data[i1+1] - data2[i2+1];
	data[i1] = tmp1r + tmp2r;
	data[i1+1] = tmp1i + tmp2i;
	data2[i2] = tmp1r - tmp2r;
	data2[i2+1] = tmp2i - tmp1i;
      }
    }
  }

  /* complex transform */
  n1over2 = n1 / 2;
  ld1over2 = ld1 / 2;
  return pubz3d(&isign, &n1over2, &n2, &n3, (floatcomplex*) w,
         &ld1over2, &ld2, table, &ntable, (floatcomplex*) work);

} /* pubzd3d */

/****************************************************************************/

