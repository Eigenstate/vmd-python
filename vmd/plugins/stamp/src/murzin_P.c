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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* returns a Murzin type P-value given a STAMP alignment of
 * See A.G. Murzin "Sweet tasting protein monellin is related to the cystatin family of 
 *  thiol proteinase inhibitors" J. Mol. Biol., 230, 689-694, 1993
 *
 * n  is the number of structurally equivalent positions
 * m  is the number of identical positions
 *
 * the routine will return -1.0 if an error is encountered */

double Fct(int N) {
	int i;
	double F;

	if(N>150) {
		F = 1e262;
	} else {
		F = 1.0;
		for(i=1; i<=N; ++i) {
			F*=(double)i;
		}
	}
	return F;
}

double murzin_P(int n,int m,double p) {


	double Pm;
	double sigma;
	double m_o;

	double t1,t2;

	sigma = sqrt(n * p * (1-p));
	m_o   = n*p;

        if(m<=(m_o+sigma)) { /* Test for validity of P calculation */
	    Pm = 1.0;
	} else {
	   t1 = Fct(n)/(Fct(m)*Fct(n-m));
	   t2 = (double)pow((double)p,(double)m)*(double)pow((double)(1-p),(double)(n-m));
	   Pm = t1 * t2;
	}
/*	printf("\n MURZIN_P sigma = %f m_o = %f P(p=%f) = %e * %e = %e\n",sigma,m_o,p,t1,t2,Pm);  */

	if(Pm<1e-100) { Pm=0.0; }
	return Pm;
}
