/*
 * Modified Hershey Roman font rendering code.
 *
 * $Id: Hershey.C,v 1.20 2015/12/03 18:03:14 johns Exp $
 */

//
//   Hershey.C
//   extracted from the hershey font
//   
//   Charles Schwieters 6/14/99
//   Various tweaks and updates by John Stone
//
//   font info:
//
//Peter Holzmann, Octopus Enterprises
//USPS: 19611 La Mar Court, Cupertino, CA 95014
//UUCP: {hplabs!hpdsd,pyramid}!octopus!pete
//Phone: 408/996-7746
//
//This distribution is made possible through the collective encouragement
//of the Usenet Font Consortium, a mailing list that sprang to life to get
//this accomplished and that will now most likely disappear into the mists
//of time... Thanks are especially due to Jim Hurt, who provided the packed
//font data for the distribution, along with a lot of other help.
//
//This file describes the Hershey Fonts in general, along with a description of
//the other files in this distribution and a simple re-distribution restriction.
//
//USE RESTRICTION:
//	  This distribution of the Hershey Fonts may be used by anyone for
//	  any purpose, commercial or otherwise, providing that:
//		  1. The following acknowledgements must be distributed with
//			  the font data:
//			  - The Hershey Fonts were originally created by Dr.
//				  A. V. Hershey while working at the U. S.
//				  National Bureau of Standards.
//			  - The format of the Font data in this distribution
//				  was originally created by
//					  James Hurt
//					  Cognition, Inc.
//					  900 Technology Park Drive
//					  Billerica, MA 01821
//					  (mit-eddie!ci-dandelion!hurt)
//		  2. The font data in this distribution may be converted into
//			  any other format *EXCEPT* the format distributed by
//			  the U.S. NTIS (which organization holds the rights
//			  to the distribution and use of the font data in that
//			  particular format). Not that anybody would really
//			  *want* to use their format... each point is described
//			  in eight bytes as "xxx yyy:", where xxx and yyy are
//			  the coordinate values as ASCII numbers.
//
//*PLEASE* be reassured: The legal implications of NTIS' attempt to control
//a particular form of the Hershey Fonts *are* troubling. HOWEVER: We have
//been endlessly and repeatedly assured by NTIS that they do not care what
//we do with our version of the font data, they do not want to know about it,
//they understand that we are distributing this information all over the world,
//etc etc etc... but because it isn't in their *exact* distribution format, they
//just don't care!!! So go ahead and use the data with a clear conscience! (If
//you feel bad about it, take a smaller deduction for something on your taxes
//next week...)
//
//The Hershey Fonts:
//	  - are a set of more than 2000 glyph (symbol) descriptions in vector
//		  ( &lt;x,y&gt; point-to-point ) format
//	  - can be grouped as almost 20 'occidental' (english, greek,
//		  cyrillic) fonts, 3 or more 'oriental' (Kanji, Hiragana,
//		  and Katakana) fonts, and a few hundred miscellaneous
//		  symbols (mathematical, musical, cartographic, etc etc)
//	  - are suitable for typographic quality output on a vector device
//		  (such as a plotter) when used at an appropriate scale.
//	  - were digitized by Dr. A. V. Hershey while working for the U.S.
//		  Government National Bureau of Standards (NBS).
//	  - are in the public domain, with a few caveats:
//		  - They are available from NTIS (National Technical Info.
//			  Service) in a computer-readable from which is *not*
//			  in the public domain. This format is described in
//			  a hardcopy publication "Tables of Coordinates for
//			  Hershey's Repertory of Occidental Type Fonts and
//			  Graphic Symbols" available from NTIS for less than
//			  $20 US (phone number +1 703 487 4763).
//		  - NTIS does not care about and doesn't want to know about
//			  what happens to Hershey Font data that is not
//			  distributed in their exact format.
//		  - This distribution is not in the NTIS format, and thus is
//			  only subject to the simple restriction described
//			  at the top of this file.
//

#include "Hershey.h"

#if defined(_MSC_VER)
#include <windows.h>
#endif

#if defined(VMDOPENGL) || defined(VMDOPENGLPBUFFER) || defined(VMDEGLPBUFFER)
#if defined(__APPLE__) && !defined (VMDMESA)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#endif

/*
 *  The Hershey romans font in ascii order (first 32 places held by space)
 *  NOTE: This font has been modified to yield fixed-width numeric 
 *        characters.  This makes it possible to produce justified numeric 
 *        text that correctly lines up in columns.
 *        The font was specifically changed for:
 *        ' ' (ascii 32), '+' (ascii 43), '-' (ascii 45), and '.' (ascii 46)
 */

#define VMDHERSHEYFIXEDNUMERICS 1

char* hersheyFontData[] = {
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
(char *) "JZ",
#if defined(VMDHERSHEYFIXEDNUMERICS)
/* force constant width for ' ' (ascii 32) */
(char *) "H\\",
#else
(char *) "JZ",
#endif
(char *) "MWRFRT RRYQZR[SZRY",
(char *) "JZNFNM RVFVM",
(char *) "H]SBLb RYBRb RLOZO RKUYU",
(char *) "H\\PBP_ RTBT_ RYIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX",
(char *) "F^[FI[ RNFPHPJOLMMKMIKIIJGLFNFPGSHVHYG[F RWTUUTWTYV[X[ZZ[X[VYTWT",
(char *) "E_\\O\\N[MZMYNXPVUTXRZP[L[JZIYHWHUISJRQNRMSKSIRGPFNGMIMKNNPQUXWZY[",
(char *) "MWRHQGRFSGSIRKQL",
(char *) "KYVBTDRGPKOPOTPYR]T`Vb",
(char *) "KYNBPDRGTKUPUTTYR]P`Nb",
(char *) "JZRLRX RMOWU RWOMU",
#if defined(VMDHERSHEYFIXEDNUMERICS)
/* force constant width for '+' (ascii 43) */
(char *) "H\\RIR[ RIR[R",
#else
(char *) "E_RIR[ RIR[R",
#endif
(char *) "NVSWRXQWRVSWSYQ[",
#if defined(VMDHERSHEYFIXEDNUMERICS)
/* force constant width for '-' (ascii 45) */
(char *) "H\\IR[R",
#else
(char *) "E_IR[R",
#endif
#if defined(VMDHERSHEYFIXEDNUMERICS)
/* force constant width for '.' (ascii 46) */
(char *) "H\\RVQWRXSWRV",
#else
(char *) "NVRVQWRXSWRV",
#endif
(char *) "G][BIb",
(char *) "H\\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF",
(char *) "H\\NJPISFS[",
(char *) "H\\LKLJMHNGPFTFVGWHXJXLWNUQK[Y[",
(char *) "H\\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW",
(char *) "H\\UFKTZT RUFU[",
(char *) "H\\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW",
(char *) "H\\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT",
(char *) "H\\YFO[ RKFYF",
(char *) "H\\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF",
(char *) "H\\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX",
(char *) "NVROQPRQSPRO RRVQWRXSWRV",
(char *) "NVROQPRQSPRO RSWRXQWRVSWSYQ[",
(char *) "F^ZIJRZ[",
(char *) "E_IO[O RIU[U",
(char *) "F^JIZRJ[",
(char *) "I[LKLJMHNGPFTFVGWHXJXLWNVORQRT RRYQZR[SZRY",
(char *) "E`WNVLTKQKOLNMMPMSNUPVSVUUVS RQKOMNPNSOUPV RWKVSVUXVZV\\T]Q]O\\L[J",
(char *) "I[RFJ[ RRFZ[ RMTWT",
(char *) "G\\KFK[ RKFTFWGXHYJYLXNWOTP RKPTPWQXRYTYWXYWZT[K[",
(char *) "H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZV",
(char *) "G\\KFK[ RKFRFUGWIXKYNYSXVWXUZR[K[",
(char *) "H[LFL[ RLFYF RLPTP RL[Y[",
(char *) "HZLFL[ RLFYF RLPTP",
(char *) "H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZVZS RUSZS",
(char *) "G]KFK[ RYFY[ RKPYP",
(char *) "NVRFR[",
(char *) "JZVFVVUYTZR[P[NZMYLVLT",
(char *) "G\\KFK[ RYFKT RPOY[",
(char *) "HYLFL[ RL[X[",
(char *) "F^JFJ[ RJFR[ RZFR[ RZFZ[",
(char *) "G]KFK[ RKFY[ RYFY[",
(char *) "G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF",
(char *) "G\\KFK[ RKFTFWGXHYJYMXOWPTQKQ",
(char *) "G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF RSWY]",
(char *) "G\\KFK[ RKFTFWGXHYJYLXNWOTPKP RRPY[",
(char *) "H\\YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX",
(char *) "JZRFR[ RKFYF",
(char *) "G]KFKULXNZQ[S[VZXXYUYF",
(char *) "I[JFR[ RZFR[",
(char *) "F^HFM[ RRFM[ RRFW[ R\\FW[",
(char *) "H\\KFY[ RYFK[",
(char *) "I[JFRPR[ RZFRP",
(char *) "H\\YFK[ RKFYF RK[Y[",
(char *) "KYOBOb RPBPb ROBVB RObVb",
(char *) "KYKFY^",
(char *) "KYTBTb RUBUb RNBUB RNbUb",
(char *) "JZRDJR RRDZR",
(char *) "I[Ib[b",
(char *) "NVSKQMQORPSORNQO",
(char *) "I\\XMX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX",
(char *) "H[LFL[ RLPNNPMSMUNWPXSXUWXUZS[P[NZLX",
(char *) "I[XPVNTMQMONMPLSLUMXOZQ[T[VZXX",
(char *) "I\\XFX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX",
(char *) "I[LSXSXQWOVNTMQMONMPLSLUMXOZQ[T[VZXX",
(char *) "MYWFUFSGRJR[ ROMVM",
(char *) "I\\XMX]W`VaTbQbOa RXPVNTMQMONMPLSLUMXOZQ[T[VZXX",
(char *) "I\\MFM[ RMQPNRMUMWNXQX[",
(char *) "NVQFRGSFREQF RRMR[",
(char *) "MWRFSGTFSERF RSMS^RaPbNb",
(char *) "IZMFM[ RWMMW RQSX[",
(char *) "NVRFR[",
(char *) "CaGMG[ RGQJNLMOMQNRQR[ RRQUNWMZM\\N]Q][",
(char *) "I\\MMM[ RMQPNRMUMWNXQX[",
(char *) "I\\QMONMPLSLUMXOZQ[T[VZXXYUYSXPVNTMQM",
(char *) "H[LMLb RLPNNPMSMUNWPXSXUWXUZS[P[NZLX",
(char *) "I\\XMXb RXPVNTMQMONMPLSLUMXOZQ[T[VZXX",
(char *) "KXOMO[ ROSPPRNTMWM",
(char *) "J[XPWNTMQMNNMPNRPSUTWUXWXXWZT[Q[NZMX",
(char *) "MYRFRWSZU[W[ ROMVM",
(char *) "I\\MMMWNZP[S[UZXW RXMX[",
(char *) "JZLMR[ RXMR[",
(char *) "G]JMN[ RRMN[ RRMV[ RZMV[",
(char *) "J[MMX[ RXMM[",
(char *) "JZLMR[ RXMR[P_NaLbKb",
(char *) "J[XMM[ RMMXM RM[X[",
(char *) "KYTBRCQDPFPHQJRKSMSOQQ RRCQEQGRISJTLTNSPORSTTVTXSZR[Q]Q_Ra RQSSU",
(char *) "NVRBRb",
(char *) "KYPBRCSDTFTHSJRKQMQOSQ RRCSESGRIQJPLPNQPURQTPVPXQZR[S]S_Ra RSSQU",
(char *) "F^IUISJPLONOPPTSVTXTZS[Q RISJQLPNPPQTTVUXUZT[Q[O",
(char *) "JZJFJ[K[KFLFL[M[MFNFN[O[OFPFP[Q[QFRFR[S[SFTFT[U[UFVFV[W[WFXFX[Y[YFZFZ[" };


#define VMD_DEFAULT_FONT_SCALE  0.0015f

//note the arbitrary scaling
inline float h2float(char c) { return VMD_DEFAULT_FONT_SCALE * (c-'R'); }


//  hersheyDrawLetter() interprets the instructions from the array
//  for that letter and renders the letter with line segments.
void hersheyDrawLetterOpenGL(unsigned char ch, int drawendpoints) {
#if defined(VMDOPENGL) || defined(VMDOPENGLPBUFFER) || defined(VMDEGLPBUFFER)
  // note: we map same set of glyphs twice here (using modulo operator)
  const char *cp = hersheyFontData[ch % 128];
  float lm = h2float(cp[0]);
  float rm = h2float(cp[1]);
  const char *p;

  glTranslatef(-lm,0,0);

  // draw font vectors for this glyph
  glBegin(GL_LINE_STRIP);
  for (p=cp+2; (*p); p+=2) {
    if (p[0] == ' ' && p[1] == 'R') {
      glEnd();
      glBegin(GL_LINE_STRIP);
    } else {
      float x =  h2float( p[0] );
      float y = -h2float( p[1] );
      glVertex2f(x,y);
    }
  }
  glEnd();

  // Draw points at each vector endpoint for this glyph, so
  // that we don't get "cracks" between the vectors when
  // rendering with much larger line widths.  This is a 
  // significant improvement for antialiased line/point
  // widths >= 2.0, but non-antialiased lines of width 1.0 or
  // 2.0 often look better without the end points drawn.
  if (drawendpoints) {
    glBegin(GL_POINTS);
    for (p=cp+2; (*p); p+=2) {
      if (p[0] == ' ' && p[1] == 'R') {
        glEnd();
        glBegin(GL_POINTS);
      } else {
        float x =  h2float( p[0] );
        float y = -h2float( p[1] );
        glVertex2f(x,y);
      }
    }
    glEnd();
  }

  glTranslatef(rm,0,0);
#endif
} /* drawLetter */



// font drawing for non-OpenGL renderers
void hersheyDrawInitLetter(hersheyhandle *hh, const char ch, float *lm, float *rm) {
  // note: we map same set of glyphs twice here (using modulo operator)
  const char *cp = hersheyFontData[ch % 128];
  hh->lm = h2float(cp[0]);
  *lm = hh->lm;
  hh->rm = h2float(cp[1]);
  *rm = hh->rm;
  hh->p=cp+2;
}

int hersheyDrawNextLine(hersheyhandle *hh, int *draw, float *x, float *y) {
  if (*(hh->p)) {
    if ( hh->p[0] == ' ' && hh->p[1] == 'R' ) {
      *draw = 0;
    } else {
      *draw = 1;
      *x =  h2float( hh->p[0] );
      *y = -h2float( hh->p[1] );
    }
    hh->p += 2;
    return 0;
  } else {
    hh->p += 2;
    return 1;
  }
}


#if 0
#include <stdio.h>
// interprets the instructions from the array
//  for that letter and prints the translation offsets segments.
void hersheyPrintLetterInfo(char ch) {
  // note: we map same set of glyphs twice here (using modulo operator)
  const char *cp = hersheyFontData[ch % 128];
  float lm = h2float(cp[0]);
  float rm = h2float(cp[1]);
  printf("hershey char: '%c'\n", ch);
  printf("hershey cmds: '%s'\n", cp);
  printf("lm: %f\n", lm);
  printf("rm: %f\n", rm);

#if 0
  printf("pretranslate(%f, %f, %f)\n", -lm, 0.0, 0.0);
  for (const char* p=cp+2 ; *p ; p+=2) {
    if ( p[0] == ' ' && p[1] == 'R' ) {
      printf("\nlinestrip: ");
    } else {
      float x =  h2float( p[0] );
      float y = -h2float( p[1] );
      printf("(%f,%f) ", x, y);
    }
  }
  printf("\nposttranslate(%f, %f, %f)\n", rm, 0.0, 0.0);
#endif

  printf("\n");
}

void hersheyPrintLetterInfo2(char ch) {
  float lm, rm, x, y, ox, oy;
  int draw, odraw;
  hersheyhandle hh;

  hersheyDrawInitLetter(&hh, ch, &lm, &rm);
  printf("pretranslate(%f, %f, %f)\n", -lm, 0.0, 0.0);
  ox=0;
  oy=0;
  odraw=0;
  while (!hersheyDrawNextLine(&hh, &draw, &x, &y)) {
    if (draw && odraw) {
      printf("line: %g %g -> %g %g\n", ox, oy, x, y);
    }

    ox=x;
    oy=y;
    odraw=draw;    
  }

  printf("\nposttranslate(%f, %f, %f)\n", rm, 0.0, 0.0);
}

int main() {
  const char *str="X. ";
  const char *cp=str;

  while (*cp != '\0') {
    hersheyPrintLetterInfo(*cp);
    hersheyPrintLetterInfo2(*cp);
    cp++;
  }

  return 0;
}
#endif
