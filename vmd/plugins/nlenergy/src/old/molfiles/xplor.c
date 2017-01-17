/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "molfiles/xplor.h"

#undef  FEOF
#define FEOF  1

#define SIXTH_ROOT_OF_TWO  1.12246204830937


static int xplor_reader(NL_FILE *f, ForcePrm *p);


int ForcePrm_read_xplor(ForcePrm *fprm, const char *fname) {
  NL_FILE *f;
  int s;

  if (NULL==(f = NL_fopen(fname, "r"))) return ERROR(ERR_FOPEN);
  else if ((s=xplor_reader(f, fprm)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return OK;
}


static int xplor_get_line(NL_FILE *f, char *line, int len);
static boolean xplor_remove_comment(char *line, boolean is_comment);
static int xplor_parse_bond(NL_FILE *f, char *line, ForcePrm *p);
static int xplor_parse_angle(NL_FILE *f, char *line, ForcePrm *p);
static int xplor_parse_dihedral(NL_FILE *f, char *line, ForcePrm *p);
static int xplor_parse_improper(NL_FILE *f, char *line, ForcePrm *p);
static int xplor_parse_nonbonded(NL_FILE *f, char *line, ForcePrm *p);
static int xplor_parse_nbfix(NL_FILE *f, char *line, ForcePrm *p);


#define BUFLEN  512
#define TOKLEN  16
#define TOKEN   "%15[a-zA-Z]"
#define TSKIP   "%*15[a-zA-Z]"
#define ANAME   "%7s"
#define ASKIP   "%*6s"
#define INT32   FMT_INT32
#define ISKIP   "%*d"
#define DREAL   FMT_DREAL
#define WS      " "
#define EQUALS  "%1[=]"
#define EQSKIP  "%*1[=]"
#define EXTRA   "%1s"


int xplor_reader(NL_FILE *f, ForcePrm *p)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  boolean comment_status = FALSE;
  int file_status = OK;
  int s;

  while (OK==(file_status = xplor_get_line(f, buf, BUFLEN))) {
    comment_status = xplor_remove_comment(buf, comment_status);
    if (sscanf(buf, WS TOKEN, tok)==1) {
      if (strcasecmp(tok,"BOND")==0) {
        if ((s=xplor_parse_bond(f, buf, p)) != OK) return ERROR(s);
      }
      else if (strcasecmp(tok,"ANGLE")==0 || strcasecmp(tok,"ANGL")==0) {
        if ((s=xplor_parse_angle(f, buf, p)) != OK) return ERROR(s);
      }
      else if (strcasecmp(tok,"DIHEDRAL")==0 || strcasecmp(tok,"DIHE")==0) {
        if ((s=xplor_parse_dihedral(f, buf, p)) != OK) return ERROR(s);
      }
      else if (strcasecmp(tok,"IMPROPER")==0 || strcasecmp(tok,"IMPR")==0) {
        if ((s=xplor_parse_improper(f, buf, p)) != OK) return ERROR(s);
      }
      else if (strcasecmp(tok,"NONBONDED")==0 || strcasecmp(tok,"NONB")==0) {
        if ((s=xplor_parse_nonbonded(f, buf, p)) != OK) return ERROR(s);
      }
      else if (strcasecmp(tok,"NBFIX")==0 || strcasecmp(tok,"NBFI")==0) {
        if ((s=xplor_parse_nbfix(f, buf, p)) != OK) return ERROR(s);
      }
      else {
        /* silently ignore what we don't recognize */
      }
    }
  }
  if (FEOF != file_status) return ERROR(file_status);
  return OK;
}


int xplor_get_line(NL_FILE *f, char *line, int len)
{
  ASSERT(len >= 3);
  /* make sure file line length does not exceed len-2 characters */
  line[len-2] = '\n';
  line[len-3] = '\n';
  if (NL_fgets(line, len, f) == NULL) {
    if (NL_feof(f)) return FEOF;
    else return ERROR(ERR_FREAD);
  }
  else if (line[len-2] != '\n' && line[len-3] != '\n') return ERROR(ERR_INPUT);
  return OK;
}


boolean xplor_remove_comment(char *line, boolean is_comment)
{
  char tok[TOKLEN];
  char *s = line;

  if ( ! is_comment ) {  /* we are not inside of an inner line comment */
    /* first test for "remarks" line */
    if (sscanf(line, WS TOKEN, tok)==1
        && (strcasecmp(tok, "REMARK")==0 || strcasecmp(tok, "REMARKS")==0)) {
      line[0] = '\0';
    }
    else {
      /* look for end of line comment, marked by '!' */
      if (NULL!=(s = strchr(line, (int)'!'))) {
        *s = '\0';
      }
      /* also look for inner line comment, marked by "{ }" */
      if (NULL!=(s = strchr(line, (int)'{'))) {
        is_comment = TRUE;
      }
    }
  }

  if ( is_comment ) {  /* we are inside of an inner line comment */
    char *t = strchr(s, (int)'}');
    if (NULL != t) {
      memset(s, (int)' ', t-s+1);  /* replace comment with white space */
      is_comment = FALSE;
    }
  }
  return is_comment;
}


int xplor_parse_bond(NL_FILE *f, char *line, ForcePrm *p)
{
  BondPrm b;
  char extra[4] = "";
  int32 id;

  if (sscanf(line, WS TSKIP WS ANAME WS ANAME WS DREAL WS DREAL EXTRA,
        b.atomType[0], b.atomType[1], &b.k, &b.r0, extra) != 4
      || strlen(b.atomType[0]) == 7
      || strlen(b.atomType[1]) == 7) {
    return ERROR(ERR_INPUT);  /* unable to parse BOND entry */
  }
  b.k *= ENERGY_INTERNAL;  /* units conversion */
  if ((id=ForcePrm_add_bondprm(p, &b)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_bondprm(p, &b)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}


int xplor_parse_angle(NL_FILE *f, char *line, ForcePrm *p)
{
  AnglePrm a;
  char tok[TOKLEN];
  int cnt;
  char extra[4] = "";
  int32 id;

  a.k_ub = 0.0;
  a.r_ub = 0.0;
  /* assume Urey-Bradley parameters are given on same line, if present */
  if ((((cnt = sscanf(line, WS TSKIP WS ANAME WS ANAME WS ANAME WS
              DREAL WS DREAL WS TOKEN WS DREAL WS DREAL EXTRA,
              a.atomType[0], a.atomType[1], a.atomType[2],
              &a.k_theta, &a.theta0, tok, &a.k_ub, &a.r_ub, extra)) != 8
          || strcasecmp(tok, "UB") != 0) && cnt != 5)
      || strlen(a.atomType[0]) == 7
      || strlen(a.atomType[1]) == 7
      || strlen(a.atomType[2]) == 7) {
    return ERROR(ERR_INPUT);  /* unable to parse ANGLE entry */
  }
  a.k_theta *= ENERGY_INTERNAL;  /* units conversion */
  a.theta0 *= RADIANS;
  a.k_ub *= ENERGY_INTERNAL;
  if ((id=ForcePrm_add_angleprm(p, &a)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_angleprm(p, &a)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}


int xplor_parse_dihedral(NL_FILE *f, char *line, ForcePrm *p)
{
  DihedPrm d;
  DihedTerm dt;
  char buf[BUFLEN];
  char tok[TOKLEN];
  int cnt = 0, mult = 0, afmt = 0;
  char equals[4] = "";
  char extra[4] = "";
  int32 id;
  int s;

  if ((s=DihedPrm_init(&d)) != OK) return ERROR(s);
  if (((cnt = sscanf(line, WS TSKIP WS ANAME WS ANAME WS ANAME WS ANAME WS
            TOKEN WS INT32,
            d.atomType[0], d.atomType[1], d.atomType[2], d.atomType[3],
            tok, &mult)) == 6
        || (cnt = sscanf(line, WS TSKIP WS ANAME WS ANAME WS ANAME WS ANAME WS
            TOKEN WS EQUALS WS INT32,
            d.atomType[0], d.atomType[1], d.atomType[2], d.atomType[3],
            tok, equals, &mult)) == 7)
      && (afmt = (strlen(d.atomType[0]) == 7
          || strlen(d.atomType[1]) == 7
          || strlen(d.atomType[2]) == 7
          || strlen(d.atomType[3]) == 7)) == 0
      && (strcasecmp(tok, "MULT")==0 || strcasecmp(tok, "MULTIPLE")==0)) {
    if (mult < 1) {  /* illegal multiplicity */
      DihedPrm_done(&d);
      return ERROR(ERR_INPUT);
    }
    else if ((s=DihedPrm_setmaxnum_term(&d, mult)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    dt.term = 0;
    if ((6 == cnt &&
          (cnt = sscanf(line, WS TSKIP WS ASKIP WS ASKIP WS ASKIP WS ASKIP WS
                        TSKIP WS ISKIP WS
                        DREAL WS INT32 WS DREAL EXTRA,
                        &dt.k_dihed, &dt.n, &dt.phi0, extra)) == 3)
        || (7 == cnt &&
          (cnt = sscanf(line, WS TSKIP WS ASKIP WS ASKIP WS ASKIP WS ASKIP WS
                        TSKIP WS EQSKIP WS ISKIP WS
                        DREAL WS INT32 WS DREAL EXTRA,
                        &dt.k_dihed, &dt.n, &dt.phi0, extra)) == 3)) {
      mult--;
      dt.term++;
      dt.k_dihed *= ENERGY_INTERNAL;  /* units conversion */
      dt.phi0 *= -RADIANS;  /* reverse sign to match CHARMM dihed func spec */
      if (dt.n <= 0) {  /* periodicity for DIHEDRAL must be positive */
        DihedPrm_done(&d);
        return ERROR(ERR_INPUT);
      }
      if ((s=DihedPrm_add_term(&d, &dt)) != OK) {
        DihedPrm_done(&d);
        return ERROR(s);
      }
    }
    else if (cnt != 0) {  /* unable to parse DIHEDRAL entry */
      DihedPrm_done(&d);
      return ERROR(ERR_INPUT);
    }
    /* search next "mult" lines in file for multiple dihedral entries */
    while (mult > 0) {
      if ((s=xplor_get_line(f, buf, BUFLEN)) != OK) {
        DihedPrm_done(&d);
        return (FEOF==s ? ERROR(ERR_INPUT) : ERROR(s));
      }
      if (xplor_remove_comment(buf, FALSE)) {
        DihedPrm_done(&d);
        return ERROR(ERR_INPUT);
      }
      if (sscanf(buf, EXTRA, extra) != 1) continue;  /* skip blank lines */
      else if ((cnt = sscanf(buf, WS DREAL WS INT32 WS DREAL EXTRA,
              &dt.k_dihed, &dt.n, &dt.phi0, extra)) != 3) {
        DihedPrm_done(&d);
        return ERROR(ERR_INPUT);
      }
      mult--;
      dt.term++;
      dt.k_dihed *= ENERGY_INTERNAL;  /* units conversion */
      dt.phi0 *= -RADIANS;  /* reverse sign to match CHARMM dihed func spec */
      if (dt.n <= 0) {  /* periodicity for DIHEDRAL must be positive */
        DihedPrm_done(&d);
        return ERROR(ERR_INPUT);
      }
      if ((s=DihedPrm_add_term(&d, &dt)) != OK) {
        DihedPrm_done(&d);
        return ERROR(s);
      }
    }
  }
  else if (4 == cnt && 0 == afmt
      && sscanf(line, WS TSKIP WS ASKIP WS ASKIP WS ASKIP WS ASKIP WS
        DREAL WS INT32 WS DREAL EXTRA,
        &dt.k_dihed, &dt.n, &dt.phi0, extra) == 3) {
    if ((s=DihedPrm_setmaxnum_term(&d, 1)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    dt.term = 1;
    dt.k_dihed *= ENERGY_INTERNAL;  /* units conversion */
    dt.phi0 *= -RADIANS;  /* reverse sign to match CHARMM dihed func spec */
    if (dt.n <= 0) {  /* periodicity for DIHEDRAL must be positive */
      DihedPrm_done(&d);
      return ERROR(ERR_INPUT);
    }
    if ((s=DihedPrm_add_term(&d, &dt)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
  }
  else {  /* unable to parse DIHEDRAL entry */
    DihedPrm_done(&d);
    return ERROR(ERR_INPUT);
  }
  if ((id=ForcePrm_add_dihedprm(p, &d)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_dihedprm(p, &d)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  DihedPrm_done(&d);
  return OK;
}


int xplor_parse_improper(NL_FILE *f, char *line, ForcePrm *p)
{
  ImprPrm m;
  char tok[TOKLEN];
  int cnt = 0, mult = 0, n = 0;
  char extra[4] = "";
  int32 id;

  if (((cnt = sscanf(line, WS TSKIP WS ANAME WS ANAME WS ANAME WS ANAME WS
            TOKEN WS INT32,
            m.atomType[0], m.atomType[1], m.atomType[2], m.atomType[3],
            tok, &mult)) == 6
        && (strcasecmp(tok, "MULT")==0 || strcasecmp(tok, "MULTIPLE")==0))
      || strlen(m.atomType[0]) == 7
      || strlen(m.atomType[1]) == 7
      || strlen(m.atomType[2]) == 7
      || strlen(m.atomType[3]) == 7) {
    return ERROR(ERR_INPUT);  /* unable to parse IMPROPER entry */
  }
  else if (4 == cnt
      && sscanf(line, WS TSKIP WS ASKIP WS ASKIP WS ASKIP WS ASKIP WS
        DREAL WS INT32 WS DREAL EXTRA,
        &m.k_impr, &n, &m.psi0, extra) == 3) {
    if (n != 0) {
      return ERROR(ERR_INPUT);  /* periodicity for IMPROPER must be set to 0 */
    }
    m.k_impr *= ENERGY_INTERNAL;  /* units conversion */
    m.psi0 *= RADIANS;
  }
  else return ERROR(ERR_INPUT);   /* unable to parse IMPROPER entry */
  if ((id=ForcePrm_add_imprprm(p, &m)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_imprprm(p, &m)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}


int xplor_parse_nonbonded(NL_FILE *f, char *line, ForcePrm *p)
{
  AtomPrm a;
  char extra[4] = "";
  int32 id;

  if (sscanf(line, WS TSKIP WS ANAME WS DREAL WS DREAL WS DREAL WS DREAL EXTRA,
        a.atomType[0], &a.emin, &a.rmin, &a.emin14, &a.rmin14, extra) != 5
      || strlen(a.atomType[0]) == 7) {
    return ERROR(ERR_INPUT);   /* unable to parse NONBONDED entry */
  }
  a.emin *= -ENERGY_INTERNAL;  /* negate to convert well depth to ener min */
  a.rmin *= SIXTH_ROOT_OF_TWO; /* rescale */
  a.emin14 *= -ENERGY_INTERNAL;
  a.rmin14 *= SIXTH_ROOT_OF_TWO;
  if ((id=ForcePrm_add_atomprm(p, &a)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_atomprm(p, &a)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}


int xplor_parse_nbfix(NL_FILE *f, char *line, ForcePrm *p)
{
  VdwpairPrm n;
  dreal a, b, a14, b14;
  char extra[4] = "";
  int32 id;

  if (sscanf(line, WS TSKIP WS ANAME WS ANAME WS
        DREAL WS DREAL WS DREAL WS DREAL EXTRA, n.atomType[0],
        n.atomType[1], &a, &b, &a14, &b14, extra) != 6
      || strlen(n.atomType[0]) == 7
      || strlen(n.atomType[1]) == 7) {
    return ERROR(ERR_INPUT);  /* unable to parse NBFIX entry */
  }
  n.emin = (-0.25 * b * b / a) * ENERGY_INTERNAL;
  n.rmin = pow(2 * a / b, 1./6);
  n.emin14 = (-0.25 * b14 * b14 / a14) * ENERGY_INTERNAL;
  n.rmin14 = pow(2 * a14 / b14, 1./6);
  //n.atomParmID[0] = -1;  /* not yet established */
  //n.atomParmID[1] = -1;
  if ((id=ForcePrm_add_vdwpairprm(p, &n)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_vdwpairprm(p, &n)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}
