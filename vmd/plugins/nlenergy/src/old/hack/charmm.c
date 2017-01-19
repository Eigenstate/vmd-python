/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "molfiles/charmm.h"

#undef  FEOF
#define FEOF  1


static int charmm_reader(NL_FILE *f, ForcePrm *p);


int ForcePrm_read_charmm(ForcePrm *fprm, const char *fname)
{
  NL_FILE *f;
  int s;

  if (NULL==(f = NL_fopen(fname, "r"))) return ERROR(ERR_FOPEN);
  else if ((s=charmm_reader(f, fprm)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return OK;
}


static int charmm_get_line(NL_FILE *f, char *line, int len);
static int charmm_remove_comment(char *line);
static int charmm_parse_bond(char *line, ForcePrm *p);
static int charmm_parse_angle(char *line, ForcePrm *p);
static int charmm_parse_dihedral(char *line, ForcePrm *p);
static int charmm_parse_improper(char *line, ForcePrm *p);
static int charmm_parse_nonbonded(char *line, ForcePrm *p);
static int charmm_parse_nbfix(char *line, ForcePrm *p);


#define BUFLEN   512
#define TOKLEN   16
#define TOKEN    "%15s"
#define ANAME    "%7[a-zA-Z0-9*%#+]"
#define DSKIP    "%*f"
#define DREAL    FMT_DREAL
#define INT32    FMT_INT32
#define WS       " "
#define EXTRA    "%1s"


enum {  /* sections of CHARMM parameter file */
  HEADER, BOND, ANGLE, DIHEDRAL, IMPROPER, CMAP, NONBONDED, NBFIX, HBOND, END
};


int charmm_reader(NL_FILE *f, ForcePrm *p)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  int status = OK;
  int state = HEADER;
  int s;

  while (OK==(status = charmm_get_line(f, buf, BUFLEN))) {
    if ((s=charmm_remove_comment(buf)) != OK) return ERROR(s);
    if (sscanf(buf, WS TOKEN, tok)==1) {
      switch (state) {
        case HEADER:
          if (strncasecmp(tok, "BOND", 4) == 0) {
            state = BOND;
          }
          /* expecting header for CHARMM parameter file */
          else if ('*' != tok[0]) return ERROR(ERR_INPUT);
          break;
        case BOND:
          if (strncasecmp(tok, "ANGL", 4) == 0
              || strcasecmp(tok, "THETA") == 0) {
            state = ANGLE;
          }
          else if ((s=charmm_parse_bond(buf, p)) != OK) return ERROR(s);
          break;
        case ANGLE:
          if (strncasecmp(tok, "DIHE", 4) == 0
              || strcasecmp(tok, "PHI") == 0) {
            state = DIHEDRAL;
          }
          else if ((s=charmm_parse_angle(buf, p)) != OK) return ERROR(s);
          break;
        case DIHEDRAL:
          if (strncasecmp(tok, "IMPR", 4) == 0
              || strcasecmp(tok, "IMPHI") == 0) {
            state = IMPROPER;
          }
          else if ((s=charmm_parse_dihedral(buf, p)) != OK) return ERROR(s);
          break;
        case IMPROPER:
          if (strncasecmp(tok, "NONB", 4) == 0
              || strncasecmp(tok, "NBON", 4) == 0) {
            state = NONBONDED;
          }
          else if (strcasecmp(tok, "CMAP") == 0) {
            state = CMAP;
          }
          else if ((s=charmm_parse_improper(buf, p)) != OK) return ERROR(s);
          break;
        case CMAP:
          if (strncasecmp(tok, "NBON", 4) == 0) {
            state = NONBONDED;
          }
          else { /* ignore CMAP section */ }
          break;
        case NONBONDED:
          if (strcasecmp(tok, "cutnb") == 0) { /* ignore cutnb keyword */ }
          else if (strcasecmp(tok, "NBFIX") == 0) {
            state = NBFIX;
          }
          else if (strcasecmp(tok, "HBOND") == 0) {
            state = HBOND;
          }
          else if (strcasecmp(tok, "END") == 0) {
            state = END;
          }
          else if ((s=charmm_parse_nonbonded(buf, p)) != OK) return ERROR(s);
          break;
        case NBFIX:
          if (strcasecmp(tok, "HBOND") == 0) {
            state = HBOND;
          }
          else if (strcasecmp(tok, "END") == 0) {
            state = END;
          }
          else if ((s=charmm_parse_nbfix(buf, p)) != OK) return ERROR(s);
          break;
        case HBOND:
          if (strcasecmp(tok, "END") == 0) {
            state = END;
          }
          else { /* ignore HBOND section */ }
          break;
        case END:
          /* expecting end of  CHARMM parameter file */
          if ('\0' != tok[0]) return ERROR(ERR_INPUT);
      }
    }
  }
  if (FEOF != status) return ERROR(status);
  else if (END != state) return ERROR(ERR_INPUT);  /* didn't find the end */
  return OK;
}


int charmm_get_line(NL_FILE *f, char *line, int len)
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


int charmm_remove_comment(char *line)
{
  char *s;
  /* look for end of line comment, marked by '!' */
  if (NULL!=(s = strchr(line, (int)'!'))) {
    *s = '\0';
  }
  return OK;
}


int charmm_parse_bond(char *line, ForcePrm *p)
{
  BondPrm b;
  char extra[4] = "";
  int32 id;

  if (sscanf(line, WS ANAME WS ANAME WS DREAL WS DREAL EXTRA,
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


int charmm_parse_angle(char *line, ForcePrm *p)
{
  AnglePrm a;
  int cnt;
  char extra[4] = "";
  int32 id;

  a.k_ub = 0.0;
  a.r_ub = 0.0;
  if (((cnt = sscanf(line, WS ANAME WS ANAME WS ANAME WS
            DREAL WS DREAL WS DREAL WS DREAL EXTRA,
            a.atomType[0], a.atomType[1], a.atomType[2],
            &a.k_theta, &a.theta0, &a.k_ub, &a.r_ub, extra)) != 7
        && cnt != 5)
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


int charmm_parse_dihedral(char *line, ForcePrm *p)
{
  DihedPrm d;
  DihedTerm dt;
  char extra[4] = "";
  int32 id;
  int s;

  if ((s=DihedPrm_init(&d)) != OK) return ERROR(s);
  if (sscanf(line, WS ANAME WS ANAME WS ANAME WS ANAME WS
        DREAL WS INT32 WS DREAL EXTRA,
        d.atomType[0], d.atomType[1], d.atomType[2], d.atomType[3],
        &dt.k_dihed, &dt.n, &dt.phi0, extra) != 7
      || strlen(d.atomType[0]) == 7
      || strlen(d.atomType[1]) == 7
      || strlen(d.atomType[2]) == 7
      || strlen(d.atomType[3]) == 7) {
    DihedPrm_done(&d);
    return ERROR(ERR_INPUT);  /* unable to parse DIHEDRAL entry */
  }
  else if (dt.n <= 0) {
    DihedPrm_done(&d);
    return ERROR(ERR_INPUT);  /* periodicity for DIHEDRAL must be positive */
  }
  dt.k_dihed *= ENERGY_INTERNAL;  /* units conversion */
  dt.phi0 *= RADIANS;
  dt.term = 1;
  id = ForcePrm_getid_dihedprm(p,
      d.atomType[0], d.atomType[1], d.atomType[2], d.atomType[3]);
  if (id < FAIL) {  /* something is wrong */
    DihedPrm_done(&d);
    return ERROR(id);
  }
  else if (FAIL == id) {  /* this DIHEDRAL does not exist */
    if ((s=DihedPrm_setmaxnum_term(&d, 1)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    if ((s=DihedPrm_add_term(&d, &dt)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    if ((s=ForcePrm_add_dihedprm(p, &d)) < 0) {
      DihedPrm_done(&d);
      return (s < FAIL ? ERROR(s) : ERROR(ERR_EXPECT));
    }
  }
  else if (id+1 < ForcePrm_dihedprm_array_length(p)) {  /* replace it */
    if ((s=DihedPrm_setmaxnum_term(&d, 1)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    if ((s=DihedPrm_add_term(&d, &dt)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    if ((s=ForcePrm_update_dihedprm(p, &d)) < 0) {
      DihedPrm_done(&d);
      return (s < FAIL ? ERROR(s) : ERROR(ERR_EXPECT));
    }
  }
  else {  /* append term for DIHEDRAL multiplicity */
    const DihedPrm *old = ForcePrm_dihedprm(p, id);
    if (NULL == old) {
      DihedPrm_done(&d);
      return ERROR(ERR_EXPECT);
    }
    if ((s=DihedPrm_copy(&d, old)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    dt.term = DihedPrm_term_array_length(&d) + 1;
    if ((s=DihedPrm_add_term(&d, &dt)) != OK) {
      DihedPrm_done(&d);
      return ERROR(s);
    }
    if ((s=ForcePrm_update_dihedprm(p, &d)) < 0) {
      DihedPrm_done(&d);
      return (s < FAIL ? ERROR(s) : ERROR(ERR_EXPECT));
    }
  }
  DihedPrm_done(&d);
  return OK;
}


int charmm_parse_improper(char *line, ForcePrm *p)
{
  ImprPrm m;
  int n = 0;
  char extra[4] = "";
  int32 id;

  if (sscanf(line, WS ANAME WS ANAME WS ANAME WS ANAME WS
        DREAL WS INT32 WS DREAL EXTRA,
        m.atomType[0], m.atomType[1], m.atomType[2], m.atomType[3],
        &m.k_impr, &n, &m.psi0, extra) != 7
      || strlen(m.atomType[0]) == 7
      || strlen(m.atomType[1]) == 7
      || strlen(m.atomType[2]) == 7
      || strlen(m.atomType[3]) == 7) {
    return ERROR(ERR_INPUT);  /* unable to parse IMPROPER entry */
  }
  else if (n != 0) {
    return ERROR(ERR_INPUT);  /* periodicity for IMPROPER must be set to 0 */
  }
  m.k_impr *= ENERGY_INTERNAL;  /* units conversion */
  m.psi0 *= RADIANS;
  if ((id=ForcePrm_add_imprprm(p, &m)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_imprprm(p, &m)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}


int charmm_parse_nonbonded(char *line, ForcePrm *p)
{
  AtomPrm a;
  int cnt = 0;
  char extra[4] = "";
  int32 id;

  a.emin14 = 0.0;
  a.rmin14 = 0.0;
  if (((cnt = sscanf(line, WS ANAME WS
            DSKIP WS DREAL WS DREAL WS DSKIP WS DREAL WS DREAL EXTRA,
            a.atomType[0], &a.emin, &a.rmin, &a.emin14, &a.rmin14, extra)) != 5
        && cnt != 3)
      || strlen(a.atomType[0]) == 7) {
    return ERROR(ERR_INPUT);  /* unable to parse NONBONDED entry */
  }
  a.emin *= ENERGY_INTERNAL;  /* units conversion */
  a.rmin *= 2.0;  /* CHARMM parameters given as rmin/2 and rmin14/2 */
  a.emin14 *= ENERGY_INTERNAL;
  a.rmin14 *= 2.0;
  if (3 == cnt) {  /* if not given, set scaled 1-4 params to same value */
    a.emin14 = a.emin;
    a.rmin14 = a.rmin;
  }
  if ((id=ForcePrm_add_atomprm(p, &a)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_atomprm(p, &a)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}


int charmm_parse_nbfix(char *line, ForcePrm *p)
{
  VdwpairPrm n;
  int cnt = 0;
  char extra[4] = "";
  int32 id;

  n.emin14 = 0.0;
  n.rmin14 = 0.0;
  if (((cnt = sscanf(line, WS ANAME WS ANAME WS
            DREAL WS DREAL WS DREAL WS DREAL EXTRA, n.atomType[0],
            n.atomType[1], &n.emin, &n.rmin, &n.emin14, &n.rmin14, extra)) != 6
        && cnt != 4)
      || strlen(n.atomType[0]) == 7
      || strlen(n.atomType[1]) == 7) {
    return ERROR(ERR_INPUT);  /* unable to parse NBFIX entry */
  }
  n.emin *= ENERGY_INTERNAL;  /* units conversion */
  n.emin14 *= ENERGY_INTERNAL;
  //n.atomParmID[0] = -1;  /* not yet established */
  //n.atomParmID[1] = -1;
  if (4 == cnt) {  /* if not given, set scaled 1-4 params to same value */
    n.emin14 = n.emin;
    n.rmin14 = n.rmin;
  }
  if ((id=ForcePrm_add_vdwpairprm(p, &n)) < 0) {
    if (id != FAIL) return ERROR(id);
    else if ((id=ForcePrm_update_vdwpairprm(p, &n)) < 0) return ERROR(id);
    /* updated parameter warning message? */
  }
  return OK;
}
