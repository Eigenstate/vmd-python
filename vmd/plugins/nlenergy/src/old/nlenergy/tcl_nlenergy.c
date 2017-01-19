#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tcl.h"
#include "nlenergy/nlenergy.h"

#undef  NLMSG
#define NLMSG NL_printf

#ifdef WIN32
#define strcasecmp  stricmp
#define strncasecmp strnicmp
#endif

/* ??? */
#if defined(NAMD_TCL) || ! defined(NAMD_VERSION)


static int help(NLEnergy *nldata, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[], int status) {
  Tcl_SetResult(interp, (char *)
"usage: <nlenergyobject> <command> [args...]\n"
"\n"
"  help\n"
"  delete\n"
"  get\n"
"  set\n"
"  add\n"
"  remove\n"
"  missing\n"
"  read\n"
"  write\n"
"  energy\n"
"\n", TCL_STATIC);
  return (OK==status ? TCL_OK : TCL_ERROR);
}


/* "nlenergy%d" molecule is to be deleted */
static void remove_tcl_nlenergy(ClientData data) {
  NLEnergy *nldata = (NLEnergy *) data;

  NLMSG("NLEnergy> Removing nlenergy%d\n", nldata->idnum);

  NLEnergy_done(nldata);
  NL_free(nldata);
  TEXT("remove ok");
}


/* process "nlenergy%d commands ..." */
static int access_tcl_nlenergy(ClientData data, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[])
{
  NLEnergy *nldata = (NLEnergy *) data;
  int idnum = (nldata ? nldata->idnum : -1);
  int s;

  INT(idnum);
  //NLMSG("NLEnergy> Hello from nlenergy%d\n", idnum);
  if (idnum < 0) return NLERR(ERR_EXPECT);

  if (objc >= 2) {
    const char *cmd = Tcl_GetString(objv[1]);
    if (strcmp(cmd,"help")==0) {
      return help(nldata, interp, objc-2, objv+2, OK);
    }
    if (2==objc && strcmp(cmd,"delete")==0) {
      char script[64];
#if 0
      if (nldata->aselname != NULL) {  /* also delete atom selection */
        snprintf(script, sizeof(script), "%s delete", nldata->aselname);
        Tcl_EvalEx(interp, script, -1, 0);
      }
#endif
      snprintf(script, sizeof(script), "unset upproc_var_%s",
          Tcl_GetString(objv[0]));
      return Tcl_EvalEx(interp, script, -1, 0);
    }
    if (strcmp(cmd,"get")==0) {
      s = NLEnergy_parse_get(nldata, interp, objc-2, objv+2);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"set")==0) {
      s = NLEnergy_parse_set(nldata, interp, objc-2, objv+2);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"add")==0) {
      s = NLEnergy_parse_add(nldata, interp, objc-2, objv+2);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"remove")==0) {
      s = NLEnergy_parse_remove(nldata, interp, objc-2, objv+2);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"missing")==0) {
      s = NLEnergy_parse_missing(nldata, interp, objc-2, objv+2);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"read")==0) {
      s = NLEnergy_parse_read(nldata, interp, objc-2, objv+2);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"write")==0) {
      s = NLEnergy_parse_write(nldata, interp, objc-2, objv+2);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"energy")==0) {
      s = NLEnergy_parse_eval(nldata, interp, objc-2, objv+2, EVAL_ENERGY);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"force")==0) {
      s = NLEnergy_parse_eval(nldata, interp, objc-2, objv+2, EVAL_FORCE);
      return (OK==s ? TCL_OK : NLERR(s));
    }
    if (strcmp(cmd,"minimize")==0) {
      s = NLEnergy_parse_eval(nldata, interp, objc-2, objv+2, EVAL_MINIMIZE);
      return (OK==s ? TCL_OK : NLERR(s));
    }
  }

  return help(nldata, interp, 0, NULL, ERR_EXPECT);
}


/* create an instance of NLEnergy with handle "nlenergy%d" */
static int tcl_nlenergy(ClientData nodata, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[])
{
  NLEnergy *nldata;
  int *pcount;
  Tcl_Obj *objmolid;
  const char *aselname;
  char name[32];
  char script[64];
  int molid;
  int s;  /* error status */

  /* return a list of all the undeleted nlenergy molecules */
  INT(objc);
  STR(objc >= 1 ? Tcl_GetString(objv[0]) : "");
  STR(objc >= 2 ? Tcl_GetString(objv[1]) : "");
  STR(objc >= 3 ? Tcl_GetString(objv[2]) : "");
  if (2==objc && strcmp(Tcl_GetString(objv[1]), "list") == 0) {
    snprintf(script, sizeof(script), "info commands nlenergy?*");
    return Tcl_EvalEx(interp, script, -1, 0);
  }
  else if (objc != 2) {
    Tcl_SetResult(interp, (char *) "usage: nlenergy <atomsel>\n", TCL_STATIC);
    return NLERR(ERR_EXPECT);
  }

  /* get molid from atom selection */
  aselname = Tcl_GetString(objv[1]);  /* point to the atom selection name */
  STR(aselname);
  snprintf(script, sizeof(script), "%s molid", aselname);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0) ||
      NULL==(objmolid = Tcl_GetObjResult(interp)) ||
      TCL_OK != Tcl_GetIntFromObj(interp, objmolid, &molid)) {
    Tcl_SetResult(interp,
        (char *) "can't determine molid from atom selection\n"
        "usage: nlenergy <atomsel>\n", TCL_STATIC);
    return NLERR(ERR_EXPECT);
  }

  /* get NLEnergy object counter */
  pcount = (int *) Tcl_GetAssocData(interp, (char *) "NLEnergyCount", NULL);
  if (NULL==pcount) return NLERR(ERR_EXPECT);
  INT(*pcount);

  /* allocate a new NLEnergy object for this molid */
  nldata = (NLEnergy *) NL_malloc(sizeof(NLEnergy));
  if (NULL==nldata) return NLERR(ERR_MEMALLOC);

  /* initialize NLEnergy object */
  if ((s=NLEnergy_init(nldata)) != OK) {
    NL_free(nldata);
    return NLERR(s);
  }

#if 0
  /* create an atom selection */
  snprintf(script, sizeof(script), "atomselect %d all", molid);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0)) return ERROR(ERR_EXPECT);
  if (NULL==(asel = Tcl_GetObjResult(interp))) return ERROR(ERR_EXPECT);
  Tcl_IncrRefCount(asel);  /* keep asel object for later use */
  aselname = Tcl_GetStringFromObj(asel, NULL);
#endif

  /* setup NLEnergy object */
  if ((s=NLEnergy_setup(nldata, interp, *pcount, molid, aselname)) != OK) {
    //Tcl_DecrRefCount(asel);  /* no longer using asel object */
    NLEnergy_done(nldata);
    NL_free(nldata);
    return NLERR(s);
  }
  //Tcl_DecrRefCount(asel);  /* no longer using asel object */

  /* create name and increment count */
  snprintf(name, sizeof(name), "nlenergy%d", *pcount);
  NLMSG("NLEnergy> Creating %s\n", name);
  (*pcount)++;

  /* make new command from this name */
  if (NULL==Tcl_CreateObjCommand(interp, name, access_tcl_nlenergy,
      (ClientData) nldata, (Tcl_CmdDeleteProc *) remove_tcl_nlenergy)) {
    NLEnergy_done(nldata);
    NL_free(nldata);
    return NLERR(ERR_EXPECT);
  }
  
  /* here I need to change the context ... */
  snprintf(script, sizeof(script), "upproc 0 %s", name);
  if (TCL_OK != Tcl_EvalEx(interp, script, -1, 0)) {
    return NLERR(ERR_EXPECT);
  }

  /* return string name of new function as result */
  Tcl_SetResult(interp, name, TCL_VOLATILE);
#if 0
  /* append the new function name and return it */
  Tcl_AppendElement(interp, name);
#endif

  return TCL_OK;
}



/*
 * callback for when the interpreter gets deleted
 * this deletes the counter allocated in Nlenergy_Init()
 */
static void Nlenergy_Delete(ClientData data, Tcl_Interp *interp) {
  NL_free(data);
}



/*
 * ??? Where should NAMDLITETCLDLL_EXPORTS get set?
 */
#if defined(NAMDLITETCLDLL_EXPORTS) && defined(_WIN32)
#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLEXPORT

#define WIN32_LEAN_AND_MEAN /* Exclude rarely used stuff from Windows headers */

#undef TRUE  /* TRUE and FALSE already defined */
#undef FALSE

#include <windows.h>

BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved )
{
  return TRUE;
}


EXTERN int Nlenergy_Init(Tcl_Interp *interp) {

#else

int Nlenergy_Init(Tcl_Interp *interp) {

#endif

  int *count = NULL;

  NLMSG("NLEnergy> initializing...\n");

  /* create the main procedure */
  if (NULL==Tcl_CreateObjCommand(interp, "nlenergy", tcl_nlenergy,
      (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL)) {
    return NLERR(ERR_EXPECT);
  }

  /* count number of instances created */
  count = (int *) NL_malloc(sizeof(int));
  if (NULL==count) return NLERR(ERR_MEMALLOC);
  *count = 0;

  /* attach counter to TCL interpreter using name "NLEnergyCount" */
  Tcl_SetAssocData(interp, (char *) "NLEnergyCount", Nlenergy_Delete, count);

  /* providing version "1.0" of NLEnergy */
  if (TCL_ERROR==Tcl_PkgProvide(interp, "nlenergy", "1.0")) {
    return NLERR(ERR_EXPECT);
  }

  return TCL_OK;
}


#endif
