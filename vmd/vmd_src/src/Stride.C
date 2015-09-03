/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: Stride.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.34 $       $Date: 2010/12/16 04:08:41 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Stride interface class.
 ***************************************************************************/

#include <stdio.h>      // for tmpfile
#include "DrawMolecule.h"
#include "Timestep.h"
#include "Residue.h"
#include "Inform.h"
#include "Stride.h"
#include "utilities.h"  // for vmd_delete_file
#include "VMDDir.h"

// Write a pdb of the given timestep.
// Write the uniq_resid of the residues written to the stride input file
// into the residues argument.
static int write_stride_record( DrawMolecule *mol, 
                                const char *inputfilename,
                                ResizeArray<int>& residues ) {
  const float *pos = mol->current()->pos;
  char name[6], resname[5];
 
  residues.clear();

  FILE *inputfile = fopen(inputfilename,"w");
  if (!inputfile) {
    msgErr << "Unable to open input file '" << inputfilename 
           << "' for writing." << sendmsg;
    return 1;
  }

  int prev = -1; // previous uniq_resid
  int atomcount = 0;

  for (int i=0; i < mol->nAtoms; i++) {
    MolAtom *atom = mol->atom(i);
    // skip if this atom isn't part of protein
    if (atom->residueType != RESPROTEIN)
      continue;
    const float *loc=pos + 3*i;
    strncpy(name, (mol->atomNames).name(atom->nameindex), 4);
    name[4]='\0';
    strncpy(resname,(mol->resNames).name(atom->resnameindex),4);  
    resname[4] = '\0';

    if (atom->uniq_resid != prev) {
      prev = atom->uniq_resid;
      residues.append(prev);
    }

    if (fprintf(inputfile,
      "ATOM  %5d %-4s %-4s %4d    %8.3f%8.3f%8.3f\n",
      ++atomcount,name,resname,residues.num()-1,loc[0],loc[1],loc[2]) < 0) {
      msgErr << "Error writing line in Stride input file for atom " << i 
             << sendmsg;
      return 1;
    }
  }
  fclose(inputfile);
  return 0;
}

/* Here's the sprintf command stride uses on output in report.c:

      sprintf(Tmp,"ASG  %3s %c %4s %4d    %c   %11s   %7.2f   %7.2f   %7.1f",
        p->ResType,SpaceToDash(Chain[Cn]->Id),p->PDB_ResNumb,i+1,
        p->Prop->Asn,Translate(p->Prop->Asn),p->Prop->Phi,
	p->Prop->Psi,p->Prop->Solv);

   Here's a typical line:
ASG  THR -    9    9    B        Bridge   -113.85    157.34      21.9      ~~~~

*/

const int BUF_LEN = 90;
const char sstypes[] = "HGIEBbT";
static int read_stride_record( DrawMolecule *mol, 
                               const char *stridefile,
                               const ResizeArray<int> &residues ) { 
  FILE *in = fopen(stridefile, "rt");
  if (!in) {
    msgErr << "Unable to find Stride output file: "
           << stridefile << sendmsg;
    return 1;
  }
  char buf[BUF_LEN], resname[4], chain[1], uniq_residstr[5], ss[1];
  const char *stype;
  const char *sstypes_start = &sstypes[0];
  int resid[1];
  while (fgets(buf, BUF_LEN, in)) {
    if (strncmp(buf, "ASG", 3)) continue;
    sscanf(buf,"ASG  %3s %c %4s %4d    %c",
      resname, chain, uniq_residstr, resid, ss);
    int index = atoi(uniq_residstr);
    if (index < 0 || index >= residues.num()) {
      msgErr << "invalid resid found in stride output file!" << sendmsg;
      msgErr << "Error found in the following line: " << sendmsg;
      msgErr << buf << sendmsg;
      fclose(in);
      return -1;
    }
    int uniq_resid = residues[index];

    stype = strchr(sstypes,ss[0]);

    if (stype == NULL) {
      mol->residueList[uniq_resid]->sstruct = SS_COIL;
      continue;
    }
    switch ((int)(stype - sstypes_start)) {
      case 0: // H
	mol->residueList[uniq_resid]->sstruct = SS_HELIX_ALPHA;
	break;
      case 1: // G
	mol->residueList[uniq_resid]->sstruct = SS_HELIX_3_10;
        break; 
      case 2: // I
	mol->residueList[uniq_resid]->sstruct = SS_HELIX_PI;
	break;
      case 3: // E
	mol->residueList[uniq_resid]->sstruct = SS_BETA;
	break;
      case 4: // B
      case 5: // b
	mol->residueList[uniq_resid]->sstruct = SS_BRIDGE;
	break;
      case 6: // T
        mol->residueList[uniq_resid]->sstruct = SS_TURN;
	break;
      default:
	msgErr << "Internal error in read_stride_record\n" << sendmsg;
	mol->residueList[uniq_resid]->sstruct = SS_COIL;
    }
  }
  fclose(in);
  return 0;
}  

int ss_from_stride(DrawMolecule *mol) {
  int rc = 0;
  char *stridebin   = getenv("STRIDE_BIN");
  char *infilename; 
  char *outfilename;
  
  if (!stridebin) {
    msgErr << "No STRIDE binary found; please set STRIDE_BIN environment variable" << sendmsg;
    msgErr << "to the location of the STRIDE binary." << sendmsg;
    return 1;
  }

  // check to see if the executable exists
  if (!vmd_file_is_executable(stridebin)) {
    msgErr << "STRIDE binary " << stridebin << " cannot be run; check permissions." << sendmsg;
    return 1;
  }

#if defined(ARCH_MACOSXX86) || defined(ARCH_MACOSXX86_64)
  infilename = tmpnam(NULL);
  outfilename = tmpnam(NULL);
  char *tmpstr;
  tmpstr  = (char *) malloc(strlen(infilename));
  strcpy(tmpstr, infilename);
  infilename = tmpstr;
  tmpstr = (char *) malloc(strlen(outfilename));
  strcpy(tmpstr, outfilename);
  outfilename = tmpstr;
#else
  infilename  = tempnam(NULL, NULL);
  outfilename = tempnam(NULL, NULL);
#endif
  if (infilename == NULL || outfilename == NULL) {
    msgErr << "Unable to create temporary files for STRIDE." << sendmsg;
    return 1;
  }

  char *stridecall = new char[strlen(stridebin)
                              +strlen(infilename)
                              +strlen(outfilename)
                              +16];

  sprintf(stridecall,"\"%s\" %s -f%s", stridebin, infilename, outfilename);

  ResizeArray<int> residues;

  if (write_stride_record(mol,infilename,residues)) {
    msgErr << "Stride::write_stride_record: unable "
           << "to write input file for Stride\n" << sendmsg;
    rc = 1;
  }

  if (!rc) {
    vmd_system(stridecall);

    if (read_stride_record(mol,outfilename,residues)) {
      msgErr << "Stride::read_stride_record: unable "
             << "to read output file from Stride\n" << sendmsg;
      rc = 1;
    }
  }

  delete [] stridecall;
  vmd_delete_file(outfilename);
  vmd_delete_file(infilename);
  free(outfilename);
  free(infilename);

  return rc;
} 
