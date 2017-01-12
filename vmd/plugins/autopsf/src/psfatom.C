#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "psfatom.h"
#include "periodic_table.h"

PsfAtom::PsfAtom(char* myname, char* myres) {
	strcpy(atomname,eatwhite(trimright(myname,5)));
	strcpy(resname,eatwhite(trimright(myres,4)));
	strcpy(element,"XX");
	element[2] = '\0';
	segname[0]='\0';
	pdbindex  = -1;
	residueid = -1;
	alias();

	next=NULL;
	//fprintf(stdout,"New atom created: %s:%s.\n",resname,atomname);
}

PsfAtom::PsfAtom(char* myindexstr, char* myname, char* myres, char* myelem, char* myresid, char* myseg) {
        char indexstr[7], residstr[6];
	char atomnum[3];
	strcpy(indexstr,eatwhite(trimright(myindexstr,6)));
	pdbindex = atoi(indexstr);
	strcpy(residstr,eatwhite(trimright(myresid,5)));
	residueid = atoi(residstr);
	strcpy(atomname,eatwhite(trimright(myname,5)));
	strcpy(resname,eatwhite(trimright(myres,4)));
	strcpy(segname,eatwhite(trimright(myseg,4)));
	strcpy(atomnum,eatwhite(trimright(myelem,2)));
	strcpy(element,get_pte_label(atoi(atomnum)));
	//printf("pdbindex %i %s; %s; %s; %s; %i\n", pdbindex, myindexstr, atomname, resname, element, residueid);
	alias();

	next=NULL;
	//fprintf(stdout,"New atom created: %s:%s.\n",resname,atomname);
}


bool PsfAtom::equals(PsfAtom compatom) {
	return (strcmp(compatom.name(),atomname) == 0);
}

void PsfAtom::setnext(PsfAtom* nextatom) {
	next=nextatom;
}

bool PsfAtom::search(PsfAtom* lookatom) {
  //fprintf(stdout,"Comparing %s:%s with %s:%s. \n",lookatom->res(),lookatom->name(),resname,atomname);
	if (strcmp(lookatom->name(),atomname) == 0) {
	  //fprintf(stdout,"  Same! \n");
		return true;
	} else {
	  //fprintf(stdout,"  Different! \n");
		if (next == NULL) {
			return false;
		} else {
			return next->search(lookatom);
		}
	}
}

PsfAtom* PsfAtom::find_index(int lookindex) {
  // fprintf(stdout,"Comparing atom %i with %i. \n",lookindex,pdbindex);
	if (lookindex==pdbindex) {
	  //fprintf(stdout,"  Same! \n");
		return this;
	} else {
	  //fprintf(stdout,"  Different! \n");
		if (next == NULL) {
			return NULL;
		} else {
			return next->find_index(lookindex);
		}
	}
}


int PsfAtom::index() {
	return pdbindex;
}

int PsfAtom::resid() {
	return residueid;
}

char* PsfAtom::name() {
	return atomname;
}

char* PsfAtom::res() {
	return resname;
}

void PsfAtom::alias() {
  if (!strcmp(resname,"HIS")) { strcpy(resname,"HSD");  return; }
  if (!strcmp(resname,"HOH")) { 
    strcpy(resname,"TIP3"); 
    if (!strcmp(atomname,"O")) strcpy(atomname,"OH2");
    return; 
  }
  if (!strcmp(resname,"G"))   { strcpy(resname,"GUA"); aliasnucleic(); return; }
  if (!strcmp(resname,"C"))   { strcpy(resname,"CYT"); aliasnucleic(); return; }
  if (!strcmp(resname,"A"))   { strcpy(resname,"ADE"); aliasnucleic(); return; }
  if (!strcmp(resname,"T"))   { strcpy(resname,"THY"); aliasnucleic(); return; }
  if (!strcmp(resname,"U"))   { strcpy(resname,"URA"); aliasnucleic(); return; }

  if (!strcmp(resname,"ILE") && !strcmp(atomname,"CD1")) { 
    strcpy(atomname,"CD"); 
    return; 
  }
  if (!strcmp(resname,"SER") && !strcmp(atomname,"HG")) { 
    strcpy(atomname,"HG1"); 
    return; 
  }
  if (!strcmp(resname,"K")) { 
    strcpy(resname,"POT"); 
    if (!strcmp(atomname,"K")) strcpy(atomname,"POT");
    return; 
  }
  if (!strcmp(resname,"ICL")) { 
    strcpy(resname,"CLA"); 
    if (!strcmp(atomname,"CL")) strcpy(atomname,"CLA");
    return; 
  }
  if (!strcmp(resname,"INA")) { 
    strcpy(resname,"SOD"); 
    if (!strcmp(atomname,"NA")) strcpy(atomname,"SOD");
    return; 
  }
  if (!strcmp(resname,"CA")) { 
    strcpy(resname,"CAL"); 
    if (!strcmp(atomname,"CA")) strcpy(atomname,"CAL");
    return; 
  }
  if (!strcmp(resname,"HEM")) { 
    strcpy(resname,"HEME"); 
    if (!strcmp(atomname,"N A")) strcpy(atomname,"NA");
    if (!strcmp(atomname,"N B")) strcpy(atomname,"NB");
    if (!strcmp(atomname,"N C")) strcpy(atomname,"NC");
    if (!strcmp(atomname,"N D")) strcpy(atomname,"ND");
    return; 
  }
  if (!strcmp(resname,"LYS")) {
    if (!strcmp(atomname,"1HZ")) { strcpy(atomname,"HZ1"); return; }
    if (!strcmp(atomname,"2HZ")) { strcpy(atomname,"HZ2"); return; }
    if (!strcmp(atomname,"3HZ")) { strcpy(atomname,"HZ3"); return; }
    return; 
  }
  if (!strcmp(resname,"ARG")) {
    if (!strcmp(atomname,"1HH1")) { strcpy(atomname,"HH11"); return; }
    if (!strcmp(atomname,"2HH1")) { strcpy(atomname,"HH12"); return; }
    if (!strcmp(atomname,"1HH2")) { strcpy(atomname,"HH21"); return; }
    if (!strcmp(atomname,"2HH2")) { strcpy(atomname,"HH22"); return; }
    return; 
  }
  if (!strcmp(resname,"ASN")) {
    if (!strcmp(atomname,"1HD2")) { strcpy(atomname,"HD21"); return; }
    if (!strcmp(atomname,"2HD2")) { strcpy(atomname,"HD22"); return; }
    return; 
  }

  return;
}

void PsfAtom::aliasnucleic() {
  if (!strcmp(atomname,"O5*")) { strcpy(atomname,"O5'"); return; }
  if (!strcmp(atomname,"C5*")) { strcpy(atomname,"C5'"); return; }
  if (!strcmp(atomname,"O4*")) { strcpy(atomname,"O4'"); return; }
  if (!strcmp(atomname,"C4*")) { strcpy(atomname,"C4'"); return; }
  if (!strcmp(atomname,"O3*")) { strcpy(atomname,"O3'"); return; }
  if (!strcmp(atomname,"C3*")) { strcpy(atomname,"C3'"); return; }
  if (!strcmp(atomname,"O2*")) { strcpy(atomname,"O2'"); return; }
  if (!strcmp(atomname,"C2*")) { strcpy(atomname,"C2'"); return; }
  if (!strcmp(atomname,"C1*")) { strcpy(atomname,"C1'"); return; }
  return;
}

void PsfAtom::print(FILE* outfile) {
	fprintf(outfile, "ATOM %-4s XX%-2s     0.00\n",atomname, eatwhite(element));
	if (next != NULL) {
		next->print(outfile);
	}
}

PsfAtom* PsfAtom::getnext() {
  return this->next;
}
