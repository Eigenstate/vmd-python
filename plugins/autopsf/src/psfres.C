#include <stdio.h>
#include "psfatom.h"
#include "psfres.h"

void PsfRes::print(FILE* outfile) {
	char resline[30];
	sprintf(resline,"RESI %s \t 0.00 \n",resname);
	fprintf(outfile,"%s",resline);
	if (head != NULL) head->print(outfile);
	fprintf(outfile,"\n");
	if (bondhead != NULL) bondhead->print(outfile);
	fprintf(outfile,"\n");
	if (next != NULL) next->print(outfile);
}

PsfRes* PsfRes::checkRes(PsfAtom* lookatom) {
	if (strcmp(resname,lookatom->res()) == 0) {
		return this;
	} else if (next != NULL) {
		return next->checkRes(lookatom); 
	} else {
		return NULL;
	}
}

bool PsfRes::search(PsfAtom* lookatom) {
  // printf("PsfRes::search for resname '%s' in '%s'.\n", lookatom->res(),resname);
	if (strcmp(lookatom->res(),resname) == 0) {
	  //fprintf(stdout,"Found: '%s'\n",resname);
		return head->search(lookatom);
	} else {
		if (next == NULL) {
			return false;
		} else {
		  //printf("PsfRes::search for '%s' in '%s'.\n", lookatom->name(),resname);
			return next->search(lookatom);
		}
	}
}

 bool PsfRes::searchbond(PsfBond* lookbond) {
   //printf("PsfRes::search for bond resname '%s-%s' in '%s'.\n", lookbond->name0(),lookbond->name1(),resname);
 	if (bondhead != NULL) {
	  //fprintf(stdout,"Compare bond %s-%s\n",bondhead->name0(),bondhead->name1());
 		return bondhead->search(lookbond);
 	} else {
	  //fprintf(stdout,"Not Found.\n");
	        return false;
 	}
 }

int PsfRes::ispatch() {
  return patch;
}

char* PsfRes::name() {
  return resname;
}

void PsfRes::addatom(PsfAtom* newatom) {
	newatom->setnext(head);
	head=newatom;
}

void PsfRes::addbond(PsfBond* newbond) {
	newbond->setnext(bondhead);
	bondhead=newbond;
}

void PsfRes::setnext(PsfRes* nextres) {
	next=nextres;
}

