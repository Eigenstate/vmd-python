#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "psfbond.h"
#include "psfatom.h"


PsfBond::PsfBond(char* myname0, char* myname1) {
        strcpy(atomname0, myname0);
	strcpy(atomname1, myname1);
	next=NULL;
	//fprintf(stdout,"New bond created: %s %s \n",atomname0,atomname1);
}

PsfBond::~PsfBond() {
	if (next != NULL) {
		delete next;
	}
}

bool PsfBond::equals(PsfBond compbond) {
	return (strcmp(compbond.name0(),atomname0)==0 && strcmp(compbond.name1(),atomname1)==0);
}

void PsfBond::setnext(PsfBond* nextbond) {
	next=nextbond;
}

bool PsfBond::search(PsfBond* lookbond) {
  //fprintf(stdout,"Comparing bond %s-%s with %s-%s \n",lookbond->name0(),atomname0,lookbond->name1(),atomname1);
  if ((strcmp(lookbond->name0(),atomname0)==0 && strcmp(lookbond->name1(),atomname1)==0) 
       || (strcmp(lookbond->name0(),atomname1)==0 && strcmp(lookbond->name1(),atomname0)==0)) {
    //fprintf(stdout,"  Same! \n");
     return true;
  } else {
    //fprintf(stdout,"  Different! \n");
     if (next == NULL) {
        return false;
     } else {
        return next->search(lookbond);
     }
  }
}


char* PsfBond::name0() {
	return atomname0;
}

char* PsfBond::name1() {
	return atomname1;
}

void PsfBond::print(FILE* outfile) {
  
        fprintf(outfile, "BOND %4s %4s\n", atomname0, atomname1);

	if (next != NULL) {
		next->print(outfile);
	}
}



