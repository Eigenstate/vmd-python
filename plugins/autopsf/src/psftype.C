#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "psftype.h"
#include "periodic_table.h"

PsfType::PsfType(char* myelem) {
        atommass = get_pte_mass(get_pte_idx(myelem));
	strcpy(element,get_pte_label(atoi(eatwhite(myelem))));
	element[2]='\0';
	strcpy(atomtype,"XX");
	//	atomtype[2]='\0';
	strcat(atomtype,element);

	next=NULL;
	//fprintf(stdout,"New type created: %s %s \n",atomtype,element);
}
PsfType::PsfType(char* mytype, float mymass, char* myelem) {
        atommass = mymass;
	strcpy(atomtype,eatwhite(mytype));
	strcpy(element,myelem);
	//element[2]='\0';

	next=NULL;
	//fprintf(stdout,"New type created: %s %s \n",atomtype,element);
}

PsfType::~PsfType() {
	if (next != NULL) {
		delete next;
	}
}

bool PsfType::equals(PsfType compatom) {
	return (strcmp(compatom.type(),atomtype) == 0);
}

void PsfType::setnext(PsfType* nextatom) {
	next=nextatom;
}

bool PsfType::search(PsfType* looktype) {
  //fprintf(stdout,"Comparing type %s with %s \n",looktype->type(),atomtype);
	if (strcmp(looktype->type(),atomtype) == 0) {
	  //fprintf(stdout,"  Same! \n");
		return true;
	} else {
	  //fprintf(stdout,"  Different! \n");
		if (next == NULL) {
			return false;
		} else {
			return next->search(looktype);
		}
	}
}


char* PsfType::type() {
	return atomtype;
}

float PsfType::mass() {
	return atommass;
}

char* PsfType::elem() {
	return element;
}

void PsfType::print(FILE* outfile) {
        static int index=1;
	fprintf(outfile, "MASS %3i   %4s  %12.4f %2s\n",index,atomtype,atommass,element);
	index++;
	if (next != NULL) {
		next->print(outfile);
	}
}

//void PsfType::addatom(PsfType* newatom) {
//	newatom->setnext(head);
//	head=newatom;
//}


// Returns pointer to first non-white char, starting at s.
//   Terminating null treated as non-white char.
char *eatwhite(char *s) 
{
  while (*s && (*s == ' ' || *s == '\t')) s++;
  return(s);
}


