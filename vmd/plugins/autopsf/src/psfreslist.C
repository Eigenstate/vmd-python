#include <stdio.h>
#include "psfatom.h"
#include "psfres.h"
#include "psfreslist.h"

bool PsfResList::search(PsfAtom* lookatom) {
  //fprintf(stdout,"Searching %s:%s\n",lookatom->res(),lookatom->name());
	if (head != NULL) {
		return head->search(lookatom);
	} else {
		return false;
	}
}

void PsfResList::addres(PsfRes* newres) {
	newres->setnext(head);
	head=newres;
}

PsfRes* PsfResList::lookForRes(PsfAtom* lookatom) {
	if (head != NULL) {
		return (head->checkRes(lookatom));
	} else {
		return NULL;
	}
}

void PsfResList::print(FILE* outfile) {
	if (head != NULL) {
		head->print(outfile);
	}
}
