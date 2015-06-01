#include <stdio.h>
#include "psfatom.h"
#include "psftype.h"
#include "psftypelist.h"

bool PsfTypeList::search(PsfType* looktype) {
	if (head != NULL) {
		return head->search(looktype);
	} else {
		return false;
	}
}

void PsfTypeList::addtype(PsfType* newtype) {
	newtype->setnext(head);
	head=newtype;
}

// PsfType* PsfTypeList::lookForType(PsfType* lookatom) {
// 	if (head != NULL) {
// 		return (head->checkType(lookatom));
// 	} else {
// 		return NULL;
// 	}
// }

void PsfTypeList::print(FILE* outfile) {
	if (head != NULL) {
		head->print(outfile);
	}
}
