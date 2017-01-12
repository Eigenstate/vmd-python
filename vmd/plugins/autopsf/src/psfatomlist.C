#include <stdio.h>
#include "psfatom.h"
#include "psfatom.h"
#include "psfatomlist.h"

PsfAtom* PsfAtomList::find_index(int lookindex) {
	if (head != NULL) {
	  //fprintf(stderr,"looking for atomindex %i in %i %s\n", lookindex, head->index(), head->name());
		return head->find_index(lookindex);
	} else {
	  //fprintf(stderr,"No atoms in list\n");
		return NULL;
	}
}

void PsfAtomList::addatom(PsfAtom* newatom) {
	newatom->setnext(head);
	head=newatom;
}

// PsfAtom* PsfAtomList::lookForAtom(PsfAtom* lookatom) {
// 	if (head != NULL) {
// 		return (head->checkAtom(lookatom));
// 	} else {
// 		return NULL;
// 	}
// }

void PsfAtomList::print(FILE* outfile) {
	if (head != NULL) {
		head->print(outfile);
	}
}
