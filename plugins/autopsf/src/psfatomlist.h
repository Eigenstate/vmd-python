#ifndef PSFATOMLIST
#define PSFATOMLIST

#include "psfatom.h"

class PsfAtomList {
  public:
    PsfAtomList() {
      head=NULL;
    }

    ~PsfAtomList() {
      while (head != NULL) {
        PsfAtom* curr = head;
        head = head->getnext();
        delete curr;
      }
    }

    PsfAtom* find_index(int);

    void addatom(PsfAtom*);

    void print(FILE*);

  private:
    PsfAtom* head;
};

#endif
