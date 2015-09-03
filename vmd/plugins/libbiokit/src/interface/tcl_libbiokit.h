#ifndef _LIBBIOKIT_TCL
#define _LIBBIOKIT_TCL

const char* seq(const char *arg1=NULL, const char *arg2=NULL, const char *arg3=NULL, const char *arg4=NULL, const char *arg5=NULL, const char *arg6=NULL);
const char* seq_usage(int argc, const char* argv[]);
const char* seq_new(int argc, const char* argv[]);
const char* seq_delete(int argc, const char* argv[]);
const char* seq_reset(int argc, const char* argv[]);
const char* seq_cleanup(int argc, const char* argv[]);
const char* seq_type(int argc, const char* argv[]);
const char* seq_name(int argc, const char* argv[]);
const char* seq_length(int argc, const char* argv[]);
const char* seq_get(int argc, const char* argv[]);
const char* seq_set(int argc, const char* argv[]);
const char* seq_position_of_residue(int argc, const char* argv[]);
const char* seq_residue_at_position(int argc, const char* argv[]);
const char* seq_get_color(int argc, const char* argv[]);
const char* seq_set_color(int argc, const char* argv[]);

int countListItems(const char* list);
char* getNextListItem(char** list);
Alphabet* determineAlphabet(const char* sequenceData);
int parsePositiveInteger(const char* string, int maxValue);
void addSequenceData(const char* sequenceData, AlignedSequence* sequence, ShortIntList* state);

#endif
