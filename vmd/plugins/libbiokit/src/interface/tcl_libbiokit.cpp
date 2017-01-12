#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "PointerList.h"
#include "ShortIntList.h"
#include "symbol.h"
#include "alphabet.h"
#include "alphabetBuilder.h"
#include "sequence.h"
#include "alignedSequence.h"
#include "tcl_libbiokit.h"

#ifdef WIN32
#define snprintf _snprintf
#endif

// -------------------------------------------------------------------
// Variables for returning output to TCL.
// changed the global var declarations to static to at least make them specific
// to this file.
static int maxOutputLength = 8192;
static char *output = 0;

// Variables for the sequence store.
static PointerList* sequences = new PointerList();
static PointerList* colorings = new PointerList();

static Alphabet* proteinAlphabet = AlphabetBuilder::createProteinAlphabet();
static Alphabet* rnaAlphabet = AlphabetBuilder::createRnaAlphabet();
static Alphabet* dnaAlphabet = AlphabetBuilder::createDnaAlphabet();

void confirmOutputLength(const int requiredLength)
{
   if (requiredLength <= maxOutputLength)
   {
      return;
   }

   free(output);
   maxOutputLength = requiredLength;
   output = (char*) malloc(maxOutputLength * sizeof(char));
}

// -------------------------------------------------------------------
const char* seq(const char *arg1, const char *arg2, const char *arg3, const char *arg4, const char *arg5, const char *arg6)
{

    if (!output)
    {
       output = (char*) malloc(maxOutputLength * sizeof(char));
    }

    // Figure out how many args we got.
    int argc = 0;
    const char* argv[6];
//    printf ("seq call: ");
    if (arg1 != NULL) {
//        printf ("%d: %s, ", argc, arg1);
        argv[argc++] = arg1;
    }
    if (arg2 != NULL) {
//        printf ("%d: %s, ", argc, arg2);
        argv[argc++] = arg2;
    }
    if (arg3 != NULL) {
//        printf ("%d: %s, ", argc, arg3);
        argv[argc++] = arg3;
    }
    if (arg4 != NULL) {
//        printf ("%d: %s, ", argc, arg4);
        argv[argc++] = arg4;
    }
    if (arg5 != NULL) {
//        printf ("%d: %s, ", argc, arg5);
        argv[argc++] = arg5;
    }
    if (arg6 != NULL) {
//        printf ("%d: %s, ", argc, arg6);
        argv[argc++] = arg6;
    }
//    printf ("\n"); 

    // Make sure we got at least one argument.
    if (argc == 0)
    {
        return seq_usage(argc, argv);
    }


    // Figure out which function to call.
    switch (argv[0][0])
    {
    case 'd':
        return seq_delete(argc, argv);
    case 'g':
        if (argc >= 2 && strncmp(argv[1], "col", 3) == 0)
            return seq_get_color(argc, argv);
        return seq_get(argc, argv);
    case 'l':
        return seq_length(argc, argv);
    case 'n':
        if (strncmp(argv[0], "name", 4) == 0)
            return seq_name(argc, argv);
        return seq_new(argc, argv);
    case 'p':
        return seq_position_of_residue(argc, argv);
    case 'r':
        if (strncmp(argv[0], "rese", 4) == 0)
            return seq_reset(argc, argv);
        else if (strncmp(argv[0], "resA", 4) == 0)
            return seq_residue_at_position(argc, argv);
        else
            return seq_usage(argc, argv);
    case 's':
        if (argc >= 2 && strncmp(argv[1], "col", 3) == 0)
            return seq_set_color(argc, argv);
        return seq_set(argc, argv);
    case 't':
        return seq_type(argc, argv);
    case 'c':
        return seq_cleanup(argc, argv);
    default:
        return seq_usage(argc, argv);
    }
}

// -------------------------------------------------------------------
const char* seq_usage(int argc, const char* argv[])
{
    // Max char at 92.
    printf("Unknown command: <seq ");
    for (int i=0; i< argc; i++)
    {
       printf("%s ", argv[i]);
    }
    printf(">\n");

    printf("usage: seq <command> [args...]\n");
    printf("\n");
    printf("Sequences and Data:\n");
    printf("  new <sequence> [auto*|protein|rna|dna]  -- create a new sequence\n");
    printf("  delete <seqid>                          -- delete a sequence\n");
    printf("  reset                                   -- reset the sequence store\n");
    printf("  cleanup                                 -- DON'T USE (internal)\n");
    printf("\n");
    printf("Sequence Properties:\n");
    printf("  type <seqid>                            -- get the type of a sequence\n");
    printf("  length <seqid>                          -- get the length of a sequence\n");
    printf("  get <seqid> [<start> <end>]             -- get the sequence; optionally from\n");
    printf("                                             positions start to end, inclusive\n");
    printf("  set <seqid> <sequence>                  -- set the sequence\n");
    printf("  resAt <seqid> <position>                -- get the residue at a position\n");
    printf("  posOf <seqid> <residue>                 -- get the position of a residue\n");
    printf("  get color <seqid> <position>            -- get the color of a position\n");
    printf("  set color <seqid> <position> <color>    -- set the color of a position\n");
    printf("  set color <seqid> <start> <end> <color> -- set the color of a position range\n");
    printf("  name get <seqid>                        -- get the name of a sequence\n");
    printf("  name set <seqid> <name>                 -- set the name of a sequence\n");
    return "";
}

/* -------------------------------------------------------------------- */
const char* seq_cleanup(int argc, const char* argv[])
{
   seq_reset(argc, argv);
   delete proteinAlphabet; proteinAlphabet = 0;
   delete rnaAlphabet; rnaAlphabet = 0;
   delete dnaAlphabet; dnaAlphabet = 0;
   delete sequences; sequences = 0;
   delete colorings; colorings = 0;
   return "";
}

/* -------------------------------------------------------------------- */
const char* seq_new(int argc, const char* argv[])
{
    if (argc >= 2 && argc <= 3)
    {
        // Parse the arguments.
        const char* sequenceData = argv[1];
//printf ("tcl_libbiokit.cpp.seq_new. seqData is '%s'. id will be %d\n", sequenceData, sequences->getSize());
        Alphabet* alphabet = NULL;
        if (argc == 2)
        {
            alphabet = determineAlphabet(sequenceData);
        }
        else if (argc == 3)
        {
            if (strcmp(argv[2], "auto") == 0)
                alphabet = determineAlphabet(sequenceData);
            else if (strcmp(argv[2], "protein") == 0)
                alphabet = proteinAlphabet;
            else if (strcmp(argv[2], "rna") == 0)
                alphabet = rnaAlphabet;
            else if (strcmp(argv[2], "dna") == 0)
                alphabet = dnaAlphabet;
            else
                return seq_usage(argc, argv);
        }

        
        // Create the sequence.
        AlignedSequence* sequence = new AlignedSequence(alphabet);
        ShortIntList* colors = new ShortIntList();
//        colors->initialize(sequence->getSize(), 0);
        addSequenceData(sequenceData, sequence, colors);
        
        // Add the sequence to the store and get its handle.
        int sequenceID = sequences->getSize();
        sequences->add(sequence);
        colorings->add(colors);
        
//printf ("tcl_libbiokit.cpp.seq_new. id: %d, size: %d, colors: %d\n", sequenceID, sequence->getSize(), colors->getSize());
        // Return the sequence id.
        sprintf(output, "%d", sequenceID);
        return output;
    }
    
    return seq_usage(argc, argv);
}

/* -------------------------------------------------------------------- */
const char* seq_delete(int argc, const char* argv[])
{
    if (argc == 2)
    {
        int sequenceID = parsePositiveInteger(argv[1], sequences->getSize()-1);
        if (sequenceID != -1)
        {
            AlignedSequence* sequence = (AlignedSequence*)sequences->get(sequenceID);
            if (sequence != NULL)
            {
                delete sequence;
                sequences->set(sequenceID, NULL);
                delete (ShortIntList*)colorings->get(sequenceID);
                colorings->set(sequenceID, NULL);
                return "";
            }
        }
        
        printf("[seq delete] Invalid sequence id: %s\n", argv[1]);
        return "";
    }
    
    return seq_usage(argc, argv);
}

/* -------------------------------------------------------------------- */
const char* seq_reset(int argc, const char* argv[])
{
   int i=0;
   // Reset the sequence store.
   if (sequences)
   {
      for (i=0; i<sequences->getSize(); i++)
      {
         AlignedSequence* sequence = (AlignedSequence*)sequences->get(i);
         if (sequence != NULL)
         {
            delete sequence;
            sequences->set(i, NULL);
            delete (ShortIntList*)colorings->get(i);
            colorings->set(i, NULL);
         }
      }
      delete sequences;
   }
   sequences = new PointerList();
   delete colorings;
   colorings = new PointerList();
    
   return "";
}

/* -------------------------------------------------------------------- */
const char* seq_type(int argc, const char* argv[])
{
    if (argc == 2)
    {
        int sequenceID = parsePositiveInteger(argv[1], sequences->getSize()-1);
        if (sequenceID != -1)
        {
            AlignedSequence* sequence = (AlignedSequence*)sequences->get(sequenceID);
            if (sequence != NULL)
            {
                if (sequence->getAlphabet() == proteinAlphabet)
                    return "protein";
                else if (sequence->getAlphabet() == rnaAlphabet)
                    return "rna";
                else if (sequence->getAlphabet() == dnaAlphabet)
                    return "dna";
            }
        }
        
        printf("[seq type] Invalid sequence id: %s\n", argv[1]);
        return "";
    }
    
    return seq_usage(argc, argv);
}

/* -------------------------------------------------------------------- */
const char* seq_length(int argc, const char* argv[])
{
    if (argc == 2)
    {
        int sequenceID = parsePositiveInteger(argv[1], sequences->getSize()-1);
        if (sequenceID != -1)
        {
            AlignedSequence* sequence = (AlignedSequence*)sequences->get(sequenceID);
            if (sequence != NULL)
            {
                sprintf(output, "%d", sequence->getSize());
                return output;
            }
        }
        
        printf("[seq length] Invalid sequence id: %s\n", argv[1]);
        return "";
    }
    
    return seq_usage(argc, argv);
}

/* -------------------------------------------------------------------- */
const char* seq_get(int argc, const char* argv[])
{
    if (argc == 2 || argc == 4)
    {
        // Parse the arguments.
        AlignedSequence* sequence = NULL;
        int sequenceID = parsePositiveInteger(argv[1], sequences->getSize()-1);
        if (sequenceID != -1) sequence = (AlignedSequence*)sequences->get(sequenceID);
        if (sequence == NULL)
        {
            printf("[seq get] Invalid sequence id: %s\n", argv[1]);
            return "";
        }
        int start = 0;
        int end = sequence->getSize()-1;
        if (argc == 4)
        {
            start = parsePositiveInteger(argv[2], sequence->getSize()-1);
            if (strcmp(argv[3], "end") == 0)
                end = sequence->getSize()-1;
            else
                end = parsePositiveInteger(argv[3], sequence->getSize()-1);
            if (start == -1 || end == -1 || start > end)
            {
                fprintf(stderr, "[seq get] Invalid range for seq %s: %s to %s\n",
                                             argv[1], argv[2], argv[3]);
                return "";
            }
        }
      
        // make sure the output array is actually long enough
        // to hold this sequence
        confirmOutputLength( (end-start)*2 );

        // Get the sequence.
        int index=0;
        int i;
        for (i=start; i <= end; i++, index+=2)
        {
            output[index] = sequence->get(i).getOne();
            output[index+1] = ' ';
        }
        if (index > 0)
            output[index-1] = '\0';
        else
            output[0] = '\0';
        return output;
    }
    
    return seq_usage(argc, argv);
}

/* -------------------------------------------------------------------- */
const char* seq_set(int argc, const char* argv[])
{
    if (argc == 3)
    {
        // Parse the arguments.
        AlignedSequence* sequence = NULL;
        int sequenceID = parsePositiveInteger(argv[1], sequences->getSize()-1);
        if (sequenceID != -1) 
        {
           sequence = (AlignedSequence*)sequences->get(sequenceID);
        }
        if (sequence == NULL)
        {
            printf("[seq set] Invalid sequence id: %s\n", argv[1]);
            return "";
        }

        const char* sequenceData = argv[2];

//        printf("precolors in seq_set: "); ((ShortIntList*)(colorings->get(sequenceID)))->printList();

        Alphabet* alphabet = sequence->getAlphabet();

        // store the old coloring so we don't have to retrieve it a bunch
        ShortIntList* oldColor = (ShortIntList*)(colorings->get(sequenceID));

        // we want size to be the number of entries in the new sequence.
        // the new sequence is like "A T C", so we would want 3 in that
        // case.
        int size = (strlen(sequenceData)+1) / 2;

        ShortIntList* colors = new ShortIntList();
        int origIndex = 0;
        int i=0;
        for(i=0; i<size; i++) 
        {
           if(sequenceData[(2*i)] == '-') 
           {
              if( sequence->get(origIndex).getOne() != '-' ) 
              {
                 colors->add((short int)0);
              } 
              else 
              {
                 colors->add(oldColor->get(origIndex));
                 origIndex++;
              }
           } 
           else   // sequenceData[2*i] was not '-'
           {
              while ( sequence->get(origIndex).getOne() == '-' ) 
              {
                 origIndex++;
              }

              colors->add(oldColor->get(origIndex));
              origIndex++;
           }
        } // end the for loop on i


        delete oldColor;
        sequences->set(sequenceID, NULL);	
        colorings->set(sequenceID, NULL);

        // Create a new sequence with the new sequence data.
        AlignedSequence* sequence2 = new AlignedSequence(alphabet,
                                                         sequence->getName());

        // we are done with the old sequence.  Get rid of it
        delete sequence;
        addSequenceData(sequenceData, sequence2, colors);

        // Add the sequence to the store.
        sequences->set(sequenceID, sequence2);
        colorings->set(sequenceID, colors);
 //       printf("pstcolors in seq_set: "); ((ShortIntList*)(colorings->get(sequenceID)))->printList();
//        printf("new sequence passed : %s\n", sequenceData);

        return "";
    }

    return seq_usage(argc, argv);
}

/* -------------------------------------------------------------------- */
const char* seq_position_of_residue(int argc, const char* argv[])
{
    if (argc == 3)
    {
        // Parse the arguments.
        AlignedSequence* sequence = NULL;
        int sequenceID = parsePositiveInteger(argv[1], sequences->getSize()-1);
        if (sequenceID != -1) sequence = (AlignedSequence*)sequences->get(sequenceID);
        if (sequence == NULL)
        {
            printf("[seq posOf] Invalid sequence id: %s\n", argv[1]);
            return "";
        }
        int residueIndex = parsePositiveInteger(argv[2], sequence->getNumberResidues()-1);
        if (residueIndex == -1)
        {
            printf("[seq posOf] Invalid residue: %s\n", argv[2]);
            return "";
        }
        
        int positionIndex = sequence->getPositionForResidue(residueIndex);
        sprintf(output, "%d", positionIndex);
        return output;        
    }
    
    return seq_usage(argc, argv);
}

// ------------------------------------------------------------------------
const char* seq_residue_at_position(int argc, const char* argv[])
{
    if (argc == 3)
    {
        // Parse the arguments.
        AlignedSequence* sequence = NULL;
        int sequenceID = parsePositiveInteger(argv[1], sequences->getSize()-1);
        if (sequenceID != -1) 
        {
           sequence = (AlignedSequence*)sequences->get(sequenceID);
        }
        if (sequence == NULL)
        {
            printf("[seq resAt] Invalid sequence id: %s\n", argv[1]);
            return "";
        }
        int positionIndex = parsePositiveInteger(argv[2], 
                                      sequence->getNumberPositions()-1);
        if (positionIndex == -1)
        {
            printf("[seq resAt] Invalid position (%s) requested for seq %d (which is %d long)\n", 
                         argv[2], sequenceID, sequence->getNumberPositions());
            return "";
        }
        
        int residueIndex = sequence->getResidueForPosition(positionIndex);
        if (residueIndex == ShortIntList::MAX)
        {
            residueIndex = -1;
        }
        sprintf(output, "%d", residueIndex);
        return output;        
    }
    
    return seq_usage(argc, argv);
}

/* ----------------------------------------------------------------------- */
const char* seq_name(int argc, const char* argv[])
{
    if (argc >= 3)
    {
        // Parse the arguments.
        AlignedSequence* sequence = NULL;
        int sequenceID = parsePositiveInteger(argv[2], sequences->getSize()-1);
//        printf("[seq name: ");
//        for (int i=1; i < argc; i++) {
//           printf(":'%s',",  argv[i]);
//        }
//        printf("; seqId: %d]\n", sequenceID);

        if (sequenceID != -1) 
        {
           sequence = (AlignedSequence*)sequences->get(sequenceID);
        }
        if (sequence == NULL)
        {
            printf("[seq name %s] Invalid sequence id: %s\n", argv[1], argv[2]);
            return "";
        }

        if (strcmp(argv[1], "get") == 0) 
        {
           return sequence->getName();
        } else if (strcmp(argv[1], "set") == 0) {
           if (argc == 4) {
              sequence->setName(argv[3]);
           } else {
               printf("seq name set <value>: incorrect usage\n");
           }
           return "";
        }
    }
    
    return seq_usage(argc, argv);
}
 
/* ----------------------------------------------------------------------- */
const char* seq_get_color(int argc, const char* argv[])
{
    if (argc == 4)
    {
        // Parse the arguments.
        AlignedSequence* sequence = NULL;
        int sequenceID = parsePositiveInteger(argv[2], sequences->getSize()-1);
        if (sequenceID != -1) sequence = (AlignedSequence*)sequences->get(sequenceID);
        if (sequence == NULL)
        {
            printf("[seq get color] Invalid sequence id: %s\n", argv[2]);
            return "";
        }

        ShortIntList* colors = (ShortIntList*)colorings->get(sequenceID);
        int colorSize = colors->getSize();

        int positionIndex = parsePositiveInteger(argv[3], colorSize-1);
        if (positionIndex == -1)
        {
            printf("[seq get color] Invalid position for seq %d: %s\n",
                                                       sequenceID, argv[3]);
            return "";
        }
        
        sprintf(output, "%d", colors->get(positionIndex));
        return output;        
    }
    
    return seq_usage(argc, argv);
}
 
/* ----------------------------------------------------------------------- */
const char* seq_set_color(int argc, const char* argv[])
{
    if (argc == 5 || argc == 6)
    {
        // Parse the arguments.
        AlignedSequence* sequence = NULL;
        int sequenceID = parsePositiveInteger(argv[2], sequences->getSize()-1);
        if (sequenceID != -1) sequence = (AlignedSequence*)sequences->get(sequenceID);
        if (sequence == NULL)
        {
            printf("[seq set color] seqId: %d, Invalid sequence id: %s\n",
                                               sequenceID, argv[2]);
            return "";
        }

        ShortIntList* colorValues = (ShortIntList*)colorings->get(sequenceID);
        int colorSize = colorValues->getSize();

        int startPositionIndex = parsePositiveInteger(argv[3], colorSize-1);
        if (startPositionIndex == -1)
        {
            printf("[seq set color] seqId: %d, Invalid pos out of %d: %s\n",
                         sequenceID, colorSize, argv[3]);
            return "";
        }
        int colorArg = 4;
        int endPositionIndex = startPositionIndex;
        if (argc == 6)
        {
            colorArg = 5;
            endPositionIndex = parsePositiveInteger(argv[4], colorSize-1);
            if (endPositionIndex == -1)
            {
                printf("[seq set color] seqId: %d, Invalid end pos out of %d: %s\n",
                        sequenceID, colorSize, argv[4]);
                return "";
            }
            if (startPositionIndex > endPositionIndex)
            {
                printf("[seq set color] seqId: %d, Invalid range: %s - %s\n", 
                                          sequenceID, argv[3], argv[4]);
                return "";
            }
        }
        int color = parsePositiveInteger(argv[colorArg], ShortIntList::MAX);
        if (color == -1)
        {
            printf("[seq set color] seqId: %d, Invalid color: %s\n", 
                                               sequenceID, argv[colorArg]);
            return "";
        }
        
        int i;
        for (i=startPositionIndex; i <= endPositionIndex; i++)
        {
            colorValues->set(i, color);
        }
        return "";
    }
    
    return seq_usage(argc, argv);
}

/* ----------------------------------------------------------------------- */
int countListItems(const char* list)
{
    int seperators = 0;
    while (*list != '\0')
        if (*list++ == ' ')
            seperators++;
    return seperators+1;
}

/* ----------------------------------------------------------------------- */
char* getNextListItem(char** list)
{
    if ((*list)[0] == '\0')
        return NULL;
    char* item = (*list);
    while ((*list)[0] != ' ' && (*list)[0] != '\0')
        *list = *list+1;
    if ((*list)[0] == ' ')
    {
        (*list)[0] = '\0';
        *list = *list+1;
    }
    return item;
}

/* ----------------------------------------------------------------------- */
Alphabet* determineAlphabet(const char* sequenceData)
{
   int protein=0, rna=0, dna=0, maxToCheck=60;
   int proteinUnknownIndex = proteinAlphabet->getSymbolIndex('?');
   int rnaUnknownIndex = rnaAlphabet->getSymbolIndex('?');
   int dnaUnknownIndex = dnaAlphabet->getSymbolIndex('?');
   
   int i;
   for (i=0; i < maxToCheck && sequenceData[i] != '\0'; i++)
   {
      if (sequenceData[i] == ' ' || sequenceData[i] == '-' || sequenceData[i] == '~' || sequenceData[i] == '.')
      {
         maxToCheck++;
      }
      else
      {
         protein += (proteinAlphabet->getSymbolIndex(sequenceData[i]) == proteinUnknownIndex)?0:1;
         rna += (rnaAlphabet->getSymbolIndex(sequenceData[i]) == rnaUnknownIndex)?0:1;
         dna += (dnaAlphabet->getSymbolIndex(sequenceData[i]) == dnaUnknownIndex)?0:1;
      }
   }

   // foobar.  do they really want int math here? 
   if ((protein*95/100) > dna && (protein*95/100) > rna)
      return proteinAlphabet;
   else if (dna > rna)
      return dnaAlphabet;
   else 
      return rnaAlphabet;
}

/* ----------------------------------------------------------------------- */
// if string is an int in [0 .. maxValue], return it.  else -1
int parsePositiveInteger(const char* string, int maxValue)
{
   int val = atoi(string);
   if (   ((val == 0 && strlen(string) == 1 && string[0] == '0') || 
            val > 0) && 
        val <= maxValue)
   {
      return val;
   }
   return -1;
}

/* ----------------------------------------------------------------------- */
void addSequenceData(const char *sequenceData, AlignedSequence* sequence, 
                                                    ShortIntList* colors)
{
//    printf ("data: %s\n", sequenceData);
//   int i;
    // If there are any spaces, assume this is a list.
    if (strstr(sequenceData, " ") != NULL)
    {
       do {
//          printf("%c", *sequenceData);
          if (*sequenceData != ' ') {
              sequence->add(*sequenceData);
          }
       } while (*(++sequenceData));

/*
        // Add the first character, if it is not a space.
        if (sequenceData[0] != ' ') {
           sequence->add(sequenceData[0]);
        }
        
        // Go through the data and add the first charcter after each space.
        sequenceData = strstr(sequenceData, " ");
        while (sequenceData != NULL)
        {
            // Skip over the space.
            sequenceData++;
            
            // If this isn't the end of the string, add the next character.
            if (sequenceData[0] != '\0') {
               sequence->add(sequenceData[0]);
            }
            
            // Find the next space.
            sequenceData = strstr(sequenceData, " ");
        }
*/
    } else {
        // Otherwise assume it is a sequence of single characters.
        sequence->addAll(sequenceData);
    }
    sequence->optimize();
   
//    printf ("size is %d\n", sequence->getSize());

//    // Set the initial colors.
    if(colors->getSize() == 0) {
       colors->initialize(sequence->getSize(), 0);
    }

//    if(colors->getSize() == 0) {
//    for(i=0; i<sequence->getSize(); i++) {
//      colors->add(0);
//     }
//    }
}

