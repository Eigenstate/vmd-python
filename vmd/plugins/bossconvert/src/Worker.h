#ifndef _WORKER_H_
#define _WORKER_H_
#include <string>
#include <list>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int writeRTFFile(string);
int writePRMFile(string);
bool contains(list<string>, string);
int writeAliasParameters();
//void clean();

#endif
