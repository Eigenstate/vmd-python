#include "typeConvert.h"


// Converts a string (temp) to a float
float charToFloat(char *temp) {

  int flag = 0;
  float ans = 0;
  int i, dec=0;

  for (i=0;i<(int)strlen(temp);i++)
    if(temp[i]=='.')
      dec = i;

  for (i=0;i<(int)strlen(temp);i++) {
    if(temp[i]==45) flag=1;
    if(temp[i]>=48 && temp[i]<=57)
      ans += pow((float)10,dec-i-1*(i<dec))*(temp[i]-48);
  }
  if (flag) ans*=-1;

  return ans;
}


// Converts a string (temp) to an int
int charToInt(char *temp) {

  int i,flag=0,ans=0;
  //int len=strlen(temp);

  while (temp[0]==' ') {
    strcpy(temp,temp+1);
    temp[strlen(temp)] = '\0';
  }
  while (temp[strlen(temp)-1]==' ') {
    temp[strlen(temp)-1] = '\0';
  }

  for (i=0;i<(int)strlen(temp);i++) {
    if(temp[i]==45) flag=1;
    if(temp[i]>=48 && temp[i]<=57)
      ans += (int)pow((float)10,(float)strlen(temp)-1-i)*(temp[i]-48);
  }
  if (flag) ans*=-1;

  return ans;
}


// Converts and integer num to a string of length len
char* intToString(int num, int len) {

  int negative = 0;
  //int ans = 0;
  int i,j;
  float k;

  char *temp = new char[len+1];

  if (num<0) negative=1;

  if (num<0) num*=-1;

  for (i=0; i<len; i++) {
    k = num/(pow((float)10,i));
    j = (int)k%10;
    temp[len-i-1] = j+48;
  }

  i = 0;
  while (temp[i]=='0')
    temp[i++] = ' ';
  if (temp[i]=='\0') temp[i-1] = '0';

  if (negative)
    temp[i-1] = '-';
    
  temp[len] = '\0';

  return temp;
}


// Accepts a character, returns a 2 character string containing
// the accepted character and the NULL character
char* charToString(char i) {

  char *temp = new char[2];
  temp[0] = i;
  temp[1] = '\0';

  return temp;
}
  

// Converts a string to lower case
char* lower(char *s) {
   int i;
  for (i=0; i<(int)strlen(s); i++) {
    if (s[i]>=65 && s[i]<=90) s[i]+=32;
  }

  return s;
}
