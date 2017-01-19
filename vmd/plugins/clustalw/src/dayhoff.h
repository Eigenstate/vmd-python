/* DAYHOFF.H
   
   Table of estimated PAMS (actual no. of substitutions per 100 residues)
   for a range of observed amino acid distances from 75.0% (the first entry
   in the array), in 0.1% increments, up to 93.0%.  

   These values are used to correct for multiple hits in protein alignments.  
   The values below are for observed distances above 74.9%.  For values above 
   93%, an arbitrary value of 1000 PAMS (1000% substitution) is used.  

   These values are derived from a Dayhoff model (1978) of amino acid 
   substitution and assume average amino acid composition and that amino 
   acids replace each other at the same rate as in the original Dayhoff model.

   Up to 75% observed distance, use Kimura's emprical formula to derive
   the correction.  For 75% or greater, use this table.  Kimura's formula
   is accurate up to about 75% and fails completely above 85%.
*/

int dayhoff_pams[]={
  195,   /* 75.0% observed d; 195 PAMs estimated = 195% estimated d */
  196,   /* 75.1% observed d; 196 PAMs estimated */
                  197,    198,    199,    200,    200,    201,    202,  203,    
  204,    205,    206,    207,    208,    209,    209,    210,    211,  212,    
  213,    214,    215,    216,    217,    218,    219,    220,    221,  222,    
  223,    224,    226,    227,    228,    229,    230,    231,    232,  233,    
  234,    236,    237,    238,    239,    240,    241,    243,    244,  245,    
  246,    248,    249,    250,    /* 250 PAMs = 80.3% observed d */          
                                  252,    253,    254,    255,    257,  258,    
  260,    261,    262,    264,    265,    267,    268,    270,    271,  273,    
  274,    276,    277,    279,    281,    282,    284,    285,    287,  289,    
  291,    292,    294,    296,    298,    299,    301,    303,    305,  307,    
  309,    311,    313,    315,    317,    319,    321,    323,    325,  328,    
  330,    332,    335,    337,    339,    342,    344,    347,    349,  352,    
  354,    357,    360,    362,    365,    368,    371,    374,    377,  380,    
  383,    386,    389,    393,    396,    399,    403,    407,    410,  414,    
  418,    422,    426,    430,    434,    438,    442,    447,    451,  456,    
  461,    466,    471,    476,    482,    487,    493,    498,    504,  511,    
  517,    524,    531,    538,    545,    553,    560,    569,    577,  586,    
  595,    605,    615,    626,    637,    649,    661,    675,    688,  703,    
  719,    736,    754,    775,    796,    819,    845,    874,    907,  945,
         /* 92.9% observed; 945 PAMs */    
  988    /* 93.0% observed; 988 PAMs */
};

