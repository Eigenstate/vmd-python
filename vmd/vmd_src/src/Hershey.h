/*
 * Modified Hershey Roman font rendering code.
 *
 * $Id: Hershey.h,v 1.7 2011/02/10 21:22:35 johns Exp $
 */
#ifndef __Hershey_h__
#define __Hershey_h__

/* private handle data structure */
typedef struct {
  float lm;
  float rm;
  const char *p;
} hersheyhandle;

/// This routine generates a sequence of OpenGL drawing commands that
/// can be captured in a display list for fast subsequent rendering of
/// each of the font glyphs
void hersheyDrawLetterOpenGL(unsigned char ch, int drawendpoints);

/// For VMD FileRenderer subclasses, it is most convenient to generate
/// the font glyphs on the fly.  This routine initializes a glyph 
/// rendering context for the requested character, to be accessed by
/// subsequent calls to hersheyDrawNextLine().  The lm and rm values
/// are X coordinate offsets to be applied before (lm) and after (rm)
/// rendering the glyph, to produce the correct text spacing.
void hersheyDrawInitLetter(hersheyhandle *hh, const char ch, 
                           float *lm, float *rm);

/// This routine computes the next glyph stroke vector, returning false
/// for a run of connected strokes, and true when the stroke ends.
/// It sets the value of the 'draw' parameter to true when X and Y 
/// are set, which allows the caller to determine when strokes begin and end,
/// and to connect the segments with whatever geometry works best for the
/// specific renderer.
int hersheyDrawNextLine(hersheyhandle *hh, int *draw, float *x, float *y);

#endif /* __Hershey_h__ */
