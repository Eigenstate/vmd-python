############################################################################
#cr
#cr            (C) Copyright 1995-2004 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: metric_type.tcl,v $
#       $Author: kvandivo $        $Locker:  $             $State: Exp $
#       $Revision: 1.2 $       $Date: 2009/03/31 17:00:52 $
#
############################################################################

# This package implements a color map for the sequence editor that colors
# sequence elements based upon the conservation of the element.

package provide seqedit 1.1
package require seqdata 1.1

# Declare global variables for this package.
namespace eval ::SeqEdit::Metric::Type {

    # Export the package namespace.
    namespace export calculate

    proc calculate {sequenceIDs} {
        
        # Initialize the color map.
        array unset metricMap 
        set colorValueMap "[::SeqEditWidget::getColorMap]\::getColorIndexForValue"

        # Go through each sequence in the list
        foreach sequenceID $sequenceIDs {
            
            # Get the sequence
            set sequence [SeqData::getSeq $sequenceID]
            
            # Go through each element in the sequence.
            set elementIndex 0
            foreach element $sequence {
#               puts "metric_type.calculate. seqID: $sequenceID, elemIdx: $elementIndex, elem: $element"
                # Set the color map entry for this element.
                if {$element == "R" || $element == "K" || $element == "H"} {
                   seq set color $sequenceID $elementIndex [$colorValueMap 0]
                } elseif {$element == "D" || $element == "E"} {
                   seq set color $sequenceID $elementIndex [$colorValueMap 1]
                } else {
                   seq set color $sequenceID $elementIndex [$colorValueMap .5]
                }
                
                # Increment the element counter.
                incr elementIndex
            }
        }
        # Return the metric map.
        #return [array get metricMap]
    }
}
