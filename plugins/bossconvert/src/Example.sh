#!/bin/sh

if [ -x Topology ]; then
    mkdir -p ExampleOutput
    cd ExampleOutput
    cp ../ExampleInput/* .
    echo "Running ../Topology -oplssb ../../oplsaa.sb -oplspar ../../oplsaa.par -imprlist ../../imprlist -bossout out test.z"
    echo ""
    ../Topology -oplssb ../../oplsaa.sb -oplspar ../../oplsaa.par -imprlist ../../imprlist -bossout out test.z
    rm -f test.z out
    echo "Output written to ExampleOutput/"
    echo ""
else
    echo "Topology executable not compiled... run make and then Example.sh"
fi
