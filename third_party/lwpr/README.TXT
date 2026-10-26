Contents of the LWPR library  
(C) 2007 Stefan Klanke and Sethu Vijayakumar
sethu.vijayakumar@ed.ac.uk

The library is freely available under the terms of the LGPL 
with an exception that allows for static linking, see the
file COPYING.

This code is based on the LWPR library from https://github.com/lhlmgr/lwpr

--- The section below describes THIS reduced copy, not the upstream distribution.
--- It was rewritten by the authors of the accompanying paper; everything above
--- and the warranty notice below are upstream's own words, unchanged.

This is a subset of LWPR 1.2.6, vendored so that the LWPR baseline in the
accompanying paper can be rebuilt from source. The algorithm sources are
unmodified. Relative to upstream, the MATLAB bindings, the autotools build, the
Python-2 setup scripts and the C example have been removed, and the CMake build
was modernised (see NOTICE at the root of the release for the full statement of
changes required by LGPL 2.1 section 2a).

Directories present here:

  include     C headers, plus the C++ wrapper lwpr.hh
  src         C sources of the library
  python      lwprmodule.c, the CPython extension exposing the LWPR class
  example_cpp a demo of the C++ wrapper
  tests       cross_check.c, a standalone numerical check
  doc         the upstream documentation source and the XML DTD

Building the Python module:

    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DNUM_THREADS=1
    make
    make install
    python -c "import pylwpr; print(pylwpr.LWPR(2, 1))"

The install prefix matters: the extension links against liblwpr.so.1, so the
library has to land somewhere the loader will find it. Keep NUM_THREADS=1 --
more than one thread changes the order of floating-point summation, and the
numbers reported in the paper come from the single-threaded build.

THIS SOURCE CODE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND, 
AND ITS AUTHOR AND THE JOURNAL OF MACHINE LEARNING RESEARCH (JMLR) 
AND JMLR'S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL 
WARRANTIES, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND 
ANY WARRANTIES OR NON INFRINGEMENT. THE USER ASSUMES ALL LIABILITY 
AND RESPONSIBILITY FOR USE OF THIS SOURCE CODE, AND NEITHER THE 
AUTHOR NOR JMLR, NOR JMLR'S PUBLISHERS AND DISTRIBUTORS, WILL BE 
LIABLE FOR DAMAGES OF ANY KIND RESULTING FROM ITS USE. 
Without limiting the generality of the foregoing, neither the 
author, nor JMLR, nor JMLR's publishers and distributors, warrant 
that the Source Code will be error-free, will operate without 
interruption, or will meet the needs of the user.
