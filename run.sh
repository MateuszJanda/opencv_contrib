#!/bin/bash

g++ -I../output_bug/include/opencv4 -L../output_bug/lib -Wl,-rpath=../output_bug/lib main.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_shape;
rm core;
ulimit -c unlimited;
./a.out;