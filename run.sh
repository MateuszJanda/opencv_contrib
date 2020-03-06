#!/bin/bash

g++ -I../output/include/opencv4 -L../output/lib -Wl,-rpath=../output/lib main.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_shape;
rm core;
ulimit -c unlimited;
./a.out;