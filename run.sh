#!/bin/bash

g++ -I../output_out/include/opencv4 -L../output_out/lib -Wl,-rpath=../output_out/lib main.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_shape;
rm core;
ulimit -c unlimited;
./a.out;