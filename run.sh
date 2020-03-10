#!/bin/bash

g++ -I../venv/local/include/opencv4 -L../venv/local/lib -Wl,-rpath=../venv/local/lib main.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_shape;
rm core;
ulimit -c unlimited;
./a.out;