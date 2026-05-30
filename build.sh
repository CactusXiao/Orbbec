mkdir -p build
cd build
export CMAKE_RUNTIME_OUTPUT_DIRECTORY=../output
cmake .. 
make
mv orbbec ../bin/orbbec
cd ..
