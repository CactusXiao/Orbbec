mkdir -p build
cmake -S . -B build
cmake --build build -j
mkdir -p bin
cp build/orbbec bin/orbbec
