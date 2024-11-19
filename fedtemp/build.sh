cd ./build
cmake -DTORCH_DIR="/mnt/d/cpp_libs/libtorch-cxx11-abi-shared-with-deps-2.5.0+cpu/share/cmake/Torch" -DBOOST_DIR="/mnt/d/cpp_libs/boost_1_86_0/tools/boost_install" ..
cmake --build . --config Release