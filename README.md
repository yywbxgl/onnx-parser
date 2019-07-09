# ONNX-parser

### 编译命令
g++ main.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o test

### bin目录是是编译后的文件，可在linux直接运行
./bin/onnxParse model_file.onnx
