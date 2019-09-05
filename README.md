# ONNX-parser
parse onnx model & onnx operator convert 

### Complie
```
g++ main.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o onnxParse

g++ op_convert.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o op_convert

```

### Usage

```
# parse onnx model
./bin/onnxParse  model.onnx

# onnx operator convert
./bin/op_convert  input.onnx  output_name

# cherck onnx model
python3 ./python_tool/cherck_model.py  out.onnx 

# run onnx model
python3 ./python_tool/onnx_run2.py  out.onnx

```

### ONNX protocol
![parse](./img/ONNX.jpg)