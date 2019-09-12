# ONNX-parser
parse onnx model & onnx operator convert 

### Complie
```
g++ onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o onnxParse

g++ onnx_parse.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -D RAW_DATA -o onnxParse_weight

g++ op_convert.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o op_convert

g++ shape_inference.cpp  onnx.pb.cc  /usr/local/lib/libprotobuf.a -std=c++11 -pthread -I/usr/local/include -o shape_inference

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