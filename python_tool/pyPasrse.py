import onnx_pb2

def loadModel(model_path):
    # read caffemodel , 序列化
    model = onnx_pb2.ModelProto()
    f = open(model_path, 'rb')
    model.ParseFromString(f.read())
    f.close()
    return model


if __name__ == "__main__":
    model = loadModel( "../model.onnx")
    # print(model.producer_name)
    # print(model.graph.node)
    # print(model.graph.initializer)
    print(model.graph.initializer[0].float_data[0:])
