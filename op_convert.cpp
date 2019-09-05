#include <iostream>
#include <fstream>
#include <string>
#include <iomanip> 
#include <vector>

#include "onnx.pb.h"

using namespace std;


void modify_input(onnx::ModelProto& model_proto)
{
    //修改producer_name
    std::string* name = model_proto.mutable_producer_name();
    *name = "sun-smile";
    cout << "---- modify producer_name, name = " << model_proto.producer_name() << endl;

    //修改input
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(0).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(1).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(2).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(3).dim_value() << endl;

    cout << "---- modify input shape " << endl;
    model_proto.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(1);
    model_proto.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_value(224);
    model_proto.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_value(224);

    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(0).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(1).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(2).dim_value() << endl;
    cout << "modele input " << model_proto.graph().input(0).type().tensor_type().shape().dim(3).dim_value() << endl;

    //修改output
    // cout << "modele output size " << model_proto.graph().output(0).type().tensor_type().shape().dim_size() << endl;
    model_proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(1);
    cout << "modele output " << model_proto.graph().output(0).type().tensor_type().shape().dim(0).dim_value() << endl;
    cout << "modele output " << model_proto.graph().output(0).type().tensor_type().shape().dim(1).dim_value() << endl;



    return;
}

//直接删除单层node
void eliminate_node(onnx::ModelProto& model_proto, std::string node_type)
{
    bool finish_flag = false;
    while(!finish_flag)
    {
        google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
        it = model_proto.mutable_graph()->mutable_node()->begin();

        for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
        {
            std::string input;
            std::string output;
            if (it->op_type() == node_type)
            {
                cout << "---- find node " << node_type << "----" << endl;
                cout << "name = " << it->name()  << endl;
                cout << "op_type = " << it->op_type() << endl;
                cout << "input = " << it->input(0) << endl;  // identify has only one input
                cout << "out = " << it->output(0) << endl;

                input = it->input(0);
                output = it->output(0);

                //找到上一层，修改上层的ouput
                for (int i=0; i<model_proto.graph().node_size(); ++i) 
                {
                    if (model_proto.graph().node(i).output(0) == input)
                    {
                        cout << "---- find prev node: " << endl;
                        cout << "name = " << model_proto.graph().node(i).name()  << endl;
                        cout << "op_type = " << model_proto.graph().node(i).op_type() << endl;
                        cout << "input = " << model_proto.graph().node(i).input(0) << endl; 
                        cout << "out = " << model_proto.graph().node(i).output(0) << endl;

                        cout << "modify the output to " << output << endl;
                        model_proto.mutable_graph()->mutable_node(i)->set_output(0, output);
                    }   
                }

                //删除当前node
                model_proto.mutable_graph()->mutable_node()->erase(it);
                cout << "eliminate current node success."  << endl << endl;
                break;
            }
        }

        if(it == model_proto.mutable_graph()->mutable_node()->end())
        {
            finish_flag = true;
        }
    }
}



void fuse_matmul_add_bias_into_gemm(onnx::ModelProto& model_proto)
{
    google::protobuf::RepeatedPtrField<onnx::NodeProto>::iterator it; 
    it = model_proto.mutable_graph()->mutable_node()->begin();
    for (; it != model_proto.mutable_graph()->mutable_node()->end(); ++it)
    {
        std::string input;
        std::string output;
        std::string gemm_B;
        std::string gemm_C;
        std::string Add_output;

        if (it->op_type() == "MatMul")
        {
            cout << "---- find MatMul node " << "----" << endl;
            cout << "name = " << it->name()  << endl;
            cout << "op_type = " << it->op_type() << endl;
            cout << "input = " << it->input(0) << "  " << it->input(1) << endl;  //  has two input
            cout << "out = " << it->output(0) << endl;

            input = it->input(0);
            gemm_B = it->input(1);
            output = it->output(0);
        }

        //找到MatMul的下一层，是否为Add
        for (int i=0; i<model_proto.graph().node_size(); ++i) 
        {
            if (model_proto.graph().node(i).input(0) == output)
            {
                cout << "---- find MatMul next node: " << endl;
                cout << "name = " << model_proto.graph().node(i).name()  << endl;
                cout << "op_type = " << model_proto.graph().node(i).op_type() << endl;
                cout << "input = " << model_proto.graph().node(i).input(0) << "  " << model_proto.graph().node(i).input(1) << endl; 
                cout << "out = " << model_proto.graph().node(i).output(0) << endl;

                gemm_C = model_proto.graph().node(i).input(1);
                Add_output =  model_proto.graph().node(i).output(0);

                if (model_proto.graph().node(i).op_type() == "Add")
                {
                    cout << "next node is add. fuse_matmul_add_bias_into_gemm." << endl;
                    // 创建新的GEMM Node
                    // onnx::NodeProto* gemm_node =  model_proto.mutable_graph()->add_node();
                    // string gemm_name =   "Gemm_" + output;
                    // string gemm_out_name =  "Gemm_" + output + "_Y";
                    // gemm_node->set_name(gemm_name);
                    // gemm_node->set_op_type("Gemm");
                    // gemm_node->add_input(input);
                    // gemm_node->add_input(gemm_B);
                    // gemm_node->add_input(gemm_C);
                    // gemm_node->add_output(gemm_out_name);

                    // cout << "---- create gemm node." << endl;
                    // cout << "name = " << gemm_node->name()  << endl;
                    // cout << "op_type = " << gemm_node->op_type() << endl;
                    // cout << "input = " << gemm_node->input(0) << "  " << gemm_node->input(1) <<  "  "<< gemm_node->input(2)  << endl; 
                    // cout << "out = " << gemm_node->output(0) << endl;

                    //修改当前add变为gemm，output shape 应该保持不变
                    cout << "---- modify Add node to gemm" << endl;
                    // model_proto.mutable_graph()->mutable_node(i)->set_name(gemm_name);
                    model_proto.mutable_graph()->mutable_node(i)->set_op_type("Gemm");
                    // model_proto.mutable_graph()->mutable_node(i)->set_input(0, input);
                    model_proto.mutable_graph()->mutable_node(i)->set_input(1, gemm_B);
                    model_proto.mutable_graph()->mutable_node(i)->add_input(gemm_C);
                    // model_proto.mutable_graph()->mutable_node(i)->set_output(0, gemm_C);

                    // cout << "op_type = " << gemm_node->op_type() << endl;
                    // cout << "input = " << model_proto.mutable_graph()->mutable_node(i).input(0) << "  " << gemm_node->input(1) <<  "  "<< gemm_node->input(2)  << endl; 
                    // cout << "out = " << gemm_node->output(0) << endl;

                    //修改MatMul节点变为reshape, 修改weiight指向new_reshape
                    cout << "---- modify MatMul node to reshape" << endl;
                    std::string new_shape_name = it->name() + "_reshape";
                    it->set_op_type("Reshape") ;
                    it->set_input(1, new_shape_name) ;
                    
                    //添加reshape 到initializer
                    cout << "---- add reshape to initializer" << endl;
                    onnx::TensorProto* new_shape = model_proto.mutable_graph()->add_initializer();  
                    new_shape->set_name(new_shape_name);
                    new_shape->set_data_type(7);  //data_type = INT64
                    new_shape->add_dims(2);
                    new_shape->add_int64_data(1);  // todo 根据gemm B 参数  已经该层的output形状 来确认reshape形状
                    new_shape->add_int64_data(2048);
                    cout << "new reshape data  " << new_shape->int64_data(0) << endl;
                    cout << "new reshape data  " << new_shape->int64_data(1) << endl;

                    //添加reshape 到input
                    cout << "---- add reshape to input" << endl;
                    onnx::ValueInfoProto* new_input = model_proto.mutable_graph()->add_input();  
                    new_input->set_name(new_shape_name);
                    new_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
                    new_input->mutable_type()->mutable_tensor_type()->set_elem_type(7);

                }
            }   
        }
    }

    return;
}

int main(int argc, char const *argv[]) 
{

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 3) 
    {
        std::cerr << "Usage: " << argv[0] << " ONNX_FILE  " <<  " OUTPUT_FILE"  << '\n';
        return -1;
    }

    onnx::ModelProto model_proto;

    {
        // Read the existing model.
        std::fstream input(argv[1], ios::in | ios::binary);
        if (!input) 
        {
            std::cout << argv[1] << ": file not found. Creating a new file." << '\n';
        } 
        else if (!model_proto.ParseFromIstream(&input)) 
        {
            std::cerr << "Failed to parse onnx model." << std::endl;
            return -1;
        }
    }

    // onnx convert
    eliminate_node(model_proto, "Identity");
    eliminate_node(model_proto, "Dropout");
    eliminate_node(model_proto, "ReduceMean");

    // fuse_matmul_add_bias_into_gemm(model_proto);
    // modify_input(model_proto);

    {
        // Write the new model back to disk.
        std::string output_file_name  = std::string(argv[2]) + ".onnx";
        fstream output(output_file_name, ios::out | ios::trunc | ios::binary);
        if (!model_proto.SerializeToOstream(&output)) 
        {
            std::cerr << "Failed to write address book." << endl;
            return -1;
        }
    }


    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
