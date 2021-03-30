#include "c74_min.h"
#include "dlfcn.h"
#include <string>
#include <thread>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>

#define DEVICE torch::kCPU
#define CPU torch::kCPU

#define B_SIZE 1024

using namespace c74::min;

class DDSPModel
{
    private:
        torch::jit::script::Module m_scripted_model;
        int m_loaded;

    public:
        DDSPModel();
        int load(std::string path);
        void perform(float *pitch, float *loudness, float *out_buffer, int buffer_size);
};

DDSPModel::DDSPModel() : m_loaded(0)
{
    at::init_num_threads();
}

int DDSPModel::load(std::string path)
{
    try
    {
        m_scripted_model = torch::jit::load(path);
        m_scripted_model.eval();
        m_scripted_model.to(DEVICE);
        m_loaded = 1;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << '\n';
        m_loaded = 0;
        return 1;
    }
}

void DDSPModel::perform(float *pitch, float *loudness, float *out_buffer, int buffer_size)
{
    torch::NoGradGuard no_grad;
    if (m_loaded)
    {
        auto pitch_tensor = torch::from_blob(pitch, {1, buffer_size, 1});
        auto loudness_tensor = torch::from_blob(loudness, {1, buffer_size, 1});

        pitch_tensor = pitch_tensor.to(DEVICE);
        loudness_tensor = loudness_tensor.to(DEVICE);

        std::vector<torch::jit::IValue> inputs = {pitch_tensor, loudness_tensor};

        auto out_tensor = m_scripted_model.forward(inputs).toTensor();
        out_tensor = out_tensor.to(CPU);

        auto out = out_tensor.contiguous().data_ptr<float>();

        memcpy(out_buffer, out, buffer_size * sizeof(float));
    }
}

class hello_world : public object<hello_world>, public sample_operator<2, 1> {
    private:
        // inlets and outlets that will be defined at runtime
        std::vector< std::unique_ptr<inlet<>> > m_inlets;
        std::vector< std::unique_ptr<outlet<>> > m_outlets;
    
        // controller variables for the model
        bool m_initialized { false };
        DDSPModel *model { nullptr };

        // buffers
        float pitch_buffer[2 * B_SIZE];
        float loudness_buffer[2 * B_SIZE];
        float out_buffer[2 * B_SIZE];

        // variable to store the position in the buffer
        int head {0};
        int model_head {0};
    
        // pointer to run a different thread for the ddsp computation
        std::thread *compute_thread;

    public:
        MIN_DESCRIPTION    {"DDSP realtime timbre transfer"};
        MIN_TAGS           {"utilities"};
        MIN_AUTHOR         {"Kling Klang Klong"};
    
        // execute the ddsp computation in a separate thread
        void thread_perform(float *pitch, float *loudness, float *out_buffer, int buffer_size)
        {
            model->perform(pitch, loudness, out_buffer, buffer_size);
        }
    
        // print a hello world message when the external class is first loaded
        message<> maxclass_setup { this, "maxclass_setup",
            MIN_FUNCTION {
                cout << "Realtime DDSP Max external v0.9" << endl;
                return {};
            }
        };
    
        // on startup
        hello_world(const atoms& args = {}) {
            // to ensure safety in possible attribute settings
            model = new DDSPModel;
            m_initialized = true;
            
            if (args.empty()) {
              error("Please specify the input model path as argument.");
            }
            else {
                symbol model_path = args[0]; // the first argument specifies the path
                int status = model->load(model_path); // try to load the model
                
                if (!status) { // if loaded correctly
                
                    // configure inlets and outlets
                    auto input_pitch_frequency = std::make_unique<inlet<>>(this, "(signal) pitch frequency");
                    auto input_loudness = std::make_unique<inlet<>>(this, "(signal) loudness");
                    auto output_ddsp = std::make_unique<outlet<>>(this, "(signal) ddsp out", "signal");
                    
                    m_inlets.push_back( std::move(input_pitch_frequency) );
                    m_inlets.push_back( std::move(input_loudness) );
                    m_outlets.push_back( std::move(output_ddsp) );
                
                    cout << "Successfully loaded the model!" << endl;
                }
                else {
                    error("Error while loading model. Probably the path is incorrect.");
                }
            }
        }
                        
        sample operator()(sample in1, sample in2) {
            pitch_buffer[head] = in1; // add sample to the pitch buffer
            loudness_buffer[head] = in2; // add sample to the loudness buffer
            
            head++; // progress with the head
            
            if (!(head % B_SIZE)) // if it is B_SIZE or B_SIZE * 2
            {
                model_head = ((head + B_SIZE) % (2 * B_SIZE)); // points to the next / previous B_SIZE spaces available
                compute_thread = new std::thread(&hello_world::thread_perform, this,
                                                pitch_buffer + model_head,
                                                loudness_buffer + model_head,
                                                out_buffer + model_head,
                                                B_SIZE); // compute the buffers in a separate thread

                head = head % (2 * B_SIZE); // set the head to the next available value
            }
                        
            return out_buffer[head]; // outputs the output buffer
        }
};


MIN_EXTERNAL(hello_world);
