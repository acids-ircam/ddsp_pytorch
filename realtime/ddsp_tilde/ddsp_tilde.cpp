#include "m_pd.h"
#include "ddsp_model.h"
#include <string>
#include <thread>

#define B_SIZE 1024

static t_class *ddsp_tilde_class;

struct ddsp_tilde
{
    t_object x_obj;
    t_float f;

    t_inlet *x_in2;
    t_outlet *x_out;

    float pitch_buffer[2 * B_SIZE];
    float loudness_buffer[2 * B_SIZE];
    float out_buffer[2 * B_SIZE];

    int head;

    std::thread *compute_thread;

    DDSPModel *model;
};

void thread_perform(ddsp_tilde *x, float *pitch, float *loudness,
                    float *out_buffer, int buffer_size)
{
    x->model->perform(pitch, loudness, out_buffer, buffer_size);
}

void *ddsp_tilde_new()
{
    ddsp_tilde *x = (ddsp_tilde *)pd_new(ddsp_tilde_class);
    x->x_in2 = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    x->x_out = outlet_new(&x->x_obj, &s_signal);
    x->model = new DDSPModel;

    x->head = 0;

    return (void *)x;
}

void ddsp_tilde_free(ddsp_tilde *x)
{
    inlet_free(x->x_in2);
    outlet_free(x->x_out);
}

void ddsp_tilde_load(ddsp_tilde *x, t_symbol *sym)
{
    int status = x->model->load(sym->s_name);
    if (!status)
    {
        post("successfully loaded model");
    }
    else
    {
        post("error loading model");
    }
}

t_int *ddsp_tilde_perform(t_int *w)
{
    ddsp_tilde *x = (ddsp_tilde *)(w[1]);
    t_sample *in1 = (t_sample *)(w[2]);
    t_sample *in2 = (t_sample *)(w[3]);
    t_sample *out = (t_sample *)(w[4]);
    int n = (int)(w[5]);

    memcpy(x->pitch_buffer + x->head, in1, n * sizeof(float));
    memcpy(x->loudness_buffer + x->head, in2, n * sizeof(float));
    memcpy(out, x->out_buffer + x->head, n * sizeof(float));

    x->head += n;

    if (!(x->head % B_SIZE))
    {
        if (x->compute_thread)
        {
            x->compute_thread->join();
        }
        int model_head = ((x->head + B_SIZE) % (2 * B_SIZE));
        x->compute_thread = new std::thread(thread_perform, x,
                                            x->pitch_buffer + model_head,
                                            x->loudness_buffer + model_head,
                                            x->out_buffer + model_head,
                                            B_SIZE);

        x->head = x->head % (2 * B_SIZE);
    }

    return (w + 6);
}

void ddsp_tilde_dsp(ddsp_tilde *x, t_signal **sp)
{
    dsp_add(ddsp_tilde_perform, 5, x,
            sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_n);
}

extern "C"
{
    void ddsp_tilde_setup(void)
    {
        ddsp_tilde_class = class_new(gensym("ddsp~"),
                                     (t_newmethod)ddsp_tilde_new,
                                     (t_method)ddsp_tilde_free,
                                     sizeof(ddsp_tilde),
                                     CLASS_DEFAULT,
                                     A_DEFFLOAT, 0);
        CLASS_MAINSIGNALIN(ddsp_tilde_class, ddsp_tilde, f);

        class_addmethod(ddsp_tilde_class,
                        (t_method)ddsp_tilde_load,
                        gensym("load"),
                        A_SYMBOL, 0);

        class_addmethod(ddsp_tilde_class,
                        (t_method)ddsp_tilde_dsp,
                        gensym("dsp"), A_CANT, 0);
    }
}