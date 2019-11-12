#include "m_pd.h"
#include "ddsp_core.hpp"
#include <math.h>
#include <iostream>

#define PD_BLOCK_SIZE 256
#define MODEL_BLOCK_SIZE 160

#define SAMPLING_RATE 16000
#define PI 3.14159265359

static t_class *ddsp_tilde_class;

void sayhello()
{
  std::cout << "hello" << std::endl;
}


typedef struct _ddsp_tilde {
  t_object x_obj;
  t_sample f;


  float *buffer;
  float phase, f0, lo;

  float result[1 + PARTIAL_NUMBER + FILTER_SIZE];

  int bufferReadHead;
  int bufferWriteHead;
  int currentCondition;

  DDSPCore *ddsp;

  t_inlet  *x_in2;
  t_outlet *x_out;
} t_ddsp_tilde;

void processWorker(t_ddsp_tilde *x, t_sample *in1, t_sample *in2){

  float f0, lo, t;

  while (x->currentCondition < PD_BLOCK_SIZE) {
    f0 = in1[x->currentCondition];
    lo = in2[x->currentCondition];

    x->ddsp->getNextOutput(f0, lo, x->result);

    for (int i(0); i<MODEL_BLOCK_SIZE; i++) {
      t = i / float(MODEL_BLOCK_SIZE);

      x->phase += 2 * PI * (x->f0 * (1-t) + f0 * t) / float(SAMPLING_RATE);
      while (x->phase >= 2*PI) x->phase -= 2*PI;

      x->buffer[x->bufferWriteHead] = 0;
      for (int p(1); p<30; p++){
        x->buffer[x->bufferWriteHead] += \
        x->result[0] * x->result[p] * sin(p * x->phase);

        if (p*f0/SAMPLING_RATE > 0.5) {
          break;
        }
      }

      x->bufferWriteHead = (x->bufferWriteHead + 1) % (3 * PD_BLOCK_SIZE);
    }

    x->bufferWriteHead = x->bufferWriteHead % (3 * PD_BLOCK_SIZE);

    x->f0 = f0;
    x->lo = lo;

    x->currentCondition += MODEL_BLOCK_SIZE;

  }
}




t_int *ddsp_tilde_perform(t_int *w)
{
  t_ddsp_tilde *x = (t_ddsp_tilde *)(w[1]);
  t_sample  *in1 =    (t_sample *)(w[2]);
  t_sample  *in2 =    (t_sample *)(w[3]);
  t_sample  *out =    (t_sample *)(w[4]);
  int          n =           (int)(w[5]);

  processWorker(x, in1, in2);

  while (n--) *out++ = x->buffer[x->bufferReadHead++];


  x->currentCondition = x->currentCondition % PD_BLOCK_SIZE;
  x->bufferReadHead = x->bufferReadHead % (3 * PD_BLOCK_SIZE);


  return (w+6);
}

void ddsp_tilde_dsp(t_ddsp_tilde *x, t_signal **sp)
{
  dsp_add(ddsp_tilde_perform, 5, x,
          sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_n);
}

void ddsp_tilde_free(t_ddsp_tilde *x)
{
  free(x->buffer);
  free(x->ddsp);
  inlet_free(x->x_in2);
  outlet_free(x->x_out);
}

void *ddsp_tilde_new(t_floatarg f)
{
  t_ddsp_tilde *x = (t_ddsp_tilde *)pd_new(ddsp_tilde_class);

  x->buffer = (float *) malloc(3 * PD_BLOCK_SIZE * sizeof(float));
  for (int i(0); i<3*PD_BLOCK_SIZE; i++) x->buffer[i] = 0;

  x->phase = 0;
  x->f0    = 0;
  x->lo    = 0;

  x->bufferReadHead  = 0;
  x->bufferWriteHead = PD_BLOCK_SIZE;
  x->currentCondition   = 0;

  x->ddsp = new DDSPCore();

  x->x_in2=inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);

  x->x_out=outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

extern "C" {
void ddsp_tilde_setup(void) {
  ddsp_tilde_class = class_new(gensym("ddsp~"),
        (t_newmethod)ddsp_tilde_new,
        0, sizeof(t_ddsp_tilde),
        CLASS_DEFAULT,
        A_DEFFLOAT, 0);

  class_addmethod(ddsp_tilde_class,
        (t_method)ddsp_tilde_dsp, gensym("dsp"), A_CANT, 0);

  CLASS_MAINSIGNALIN(ddsp_tilde_class, t_ddsp_tilde, f);
}
}
