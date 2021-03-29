#include "m_pd.h"

static t_class *pan_tilde_class;

typedef struct _pan_tilde
{
  t_object x_obj;
  t_sample f_pan;
  t_sample f;

  t_inlet *x_in2;
  t_inlet *x_in3;
  t_outlet *x_out;
} t_pan_tilde;

t_int *pan_tilde_perform(t_int *w)
{
  t_pan_tilde *x = (t_pan_tilde *)(w[1]);
  t_sample *in1 = (t_sample *)(w[2]);
  t_sample *in2 = (t_sample *)(w[3]);
  t_sample *out = (t_sample *)(w[4]);
  int n = (int)(w[5]);
  t_sample f_pan = (x->f_pan < 0) ? 0.0 : (x->f_pan > 1) ? 1.0 : x->f_pan;

  while (n--)
    *out++ = (*in1++) * (1 - f_pan) + (*in2++) * f_pan;

  return (w + 6);
}

void pan_tilde_dsp(t_pan_tilde *x, t_signal **sp)
{
  dsp_add(pan_tilde_perform, 5, x,
          sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_n);
}

void pan_tilde_free(t_pan_tilde *x)
{
  inlet_free(x->x_in2);
  inlet_free(x->x_in3);
  outlet_free(x->x_out);
}

void *pan_tilde_new(t_floatarg f)
{
  t_pan_tilde *x = (t_pan_tilde *)pd_new(pan_tilde_class);

  x->f_pan = f;

  x->x_in2 = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  x->x_in3 = floatinlet_new(&x->x_obj, &x->f_pan);
  x->x_out = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

void pan_tilde_setup(void)
{
  pan_tilde_class = class_new(gensym("pan~"),
                              (t_newmethod)pan_tilde_new,
                              0, sizeof(t_pan_tilde),
                              CLASS_DEFAULT,
                              A_DEFFLOAT, 0);

  class_addmethod(pan_tilde_class,
                  (t_method)pan_tilde_dsp, gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(pan_tilde_class, t_pan_tilde, f);
}
