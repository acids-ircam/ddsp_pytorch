/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {


    typedef struct t_pxobject {
        t_object z_ob;	///< The standard #t_object struct.
        long z_in;
        void *z_proxy;
        long z_disabled;	///< set to non-zero if this object is muted (using the pcontrol or mute~ objects)
        short z_count;		///< an array that indicates what inlets/outlets are connected with signals
        short z_misc;		///< flags (bitmask) determining object behaviour, such as #Z_NO_INPLACE, #Z_PUT_FIRST, or #Z_PUT_LAST
    } t_pxobject;


    typedef void(*t_perfroutine64)(t_object *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);





/**	A vector of audio samples (doubles).
    @ingroup msp
 */
typedef std::vector<double>				t_mock_audiovector;

/**	A collection of audio vectors, e.g. audio for multiple inlets where each inlet receives a vector.
    @ingroup msp
 */
typedef std::vector<t_mock_audiovector> t_mock_audiovectors;

/**	A vector of ints.
    Used for the 'count' argument to the dsp method that indicates which inlets/outlets are connected.
    @ingroup msp
 */
typedef std::vector<int>				t_int_vec;


/** Expose t_mock_audiovector for use in std output streams.
    @ingroup msp
 */
template <class charT, class traits>
std::basic_ostream <charT, traits>& operator<< (std::basic_ostream <charT, traits>& stream, const t_mock_audiovector& a)
{
    char str[4096];

    snprintf(str, 4096, "%s", "<audio vector>"); // TODO: something more informative would be nice here...
    return stream << str;
}


/**	An item in the dspchain.
    This is managed by the dspchain itself -- not intended for direct access or use.
    @ingroup msp
 */
class t_mock_dspchain_item {
    t_object			*m_obj;
    t_perfroutine64		m_perf;
    size_t				m_inputcount;
    size_t				m_outputcount;
    size_t				m_framecount;
    double				**m_ins;
    double				**m_outs;

public:
    t_mock_dspchain_item(void *x, t_mock_audiovectors& inputs, t_mock_audiovectors& outputs)
    : m_perf(NULL)
    {
        m_obj = (t_object*)x;
        m_framecount  = inputs[0].size();
        m_inputcount  = inputs.size();
        m_outputcount = outputs.size();
        m_ins  = (double**)malloc(sizeof(double*) * m_inputcount);
        m_outs = (double**)malloc(sizeof(double*) * m_outputcount);

        // adapt the outputs (and additional inputs) dimensions to that of the first signal
        for (int i=1; i<m_inputcount; i++)
            inputs[i].resize(m_framecount);
        for (int i=0; i<m_outputcount; i++)
            outputs[i].resize(m_framecount);

        // inputs and outputs are references to external memory from the t_mock_audiovectors passed-in.
        for (int i=0; i<m_inputcount; i++)
            m_ins[i] = &inputs[i][0];
        for (int i=0; i<m_outputcount; i++)
            m_outs[i] = &outputs[i][0];
    }


    /**	Copy constructor -- essential for the ins and outs when copied about an STL container.	*/
    t_mock_dspchain_item(const t_mock_dspchain_item& other)
    : m_obj(other.m_obj), m_perf(other.m_perf), m_inputcount(other.m_inputcount), m_outputcount(other.m_outputcount), m_framecount(other.m_framecount)
    {
        m_ins  = (double**)malloc(sizeof(double*) * m_inputcount);
        m_outs = (double**)malloc(sizeof(double*) * m_outputcount);

        for (int i=0; i<m_inputcount; i++)
            m_ins[i] = other.m_ins[i];
        for (int i=0; i<m_outputcount; i++)
            m_outs[i] = other.m_outs[i];
    }


    ~t_mock_dspchain_item()
    {
        free(m_ins);
        free(m_outs);
    }

    /**	Execute the perform routine. */
    void tick()
    {
        m_perf(m_obj, NULL, m_ins, static_cast<long>(m_inputcount), m_outs, static_cast<long>(m_outputcount), static_cast<long>(m_framecount), 0, NULL);
    }

    /**	Set the perform method that is to be used for this item. */
    void setperf(t_perfroutine64 perf)
    {
        m_perf = perf;
    }
};


/**	We keep the dspchain items in a vector rather than a list so that they are contiguous in memory.	*/
typedef std::vector<t_mock_dspchain_item>	t_mock_dspchain_items;


/**	The big kahuna.  The mock dspchain.
    When an object is added the chain, the chain calls the 'dsp64' method on that object.
    A callback then sets the perform routine for that object.

    A vector of all items is maintained.
    Executing the dspchain is simply executing every item in that vector.
    Memory for the input and output is entirely handled by the caller!
    That makes this mock implementation simpler, and allows for inspection and full control of the memory in a unit-testing context.
 */
class t_mock_dspchain {
    t_mock_dspchain_items	m_chainitems;	///< all items in the dspchain
    t_mock_dspchain_item*	m_current = nullptr;		///< object currently being added to the dspchain

public:

    /** Add an object to the dspchain.  Calls an object's DSP method
        @param	x				The object to add.
        @param	inputs			The vectors of audio to use as input.
                                The size of the first audio vector will be the vector size used by this object.
        @param	outputs			The vectors of audio to use as output.
        @param	inputs_used		Optional specification of which inlets are connected or not.  Defaults to assuming they are all connected.
        @param	outputs_used	Optional specification of which inlets are connected or not.  Defaults to assuming they are all connected.
     */
    void add(void *x, t_mock_audiovectors& inputs, t_mock_audiovectors& outputs, t_int_vec* inputs_used=NULL, t_int_vec* outputs_used=NULL)
    {
        short					*count;
        int						i = 0;
        double					sr = 44100;				// TODO: use sys_getsr()
        size_t					vs = inputs[0].size();
        t_mock_dspchain_item	item(x, inputs, outputs);
        t_object				*o = (t_object*)x;
        t_mock_inlets			*inlets = (t_mock_inlets*)o->o_inlet;
        size_t					inletcount = inlets->size();

        if (!vs)
            vs = 64;			// TODO: use sys_getblksize()

        // use the number inlets, not the number of inputs -- they can differ
        count = (short*)sysmem_newptrclear(static_cast<long>( sizeof(short) * (inletcount+outputs.size()) ));
        if (inputs_used) {
            for (int j=0; j < inputs_used->size(); j++)
                count[i++] = inputs_used->at(j);
        }
        else {		// assume they are all connected
            for (int j=0; j<inletcount; j++)
                count[i++] = 1;
        }
        if (outputs_used) {
            for (int j=0; j < outputs_used->size(); j++)
                count[i++] = outputs_used->at(j);
        }
        else {		// assume they are all connected
            for (int j=0; j<outputs.size(); j++)
                count[i++] = 1;
        }

        m_current = &item;

        object_method((t_object*)x, gensym("dsp64"), (void*)this, count, (void*)(size_t)sr, (void*)(size_t)vs, 0);

        m_current = NULL;
        m_chainitems.push_back(item);

        free(count);
    }


    /** Callback from the DSP method to define the actual perform routine to use. */
    void add_performroutine(t_object *x, t_perfroutine64 perf)
    {
        m_current->setperf(perf);
    }


    /**	Call the perform methods for every object in the chain. */
    void tick()
    {
        for_each(m_chainitems.begin(), m_chainitems.end(), std::mem_fn(&t_mock_dspchain_item::tick));
    }
};


extern "C" inline void dsp_add64(t_object *dsp64, t_object *x, t_perfroutine64 perf, long flags, void *userparam)
{
    t_mock_dspchain *dspchain = (t_mock_dspchain*)dsp64;

    dspchain->add_performroutine(x, perf);
}


MOCK_EXPORT void class_dspinit(t_class *c)
{
    ;
}


MOCK_EXPORT void z_dsp_free(t_pxobject *x)
{
    ;
}


MOCK_EXPORT void z_dsp_setup(t_pxobject* x, long inletcount)
{
    for (long i=inletcount-1; i>=1; i--)
        proxy_new(x, i, NULL); // we are using mock inlets, so we don't have to track the pointer or free it manually
}


    using t_buffer_obj = t_object;
    using t_buffer_ref = t_object;
    struct t_buffer_info;

    MOCK_EXPORT t_buffer_obj* buffer_ref_getobject(t_buffer_ref* x) {
        return nullptr;
    }

    MOCK_EXPORT float* buffer_locksamples(t_buffer_obj* buffer_object) {
        return nullptr;
    }


    MOCK_EXPORT void buffer_unlocksamples(t_buffer_obj* buffer_object) {
    }


    MOCK_EXPORT t_max_err buffer_getinfo(t_buffer_obj* buffer_object, t_buffer_info* info) {
        return 0;
    }


    MOCK_EXPORT t_max_err buffer_edit_begin(t_buffer_obj* buffer_object) {
        return 0;
    }


    MOCK_EXPORT t_max_err buffer_edit_end(t_buffer_obj* buffer_object, long valid) {
        return 0;
    }

}}
