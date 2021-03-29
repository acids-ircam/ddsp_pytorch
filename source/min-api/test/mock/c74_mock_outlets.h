/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_mock_atoms.h"

namespace c74 {
namespace max {


/**	Internal implementation of mock-outlets.
    We use these outlets instead of the normal Max outlets so that we can intercept, log, and examine the messages in a testing context.
    We also use these outlets because the entire Max kernel is not available to us in the minimal testing context.
    @ingroup outlets
 */
class t_mock_outlet {
    void		*m_owner;		///< pointer the max object owning the inlet/outlet
    t_sequence	m_messages;		///< sequence of data sent to the inlet/outlet
    char		m_type[256];	///< outlets: message name that sent (for type-checked outlets)
    t_ptr_int	m_id;			///< id is used to identify each outlet -- we use this instead of a pointer

    static int	m_id_counter;	///< used to generate globally unique values for m_id

public:

    /** Constructor, as wrapped by outlet_new() */
    t_mock_outlet(void *x, const char *type):
    m_owner(x), m_id(0)
    {
        m_type[0] = 0;
        if (type)
            strncpy(m_type, type, 256);
        m_id_counter++;
        m_id = m_id_counter;
    }

    /** Copy constructor.  Essential because we store instances in std container classes. */
    t_mock_outlet(const t_mock_outlet& source):
    m_owner(source.m_owner), m_id(source.m_id)
    {
        strncpy(m_type, source.m_type, 256);
    }


    /**	Request the globally unique id of this outlet. */
    t_ptr_int get_id()
    {
        return m_id;
    }


    /**	Request a reference to the sequence of messages sent to this outlet. */
    t_sequence& get_messages()
    {
        return m_messages;
    }


    /**	Push a new message to the outlet. */
    void push(t_atom_long value)
    {
        t_atom_vector av;

        av.resize(2);
        atom_setsym(&av[0], gensym("int"));
        atom_setlong(&av[1], value);
        m_messages.push_back(av);
    }

    /**	Push a new message to the outlet. */
    void push(t_atom_float value)
    {
        t_atom_vector av;

        av.resize(2);
        atom_setsym(&av[0], gensym("float"));
        atom_setfloat(&av[1], value);
        m_messages.push_back(av);
    }

    /**	Push a new message to the outlet. */
    void push(t_symbol *name, long argc, const t_atom *argv)
    {
        t_atom_vector av;

        if (!strcmp(name->s_name, "list")) {
            av.resize(argc);
            for (long i=0; i<argc; i++)
                av[i] = argv[i];
        }
        else {
            av.resize(argc+1);
            atom_setsym(&av[0],name);
            for (long i=0; i<argc; i++)
                av[i+1] = argv[i];
        }
        m_messages.push_back(av);
    }
};

int t_mock_outlet::m_id_counter = 0;


/**	A set of outlets which will belong to a particular #t_object.
    @ingroup outlets
 */
typedef std::vector<t_mock_outlet>	t_mock_outlets;
typedef t_mock_outlets::iterator	t_mock_outlet_iter;


/**	Associates a #t_object with its #t_mock_outlets
    @ingroup outlets
 */
typedef std::unordered_map<void*, t_mock_outlets>	t_map_object_2_outlet_set;
t_map_object_2_outlet_set g_object_to_outletset;


/**	Get a reference to an outlet.

    @ingroup	outlets
    @param o	the #t_object whose outlet reference you want
    @param i	the index (zero-based) of the outlet reference you want
    @return		a reference to an outlet
 */
MOCK_EXPORT_CPP t_mock_outlet& object_getoutlet(void *o, int i) {
    t_mock_outlets& mock_outlets = g_object_to_outletset[o];
    return mock_outlets[i];
}


/**	Get a reference to an outlet's sequence.
    The sequence is the 'log' of all messages sent to the outlet.

    @ingroup	outlets
    @param o	the #t_object whose outlet sequence you want
    @param i	the index (zero-based) of the outlet whose sequence you want
    @return		a reference to the sequence
 */
MOCK_EXPORT t_sequence* object_getoutput(void *o, int outletnum) {
    t_mock_outlet& outlet = object_getoutlet(o, outletnum);
    return &outlet.get_messages();
}


/**	Create a new outlet.
    This mocks the behavior of Max's real outlet_new().

    @ingroup	outlets
    @param x	a pointer to the #t_object that grow a new outlet
    @param s	an optional message-type specifier for type-checking
    @return		an outlet id -- THIS IS DIFFERENT THAN IN MAX -- MAX RETURNS A POINTER TO THE OUTLET ITSELF

    @remark		We create the outlet on the stack and then copy it into the vector.
                To look up based on outlet id (not pointer, because the pointer could change when the vector is updated) we need to get the set of outlets for this object.
                We cannot keep this in the #t_object's o_outlet member because this is not accessible from the outlet_float() etc. calls.
                Thus all outlets are assigned a globally unique id and the outlet sets are then stored hash keyed with the object pointer.

    @remark		For an API improvement in the future we should have new outlet calls, e.g. object_outlet_float() which take an object pointer.
 */
MOCK_EXPORT void *outlet_new(void *x, const char *s)
{
    t_mock_outlets& mock_outlets = g_object_to_outletset[x];
    t_mock_outlet	outlet(x, s);

    mock_outlets.insert(mock_outlets.begin(), outlet);
    return (void*)outlet.get_id();
}


/**	(Internal) Send a value to an outlet.

    @ingroup	outlets
    @param		outlet_id	The id of an outlet as returned by our mock version of outlet_new()
    @param		value		The value (float or int) to send
    @return					In Max it should 0 if stackoverflow occurred?  We return 1 to indicate success?

    @remark		As mentioned in the documentation for outlet_new(), there are some vagaries to work around because outlet calls are not scoped to the owning #t_object.
                So because outlet_float() doesn't know its #t_object like a potential object_outlet_float() would,
                we have to search all outlets for all objects in order to send a message.
                This would clearly represent a performance problem in a real environment, but for our simple mock testing environment it should be tolerable.

    @remark		The search method currently employed is to iterate linearly through every object to get its outlet set.
                There is no way to know to which object an outlet id belongs, so we have to just search until we find it.
                Within an outlet set we also currently just iterate sequentially.
                We could clearly be smarter about this, using a binary search or other mechanism.
                However we don't usually have that many outlets so saving this for a rainy day...

    @seealso	outlet_anything()
 */
template <class T>
void *outlet_single(t_ptr_int outlet_id, T value)
{
    for (t_map_object_2_outlet_set::iterator i = g_object_to_outletset.begin(); i != g_object_to_outletset.end(); i++) {
        t_mock_outlets& outletset = i->second;

        for (t_mock_outlet_iter j = outletset.begin(); j != outletset.end(); j++) {
            t_mock_outlet& outlet = *j;

            if (outlet.get_id() == outlet_id) {
                outlet.push(value);
                goto done;
            }
        }
    }
done:
    return (void*)1;
}


/**	Send a message to an outlet.
    This mocks the behavior of Max's real function by this same name.

    @ingroup		outlets
    @param x		the outlet's id -- THIS IS DIFFERENT THAN IN MAX, WHICH EXPECTS A POINTER TO THE OUTLET ITSELF
    @param value	the value to send
    @return			usually ignored
 */
MOCK_EXPORT void *outlet_int(void *x, t_atom_long value)
{
    return outlet_single((t_ptr_int)x, value);
}


/**	Send a message to an outlet.
    This mocks the behavior of Max's real function by this same name.

    @ingroup		outlets
    @param x		the outlet's id -- THIS IS DIFFERENT THAN IN MAX, WHICH EXPECTS A POINTER TO THE OUTLET ITSELF
    @param value	the value to send
    @return			usually ignored
 */
MOCK_EXPORT void *outlet_float(void *x, double value)
{
    return outlet_single((t_ptr_int)x, (t_atom_float)value);
}



/**	Send a message to an outlet.
    This mocks the behavior of Max's real function by this same name.

    @ingroup		outlets
    @param x		the outlet's id -- THIS IS DIFFERENT THAN IN MAX, WHICH EXPECTS A POINTER TO THE OUTLET ITSELF
    @param s		name of the message to send
    @param ac		count of atoms in av
    @param av		pointer to the first of an array of atoms
    @return			usually ignored
 */
MOCK_EXPORT void *outlet_anything(void *x, t_symbol *s, short ac, const t_atom *av)
{
    t_ptr_int outlet_id = (t_ptr_int)x;

    for (t_map_object_2_outlet_set::iterator i = g_object_to_outletset.begin(); i != g_object_to_outletset.end(); i++) {
        t_mock_outlets& outletset = i->second;

        for (t_mock_outlet_iter j = outletset.begin(); j != outletset.end(); j++) {
            t_mock_outlet& outlet = *j;

            if (outlet.get_id() == outlet_id) {
                outlet.push(s, ac, av);
                goto done;
            }
        }
    }
done:
    return (void*)1;
}



/**	Send a message to an outlet.
This mocks the behavior of Max's real function by this same name.

@ingroup		outlets
@param x		the outlet's id -- THIS IS DIFFERENT THAN IN MAX, WHICH EXPECTS A POINTER TO THE OUTLET ITSELF
@param s		ignored
@param ac		count of atoms in av
@param av		pointer to the first of an array of atoms
@return			usually ignored
*/
MOCK_EXPORT void *outlet_list(void *x, t_symbol *s, short ac, const t_atom *av)
{
    return outlet_anything(x, gensym("list"), ac, av);
}


}} // namespace c74::max
