/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

MOCK_EXPORT void cpost(const char *fmt, ...);

// for documentation, see implementation below
void object_setinlet(t_object *x, long inletnum);


/**	Internal implementation of mock-inlets.
    We use these inlets instead of the normal Max outlets so that we can intercept, log, and examine the messages in a testing context.
    We also use these inlets because the entire Max kernel is not available to us in the minimal testing context.
    @ingroup inlets
 */
class t_mock_inlet {
    void		*m_owner;		///< pointer the max object owning the inlet
    t_sequence	m_messages;		///< sequence of data sent to the inlet
    long		m_id;			///< proxies: the inlet id number
    long		*m_stuffloc;	///< proxies: struct member where the current inlet is written

public:
    /** Constructor used when resizing an object's vector of inlets. */
    t_mock_inlet():
    m_owner(NULL), m_id(0), m_stuffloc(NULL)
    {}

    /** Constructor, as wrapped by proxy_new().	*/
    t_mock_inlet(void *x, long id, long *stuffloc):
    m_owner(x), m_id(id), m_stuffloc(stuffloc)
    {}

    /** Copy constructor.  Essential because we store instances in std container classes. */
    t_mock_inlet(const t_mock_inlet& source):
    m_owner(source.m_owner), m_id(source.m_id), m_stuffloc(source.m_stuffloc)
    {}


    /**	Push a new message to the inlet. */
    template <class T>
    void *push(T value)
    {
        t_object		*o = (t_object*)m_owner;
        t_mock_messlist *mock_messlist = (t_mock_messlist*)o->o_messlist;
        method			m = NULL;

        if (typeid(value) == typeid(long) || typeid(value) == typeid(int))
            m = (*mock_messlist)["int"];
        else if (typeid(value) == typeid(double) || typeid(value) == typeid(float))
            m = (*mock_messlist)["float"];
        else {
            cpost("OUCH! unknown type pushed to inlet!\n");
            return NULL;
        }

        object_setinlet(o, m_id);
        return (*m)(m_owner, value);
    }

    void *push(const char *value)
    {
        t_object		*o = (t_object*)m_owner;
        t_mock_messlist *mock_messlist = (t_mock_messlist*)o->o_messlist;
        t_symbol		*s = gensym(value);
        method			m = (*mock_messlist)[value];
        t_bool			anything = false;

        if (!m) {
            m = (*mock_messlist)["anything"];
            anything = true;
        }
        if (!m) {
            std::cout << "no method named '" << value << "'" << std::endl;
            return NULL;
        }

        object_setinlet(o, m_id);
        if (anything)
            return (*m)(o, s, 0, NULL);
        else
            return (*m)(o, s);
    }


};


/**	A set of inlets which will belong to a particular #t_object via #object_inlets.
    @ingroup inlets
 */
typedef std::vector<t_mock_inlet>	t_mock_inlets;

/**	A wrapper for an #t_mock_inlets used by #t_object to support getting an inlet number for a message with proxies.
    @ingroup inlets
 */
class object_inlets {
public:
    t_mock_inlets	mock_inlets;
    long			current_inlet;

    /** Constructor gives us one inlet by default.
        @param	x	Pointer to the owning #t_object.
     */
    explicit object_inlets(void* x)
    : current_inlet(0)
    {
        t_mock_inlet inlet(x, 0, NULL);

        mock_inlets.push_back(inlet);
    }


};


/**	Create a new proxy inlet.
    This mocks the behavior of Max's real proxy_new().

    @ingroup			inlets
    @param	x			The object to which to add the proxy inlet.
    @param	id			The index of the inlet to add.
    @param	stuffloc	A pointer to a variable that will be filled-in with the current inlet id when a message is received.
    @remark				In our mock implementation proxies are no different from normal inlets.
                        Proxies are freed with object_free().
                        That's okay though -- in our implementation, proxy_new() always returns NULL and object_free() checks for NULL.
*/
MOCK_EXPORT void *proxy_new(void *x, long id, long *stuffloc)
{
    t_object		*o = (t_object*)x;
    object_inlets	*inlets = (object_inlets*)o->o_inlet;
    t_mock_inlets&	mock_inlets = inlets->mock_inlets;

    t_mock_inlet inlet(x, id, stuffloc);

    if (id+1 > mock_inlets.size())
        mock_inlets.resize(id+1);
    mock_inlets[id] = inlet;
    return NULL;
}


/**	Find out what inlet received the current message.
    This mocks the behavior of Max's real proxy_new().

    @ingroup	inlets
    @param	x	The object instance receiving the message.
    @return		The zero-based index of the inlet that received the message.
 */
MOCK_EXPORT long proxy_getinlet(t_object *x)
{
    object_inlets *inlets = (object_inlets*)x->o_inlet;

    return inlets->current_inlet;
}


/**	(Internal) Set the current inlet id when a message is sent to the object.

    @ingroup			inlets
    @param	x			The object receiving the message.
    @param	inletnum	The id number of the inlet.
 */
void object_setinlet(t_object *x, long inletnum)
{
    t_object		*o = (t_object*)x;
    object_inlets	*inlets = (object_inlets*)o->o_inlet;

    inlets->current_inlet = inletnum;
}


}} // namespace c74::max
