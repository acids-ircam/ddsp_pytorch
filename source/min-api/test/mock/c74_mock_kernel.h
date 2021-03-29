/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

/**	A symbol table, mapping strings to their #t_symbol counterparts.
    @ingroup	kernel
 */
typedef std::unordered_map<std::string, t_symbol>	t_mock_symboltable;
typedef t_mock_symboltable::iterator				t_mock_symboliter;


/**	Global symbol table instance.
    Used by gensym().

    @ingroup	kernel
    @remark		Ideally the symbol table would not be global.
                Intead you would pass a reference to the symbol table into gensym() as the first argument.
                That would allow us to start each test, for example, completely in control of symbol table construction and state.
                The gensym() calls then would then become a macro that passed in the correct symbol table reference.
                Perhaps the symbol table should be a member of a "microkernel" object so that we can easily construct and tear-down
                an entire mock environment easily.
 */
static t_mock_symboltable	mock_symboltable;


/** Lookup a symbol by its string and, if needed, generate an entry for it in the global symbol table.
    @ingroup	kernel
    @param		string	the string whose #t_symbol will be returned
    @return				a pointer to a #t_symbol

    @remark		This implementation is potentially flawed.
                We are returning pointers to t_symbols in a std container.
                In the case of a vector the pointer could become invalid because of memory reallocation and copying.
                This is not a vector, it's an unordered_map... should research the safety of relying on pointers to items in the container.
                The alternative is to do as we do with mock outlets where we return an id rather than a raw pointer.
 */
MOCK_EXPORT t_symbol* gensym(const char* string)
{
    std::string			s(string);
    t_mock_symboliter	i = mock_symboltable.find(s);

    if (i != mock_symboltable.end())
        return (t_symbol*)&i->second;
    else {
        t_symbol	symbol;

        symbol.s_name = new char[s.size()+1];
        symbol.s_thing = 0;
        strncpy((char*)symbol.s_name, s.c_str(), s.size()+1);

        mock_symboltable[s] = symbol;
        return &mock_symboltable[s];
    }
}


MOCK_EXPORT t_symbol* gensym_tr(const char* string) {
    return gensym(string);
}


}} // namespace c74::max
