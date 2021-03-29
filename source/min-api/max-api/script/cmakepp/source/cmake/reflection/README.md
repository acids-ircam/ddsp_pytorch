# `CMake` Script Parsing, Token Manipulation, Reflection

CMake's language lends itself to be parsed and tokenized easily. The [specification](http://www.cmake.org/cmake/help/v3.0/manual/cmake-language.7.html#syntax) shows which tokens are available.  Since the functional blocks of CMake only are `command_invocations`s the language's structure very simple.  

My parser takes any cmake code (the version of code supported is still to be determined as for example bracket arguments are not supported but bracket comments are) and creates a list of tokens.  These tokens are also part of a linked list.  This linked list can be used to modify token values or add new tokens which in essence allows reflection and manipulation of the source code.

```
<cmake token>: {
    type:<token type>
    value: <string>
    literal_value: <string>
    line: <uint>
    column: <uint>
    length: <uint>
    *next :<cmake token>
    *previous: <cmake token>
}
<nesting token> ::= <cmake token> v {
    type: "nesting",
    value: "(",
    literal_value: "(",
    end: <nesting end token>
}
<nesting end token> ::= <cmake token> v {
    type: "nesting_end",
    value: ")",
    literal_value: ")",
    begin: <nesting begin token>
}
```






## Function List


* [cmake_token_range](#cmake_token_range)
* [cmake_tokens](#cmake_tokens)
* [cmake_tokens_parse](#cmake_tokens_parse)
* [cmake_token_range_serialize](#cmake_token_range_serialize)
* [cmake_token_range_to_list](#cmake_token_range_to_list)
* [cmake_token_advance](#cmake_token_advance)
* [cmake_token_go_back](#cmake_token_go_back)
* [cmake_token_range_filter](#cmake_token_range_filter)
* [cmake_token_range_filter_values](#cmake_token_range_filter_values)
* [cmake_token_range_insert](#cmake_token_range_insert)
* [cmake_token_range_remove](#cmake_token_range_remove)
* [cmake_token_range_replace](#cmake_token_range_replace)
* [cmake_invocation_filter_token_range](#cmake_invocation_filter_token_range)
* [cmake_invocation_get_arguments_range](#cmake_invocation_get_arguments_range)
* [cmake_invocation_remove](#cmake_invocation_remove)
* [cmake_invocation_token_set_arguments](#cmake_invocation_token_set_arguments)

## Function Descriptions

## <a name="cmake_token_range"></a> `cmake_token_range`

 `(<cmake token range>|<cmake token>...|<cmake code>)-><cmake token range>`

 coerces the input to become a token range 
 if the input already is a token range it is returned
 if the input is a list of tokens the token range will be extracted
 if the input is a string it is assumed to be cmake code and parsed to return a token range




## <a name="cmake_tokens"></a> `cmake_tokens`

 `(<cmake code>|<cmake token>...)-><cmake token>...`

 coerces the input to a token list




## <a name="cmake_tokens_parse"></a> `cmake_tokens_parse`

 `(<cmake code> [--extended])-><cmake token>...`

 this function parses cmake code and returns a list linked list of tokens 

 ```
 <token> ::= { 
  type: "command_invocation"|"bracket_comment"|"line_comment"|"quoted_argument"|"unquoted_argument"|"nesting"|"nesting_end"|"file"
  value: <string> the actual string as is in the source code 
  [literal_value : <string>] # the value which actually is meant (e.g. "asd" -> asd  | # I am A comment -> ' I am A comment')
  next: <token>
  previous: <token>
 }
 <nesting token> ::= <token> v {
   "begin"|"end": <nesting token>
 }
 <extended token> ::= (<token>|<nesting token>) v {
  line:<uint> # the line in which the token is found
  column: <uint> # the column in which the token starts
  length: <uint> # the length of the token 
 }
 ```




## <a name="cmake_token_range_serialize"></a> `cmake_token_range_serialize`

 `(<start:<cmake token>> <end:<cmake token>>?)-><cmake code>`
 
 generates the cmake code corresponding to the cmake token range




## <a name="cmake_token_range_to_list"></a> `cmake_token_range_to_list`

 `(<start:<cmake token>> [<end: <cmake token>])-><cmake token>...`

 returns all tokens for the specified range (or the end of the tokens)




## <a name="cmake_token_advance"></a> `cmake_token_advance`

 `(<&<token>>)-><token>`

 advances the current token to the next token




## <a name="cmake_token_go_back"></a> `cmake_token_go_back`

 `(<&cmake token>)-><cmake token>`
 
 the token ref contains the previous token after invocation




## <a name="cmake_token_range_filter"></a> `cmake_token_range_filter`

 `(<cmake token range> <predicate> [--reverse] [--skip <uint>] [--take <uint>])-><cmake token>...`

 filters the specified token range for tokens matching the predicate (access to value and type)
 e.g. `cmake_token_range_filter("set(a b c d)" type MATCHES "^argument$" AND value MATCHES "[abd]" --reverse --skip 1 --take 1 )` 
 
 




## <a name="cmake_token_range_filter_values"></a> `cmake_token_range_filter_values`

 `(...)->...` 

 convenience function
 same as cmake_token_range_filter however returns the token values




## <a name="cmake_token_range_insert"></a> `cmake_token_range_insert`

 `(<where:<cmake token>> <cmake token range> )-><token range>`

 inserts the specified token range before <where>




## <a name="cmake_token_range_remove"></a> `cmake_token_range_remove`

 `(<cmake token range>)-><void>`

 removes the specified token range from the linked list




## <a name="cmake_token_range_replace"></a> `cmake_token_range_replace`

 `(<range:<cmake token range>> <replace_range:<cmake token range>>)-><cmake token range>`
 
 replaces the specified range with the specified replace range
 returns the replace range




## <a name="cmake_invocation_filter_token_range"></a> `cmake_invocation_filter_token_range`

 `(<cmake token range> <predicate> [--skip <uint>] [--take <uint>] [--reverse])-><cmake invocation>...`

 searches for invocations matching the predicate allowing to skip and take a certain amount of matches
 also allows reverse serach when specifying the corresponding flag.

 the predicate is the same as what one would write into an if clause allows access to the following variables:
 * invocation_identifier
 * invocation_arguments
 e.g. `invocation_identifier MATCHES "^add_.*$"` would return only invocations starting with add_
 also see `eval_predicate`
 ```
 <cmake invocation> ::= {
    invocation_identifier: <string>      # the name of the invocation
    invocation_arguments: <string>...    # the arguments of the invocation
    invocation_token: <cmake token>      # the token representing the invocation
    arguments_begin_token: <cmake token> # the begin of the arguments of the invocation (after the opening parenthesis)
    arguments_end_token: <cmake token>   # the end of the arguments of the invocation (the closing parenthesis)
 }
 ```





## <a name="cmake_invocation_get_arguments_range"></a> `cmake_invocation_get_arguments_range`

 `(<invocation:<command invocation>>)->[<start:<token>> <end:<token>>]`
 
 returns the token range of the invocations arguments given an invocation token




## <a name="cmake_invocation_remove"></a> `cmake_invocation_remove`

 `(<cmake invocation>)-><void>`

 removes the specified invocation from its context by removing the invocation token and the arguments from the linked list that they are part




## <a name="cmake_invocation_token_set_arguments"></a> `cmake_invocation_token_set_arguments`

 `(<command invocation token> <values: <any...>>)-><void>`

 replaces the arguments for the specified invocation by the
 specified values. The values are quoted if necessary







