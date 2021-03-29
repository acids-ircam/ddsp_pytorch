# `cmakepp` Expression Syntax

## <a name="expr_motivation"></a> Motivation

"CMake's syntax is not sexy." Is a statement that probably everyone can understand. It does not allow the developer rudimentary constructs which almost all other languages have.  But because CMake's language is astonishingly flexible I was able to create a lexer and parser and interpreter for a custom 100 % compatible syntax which (once "JIT compiled") is also fast. Which I call `cmakepp expressions` or `expr` for short. I want to emphasise that this all is written in 100% compatible cmake and you can use it *without* the need to preprocess cmake files.

## <a name="expr_example"></a> Example

The easiest way to introduce the new `expr` syntax and its usefulness is by example. So here it is (Note that these example are actually executed during the generation of this document using my cmake template engine ([link](#)) so the result displayed is actually computed).

You can run these examples yourself - just look at the [installing cmakepp](#) section in `cmakepp`'s documentation.  (Which boils down to downloading a single file and including it)

```cmake

## results of the evaluation are printed as normal strings or
## as json if the result contains an object (complex data)

# values
## number
expr(1) # => `1`

## bool
expr(true) # => `true`
expr(false) # => `false`

## null
expr(null) # => ``

## single quoted string
expr("'hello world'") # => `hello world`
expr("'null'") # => `null`

## double quoted string
expr("\"hello world\"") # => `hello world`

## separated string (a string which is a single argument)
expr("hello world") # => `hello world`

## unquoted
expr(hello)  # => `hello`

## list
expr([1,2,3])  # => `1;2;3`

## object
expr({first_name:Tobias, 'last_name': "Becker" })  # => {"first_name":"Tobias","last_name":"Becker"}

## complex value
expr({value:[1,{a:2,b:{c:[3,4,5],d:2}},{}]})  # => {"value":[1,{"a":2,"b":{"c":[3,4,5],"d":2}},{}]}

## concatenation
expr(hello world 1 null 2 'null' 3) # => `helloworld12null3`


# scope variables
## variable
set(my_var "hello world")
expr($my_var) # => `hello world`

## variable scope variable
set(my_var "hello world")
set(var_name my_var)
expr($($var_name))  # => `hello world`

## assign scope variable
expr($a = 1)
assert("${a}" EQUAL 1)  # => `1`

# function
## test function
function(my_add lhs rhs)
    math(EXPR sum "${lhs} + ${rhs}")
    return("${sum}")
endfunction()
## call normal cmake function
expr(my_add(1,2))  # => `3`

## use function invocation as argument (`f(g(x))`)
expr(my_add(my_add(my_add(1,2),3),4)) # => `10`

## use bind call which inserts the left hand value into a function as the first parameter
expr("hello there this is my new title"::string_to_title()) # => `Hello There This Is My New Title`

## **enabling cmakepp expressions for the current function or file scope**
## All expressions before the call to `cmakepp_enable_expressions` are not
## evaluated as expressions.  All code after the call to
## `cmakepp_enable_expressions` is evaluated.  (also sub functions, but not includes)
## This is a very hacky approach beacuse it works by parsing
## the and transpiling current file. It works in standard cases. In complex
## cases I suggest you rather define a new function which uses
## the `arguments_expression` macro which is more consistent in behaviour.
## Please consult the documentation of `cmakepp_enable_expressions` before
## using it.
function(my_test_function)
    ## here expressions ARE NOT evaluated yet
    set(result "unevaluated $[hello world]")
    ## this call causes cmakepp expressions to be evaluated for
    ## the rest of the current function body
    cmakepp_enable_expressions(${CMAKE_CURRENT_LIST_LINE})
    ## here expressions ARE being evaluated
    set(result "${result} evaluated: $[hello world]")
    return(${result})
endfunction()
my_test_function() # => `unevaluated $[hello world] evaluated: helloworld`


## defining a cmake function which parses it parameters using `expr`
## This a replacement for `cmake_parse_arguments` which is more mighty and
## also very easy to use
##
function(my_other_test_function) ## note: no arguments defined
    ## 0 ${ARGC} specify the range of arguments of `my_other_test_function`
    ## which are to be parsed.  The normal case is `0 ${ARGC}` however you can
    ## also choose to mix cmake argument handling and expression arguments
    ## here you can also specify arguments which are extracted
    arguments_expression(0 ${ARGC} var_1 var_2 var_3)

    ## all arguments which are left unparsed are returned in encoded list
    ## format and need to be `encoded_list_decode`'d before they can be used
    ans(unused_arguments)

    set(result  "result of arguments_expression macro: \n")

    set(counter 0)
    foreach(unused_arg ${unused_arguments})
        math(EXPR counter "${counter} + 1")
        encoded_list_decode("${unused_arg}")
        ans(decoded_unused_arg)
        set(result "${result}unused_arg[${counter}]: '${decoded_unused_arg}'\n")
    endforeach()

    ## defined arguments are available as scope variables
    set(result "${result}var_1: ${var_1}\n")
    set(result "${result}var_2: ${var_2}\n")
    set(result "${result}var_3: ${var_3}\n")

    ## named arguments which are not defined are not
    ## imported into the functions scope. They are still
    ## accessible through `arguments_expression_result` (a map)
    format("var_4: {arguments_expression_result.var_4}\nvar_5:{arguments_expression_result.var_5}\n")
    ans(formatted)
    set(result "${result}${formatted}")

    ## return the string
    return_ref(result)
endfunction()
## Explanation of function call:
## the first argument `[a,eval_math('3*5'),c]` is treated as a single list and
## put into `var_1`.
## the second argument is spread out (because of ellipsis `...` operation)
## and therfore only the first value lands in `var_3`
## named vars have a higher precedence than positional vars therefore `var_2`
## contains `hello world`
## unused named vars are ignored but can be still be accessed using the
## `arguments_expression_result` variable which is automatically defined
## after calling the `arguments_expression` macro
## please note that commas are important
my_other_test_function(
    [a,eval_math('3*5'),c],
    [d,e]...,
    var_2: "hello world" ,
    f,[g,h],
    var_4: "another value"
)  # => `result of arguments_expression macro: 
unused_arg[1]: 'e'
unused_arg[2]: 'f'
unused_arg[3]: 'g;h'
var_1: a;15;c
var_2: hello world
var_3: d
var_4: another value
var_5:
`
```

## <a name="expr_functions"></a> FunctionsandDatatypes

I provide the following functions for you to interact with `expr`.



* [expr](#expr)
* [arguments_expression](#arguments_expression)
* [cmakepp_enable_expressions](#cmakepp_enable_expressions)

## <a name="expr"></a> `expr`

 `(<expression>)-><any>`

 parses, compiles and evaluates the specified expression. The compilation result
 is cached (per cmake run)





## <a name="arguments_expression"></a> `arguments_expression`

 `(<begin index> <end index> <parameter definition>...)-><any>...`


 parses the arguments of the calling function cmakepp expressions
 expects `begin index` to be the index of first function parameters (commonly 0)
 and `end index` to be the index of the last function parameter to parse (commonly ${ARGC})
 var args are named arguments which will be set to be available in the function scope

 named arguments passed to function have a higher precedence than positional arguments 

 __sideffects__:
 * `arguments_expression_result` is a address of an object containing all parsed data
 * scope operations may modify the parent scope of the function
 





## <a name="cmakepp_enable_expressions"></a> `cmakepp_enable_expressions`

 `(${CMAKE_CURRENT_LIST_LINE})-><any>`

 you need to pass `${CMAKE_CURRENT_LIST_LINE}` for this to work

 this macro enables all expressions in the current scope
 it will only work in a CMake file scioe or inside a cmake function scope.
 You CANNOT use it in a loop, if statement, macro etc (everything that has a begin/end)
 Every expression inside that scope (and its subscopes) will be evaluated.  

 **Implementation Note**:
 This is achieved by parsing the while cmake file (and thus potentially takes very long)
 Afterwards the line which you pass as an argument is used to find the location of this macro
 every argument for every following expression in the current code scope is scanned for
 `$[...]` brackets which are in turn lexed,parsed and compiled (see `expr()`) and injected
 into the code which is in turn included






## <a name="expr_definition"></a> TheExpressionLanguageDefinition

I mixed several constructs and concepts from different languages which I like - the syntax should be familiar for someone who knows JavaScript and C++.  I am not a professional language designer so some decisions might seem strange however I have tested everything thouroughly and am happy to fix any bugs.

For better or worse here is the language definition which is still a work in progress.

```cmake
## the forbidden chars are used by the tokenizer and using them will cause the world to explode. They are all valid ASCII codes < 32 and > 0 
<forbidden char> ::=  SOH | NAK | US | STX | ETX | GS | FS | SO  
<escapeable char> :: = "\" """ "'"  
<free char> ::= <ascii char>  /  <forbidden char> / <escapablechar> 
<escaped char> 
<quoted string content> ::= <<char>|"\" <escaped char>>* 

## strings 
<double quoted string> ::= """ <quoted string content> """
    expr("\"\"") # => ``
    expr("\"hello\"") # => `hello`
    expr("\"\\' single quote\"") # => `' single quote`
    expr("\"\\\" double quote\"") # => `" double quote`
    expr("\"\\ backslash\"") # => `\ backslash`

<single quoted string> ::= "'" <quoted string content> "'"
    expr("''") # => ``
    expr('') # => ``
    expr("'hello'") # => `hello`
    expr('hello') # => `hello`
    expr("'hello world'") # => `hello world`
    expr("'\\' single quote'") # => `' single quote`
    expr("'\\\" double quote'") # => `" double quote`
    expr("'\\\\ backslash'") # => `\\ backslash`

<unquoted string> ::= 
    expr(hello) # => `hello`

<separated string> ::= 
    expr("") # => ``
    expr("hello world") # => `hello world`
    

<string> ::= <double quoted string> | <single quoted string> | <unquoted string> | <separated string>

## every literal is a const string 
## some unquoted strings have a special meaning
## quoted strings are always strings even if their content matches one
## of the specialized definitions
<number> ::= /0|[1-9][0-9]*/
    number
        expr(0) # => `0`
        expr(1) # => `1`
        expr(912930) # => `912930`
    NOT number:
        expr(01) # => `01`
        expr('1') # => `1`
        expr("\"1\"") # => `1`

<bool> ::= "true" | "false"
    bool
        expr(true) # => `true`
        expr(false) # => `false`
    NOT bool
        expr('true') # => `true`
        expr(\"false\") # => `false`

<null> ::= "null"
    null
        expr(null) # => ``
    NOT null
        expr('null') # => `null`

<literal> ::= <string>|<number>|<bool>|<null>
    valid literal
        expr("hello world") # => `hello world`
        expr(123) # => `123`
        expr('123') # => `123`
        expr(true) # => `true`
        expr(null) # => ``
        expr("") # => ``
        expr(abc) # => `abc`

<list> ::= "[" <empty> | (<value> ",")* <value> "]"
    expr("[1,2,3,4]") # => `1;2;3;4`
    expr("[1]") # => `1`
    expr("[string_to_title('hello world'), 'goodbye world'::string_to_title()]") # => `Hello World;Goodbye World`

<key value> ::= <value>:<value>
<object property> ::= <key value> | <value>
<object> ::= "{" <empty> | (<object property> ",")* <object property> "}"
    expr({}) # => {} 
    expr({a:b}) # => {"a":"b"} 
    expr({a: ("hello world"::string_to_title())}) # => {"a":"Hello World"} 

<paren> ::= "(" <value> ")"
## ellipsis are treated differently in many situation
## if used in a function parameter the arguments will be spread out
## if used in a navigation or indexation expression the navigation or
## indexation applied to every element in value
<ellipsis> ::= <value> "..." 

## if lvalue is <null> it is set to the default value (a new map)
<default value> ::= <lvalue> "!" 

<value> ::= <paren> |
            | <value dereference>
            | <value reference>
            | <ellipsis>
            | <literal>
            | <interpolation>
            | <bind call>
            | <call>
            | <list>
            | <object>
            | <scope variable>
            | <indexation>
            | <navigation>
            | <paren>
            | <default value>

## a parameter can be a value or the returnvalue operator "$"
## if the return value operator is specified the output of that parameter
## is treated as the result of the function call 
## this is usefull for using function which do not adhere to the
## return value convention. If used multiple times inside a call
## the results are accumulated.   
<parameter> ::= <value> | "$" 

<parameters> ::= "(" <empty> | (<parameter> "," )* <parameter>  ")"

<normal call> ::= <value> <parameters>
    expr(string_to_title("hello world")) # => `Hello World` 
    expr("eval_math('3 + 4')") # => `7`

## bind call binds the left hand value as the first parameter of the function
<bind call> ::=  <value> "::" <value> <parameter>  
    expr("hello world"::string_to_title()) # => `Hello World` 

<call> ::= <normal call> | <bind call>

<scope variable> ::= "$" <literal> 
    set(my_var hello!) 
    expr($my_var) # => `hello!` 


<index> ::= "-"? <number> | "$" | "n"
<increment> ::= "-"? <number>
<range> ::= <empty> | <index> | <index> ":" <index> | <index> ":" <index> : <increment>

<indexation parameter> ::= <range> | <value>
<indexation parameters> ::= (<indexation parameter> ",")* <indexation parameter>
<indexation> ::= <value>"[" <indexation parameters> "]"
    expr("{a:1,b:2}['a','b']") # => `1;2` 
    expr("[{a:1,b:2},{a:3,b:4}]...['a','b']") # => `1;3;2;4`

<lvalue> ::= <scope variable> | <navigation> | <indexation> 


<assign> ::= <lvalue> "=" <value>
    ## sets the scope variable my_value to 1
    expr($my_value = 1)
    expr($my_value) # => `1` 
    ## coerces the scope value my_other_value to be an object
    #expr($my_other_value!.a!.b = 123)
    #expr($my_other_value) # => `` 


<expression> ::= <assign> | <value> 

```

## Caveats 

### Syntax
The syntax is not implemented completely yet (missing return value token `$`).  There are some conflicts in operation precedence (which can currently be solved with parenthesis).  All Examples described here work however.


### Speed
Compiling these expressions is slow: a simple expression compiles in hundreds of milliseconds. Complex expressions can take several seconds.  However this is only when they are first encountered. afterwards the speed is fast even if the expressions are complex. Currently I cache the expressions everytime cmake starts. An example is when using `arguments_expression(...)` It takes up to 1000ms to compile 

```cmake
my_other_test_function(
    [a,eval_math('3*5'),c],
    [d,e]...,
    var_2: "hello world" ,
    f,[g,h],
    var_4: "another value"
)
```
however execution time is less than 10 ms 


## Future Work

### Syntax

I still need to implement correct math in the expressions which is a big minus in cmake (anyone who every had to work with negative values knows what I mean).  Expressions like `(-$a + 33) / -12 ` should be simple enough to get to work.  

The return token  `$` needs to work. currently I do not support it.

Range based get and set needs to be implemented.   
```
# let a = [{b:'a'},{b:'b'},{b:'c'},{b:'d'}]
# then 
$a[1:2].b = [1,2]... 
## should result in 
# let a = [{b:'a'},{b:'1'},{b:'2'},{b:'d'}]
```


### Speed

I still need to cache expressions between cmake runs which will decrease the time needed by alot (as expressions only need to be reevaluated whenever it changes)

### Afterwards

When the syntax is complete and this feature works well the next step is to incorporate it into CMake using C code which will significantly increase the speed.  This will make everything much, much faster and might even get rid of those hideous generator expressions. 