## `(<cmakepp code>)-><cmake code>`
##
## `<cmakepp code> ::= superset of cmake code with added expression syntax in $[...] `
## 
## compiles the specified cmakepp code to pure cmake code
## replacing `$[...]` with the result of the cmakepp expression syntax (see `expr(...)`)
## e.g.
## ```
## function(my_name)
##  return("tobi")
## endfunction()
##  message("hello $[my_name()]::string_to_upper()") # prints `hello TOBI`
## ```
##
function(cmakepp_expr_compile content)
  cmake_token_range(" ${content}")##prepend whitespace (because of replace edgecase)
  ans(range)

  cmake_invocation_filter_token_range("${range}")
  ans(invocations)

  string_codes()
  set(regex "[\\$${bracket_open_code}${bracket_close_code}]|[^\\$${bracket_open_code}${bracket_close_code}]+")

  foreach(invocation ${invocations})
    map_tryget("${invocation}" arguments_begin_token)
    ans(args_begin)
    map_tryget("${invocation}" arguments_end_token)
    ans(args_end)
    map_tryget("${invocation}" invocation_token)
    ans(invocation_token)


    cmake_token_range_filter("${args_begin};${args_end}" NOT type MATCHES "(comment)")
    ans(argument_tokens)


    set(invocation_string)
    foreach(token ${argument_tokens})
      map_tryget("${token}" value)
      ans(value)
      set(invocation_string "${invocation_string}${value}")
    endforeach()
    encoded_list("${invocation_string}")
    ans(invocation_string)



    string(REGEX MATCHALL "${regex}"  tokens "${invocation_string}")
    set(compiled_code)
    set(is_expression 0)
    set(depth 0)
    set(current_arguments)

    while(true)
      set(expression_end false)
      set(expression_begin false)
      list(LENGTH tokens token_count)
      if(NOT token_count)
        break()
      endif()

      list(GET tokens 0 token)
      list(REMOVE_AT tokens 0)

      if("${token}_" STREQUAL "${bracket_open_code}_")
        math(EXPR depth "${depth} + 1")
        
      elseif("${token}_" STREQUAL "${bracket_close_code}_")
        if(${depth} EQUAL ${is_expression})
          set(expression_end true)
          set(is_expression 0)
        endif()
        math(EXPR depth "${depth} - 1")

      elseif("${token}_" STREQUAL "$_" AND NOT is_expression AND ${token_count} GREATER 1)
        list(GET tokens 0 next_token)
        if("${next_token}_" STREQUAL "${bracket_open_code}_")
          list(REMOVE_AT tokens 0)
          math(EXPR depth "${depth} + 1")
          set(is_expression "${depth}")
          set(expression_begin true)
          set(token "$(")
        endif()
      endif()


      if(is_expression AND NOT expression_end AND NOT expression_begin)
        set(current_expression "${current_expression}${token}")
      elseif(NOT expression_end AND NOT expression_begin)
        set(current_arguments "${current_arguments}${token}")
      endif()

      if(expression_begin)
        set(current_expression)
      endif()
      if(expression_end)
         encoded_list_decode("${current_expression}")
         ans(current_code)
         eval("expr_parse(interpret_expression \"\" ${current_code})")
         ans(ast)
         ast_reduce_code("${ast}")
         ans(current_compiled_code)
         map_tryget("${ast}" value)
         ans(value)
         next_id()
         ans(ref)
        
         set(argument_value "\${${ref}}")
         set(compiled_code "${compiled_code}${current_compiled_code}set(${ref} ${value})\n")
         set(current_arguments "${current_arguments}${argument_value}")
      endif()


     endwhile()

       if(compiled_code)
        cmake_token_range_insert("${invocation_token}" "${compiled_code}")
       endif()
       if(current_arguments)
         cmake_token_range_replace("${args_begin};${args_end}" "${current_arguments}")
       endif()

  endforeach()



  cmake_token_range_serialize("${range}")
  ans(result)
  return_ref(result)
endfunction()
