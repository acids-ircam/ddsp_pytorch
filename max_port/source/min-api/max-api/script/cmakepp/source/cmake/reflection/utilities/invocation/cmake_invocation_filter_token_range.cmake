## `(<cmake token range> <predicate> [--skip <uint>] [--take <uint>] [--reverse])-><cmake invocation>...`
##
## searches for invocations matching the predicate allowing to skip and take a certain amount of matches
## also allows reverse serach when specifying the corresponding flag.
##
## the predicate is the same as what one would write into an if clause allows access to the following variables:
## * invocation_identifier
## * invocation_arguments
## e.g. `invocation_identifier MATCHES "^add_.*$"` would return only invocations starting with add_
## also see `eval_predicate`
## ```
## <cmake invocation> ::= {
##    invocation_identifier: <string>      # the name of the invocation
##    invocation_arguments: <string>...    # the arguments of the invocation
##    invocation_token: <cmake token>      # the token representing the invocation
##    arguments_begin_token: <cmake token> # the begin of the arguments of the invocation (after the opening parenthesis)
##    arguments_end_token: <cmake token>   # the end of the arguments of the invocation (the closing parenthesis)
## }
## ```
##
function(cmake_invocation_filter_token_range range)
  arguments_encoded_list(1 ${ARGC})
  ans(args)

  list_extract_flag_name(args --reverse)
  ans(reverse)

  cmake_token_range("${range}")
  ans(range)
  list_extract(range begin end)
  set(current ${begin})
  list_extract_labelled_keyvalue(args --skip)
  ans(skip)
  list_extract_labelled_keyvalue(args --take)
  ans(take)
  if("${take}_" STREQUAL "_")
    set(take -1)
  endif()


  set(result)
  while(take AND current)
    cmake_token_range_filter("${current};${end}" type STREQUAL "command_invocation" --take 1 ${reverse})
    ans(invocation_token)
    if(NOT invocation_token)
      break()
    endif()
    if(reverse)
      set(end)
    endif()
    
    map_tryget("${invocation_token}" line)
    ans(line)
    cmake_token_range_filter("${invocation_token};${end}" type STREQUAL "nesting" --take 1)
    ans(arguments_begin_token)

    map_tryget(${arguments_begin_token} end)
    ans(arguments_end_token)
    map_tryget(${arguments_end_token} next)
    ans(arguments_after_end_token)

    cmake_token_range_filter_values("${invocation_token};${arguments_after_end_token}" 
      type MATCHES "(command_invocation)|(nesting)|(argument)")
    ans(invocation)



    ## get invocation_identifier and invocation_arguments
    set(invocation_arguments ${invocation})
    list_pop_front(invocation_arguments)
    ans(invocation_identifier)
    list_pop_front(invocation_arguments)
    list_pop_back(invocation_arguments)
    

    list(LENGTH args predicate_exists)
    if(predicate_exists)
      eval_predicate(${args})
      ans(predicate_holds)
    else()
      set(predicate_holds true)
    endif()
    #print_vars(invocation_identifier invocation_arguments predicate_holds)
    #print_vars(invocation_token.type invocation_token.value predicate_holds args)

    ## check if invocation matches the custom predicate
    ## skip and take the specific invocations
    if(predicate_holds)
      if(skip)
        math(EXPR skip "${skip} - 1")
      else()
        cmake_token_advance(arguments_begin_token)
        map_capture_new(
          invocation_identifier 
          invocation_arguments 
          invocation_token 
          arguments_begin_token 
          arguments_end_token
        )
        ans_append(result)
        if(${take} GREATER 0)
          math(EXPR take "${take} - 1")
        endif()
      endif() 
    endif() 

    if(reverse)
      map_tryget(${invocation_token} previous )
      ans(end)
    else()
      set(current ${arguments_after_end_token})
    endif()
  endwhile()
  return_ref(result)
endfunction()
