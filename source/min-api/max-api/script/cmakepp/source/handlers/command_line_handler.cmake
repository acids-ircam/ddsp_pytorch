
  function(command_line_handler)
    this_set(name "${ARGN}")

    ## forwards the object call operation to the run method
    this_declare_call(call)
    function(${call})

      obj_member_call(${this} run ${ARGN})
      ans(res)
      return_ref(res)
    endfunction()

    method(run)
    function(${run})
      handler_request(${ARGN})
      ans(request)
      assign(handler = this.find_handler(${request}))
      list(LENGTH handler handler_count)  


      if(${handler_count} GREATER 1)
        return_data("{error:'ambiguous_handler',description:'multiple command handlers were found for the request',request:$request}" )
      endif()

      if(NOT handler)
        return_data("{error:'no_handler',description:'command runner could not find an appropriate handler for the specified arguments',request:$request}")
      endif() 
      ## remove first item
      assign(request.input[0] = '') 
      set(parent_handler ${this})
      assign(result = this.execute_handler(${handler} ${request}))
      return_ref(result)

    endfunction()


    method(run_interactive)
    function(${run_interactive})
      if(NOT ARGN)
        echo_append("please enter a command>")
        read_line()
        ans(command)
      else()
        echo("executing command '${ARGN}':")
        set(command "${ARGN}")
      endif()
      obj_member_call(${this} run ${command})
      ans(res)
      table_serialize(${res})
      ans(formatted)
      echo(${formatted})
      return_ref(res)
    endfunction()

    ## compares the request to the handlers
    ## returns the handlers which matches the request
    ## can return multiple handlers
    method(find_handler)
    function(${find_handler})
      handler_request("${ARGN}")
      ans(request)
      this_get(handlers)
      handler_find(handlers "${request}")
      ans(handler)
      return_ref(handler)
    endfunction()

    ## executes the specified handler 
    ## the handler must not be part of this command runner
    ## it takes a handler and a request and returns a response object
    method(execute_handler)
    function(${execute_handler} handler)
      handler_request(${ARGN})
      ans(request)
      map_set(${request} runner ${command_line_handler})
      map_new()
      ans(response)
      handler_execute("${handler}" ${request} ${response})
      return_ref(response)
    endfunction()

    ## adds a request handler to this command handler
    ## request handler can be any function/function definition 
    ## or handler object
    method(add_handler)
    function(${add_handler})
      request_handler(${ARGN})
      ans(handler)
      if(NOT handler)
        return()
      endif()
      map_append(${this} handlers ${handler})
      
      return(${handler})
    endfunction()

  ## property contains a managed list of handlers
  property(handlers)
  ## setter
  function(${set_handlers} obj key new_handlers)
    map_tryget(${this} handlers)
    ans(old_handlers)
    if(old_handlers)
      list(REMOVE_ITEM new_handlers ${old_handlers})
    endif()

    set(result)
    foreach(handler ${new_handlers})
      set_ans("")
      obj_member_call(${this} add_handler ${handler})
      ans(res)
      list(APPEND result ${res})
    endforeach()
    return_ref(result)
  endfunction()
  ## getter
  function(${get_handlers})
    map_tryget(${this} handlers)
    return_ans()
  endfunction()


endfunction()

