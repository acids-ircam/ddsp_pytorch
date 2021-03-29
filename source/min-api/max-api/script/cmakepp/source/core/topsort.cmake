# executes the topological sort for a list of nodes (passed as varargs)
# get_hash is a function to be provided which returns the unique id for a node
# this is used to check if a node was visited previously
# expand should take a node and return its successors
# this function will return nothing if there was a cycle or if no input was given
# else it will return the topological order of the graph
function(topsort get_hash expand)
  function_import("${get_hash}" as __topsort_get_hash REDEFINE)
  function_import("${expand}" as __topsort_expand REDEFINE)
  # visitor function
  function(topsort_visit result visited node)
    # get hash for current node
    __topsort_get_hash("${node}")
    ans(hash)

    map_tryget("${visited}" "${hash}")
    ans(mark)
    
    if("${mark}" STREQUAL "temp")
      #cycle found
      return(true)
    endif()
    if(NOT mark)
      map_set("${visited}" "${hash}" temp)
      __topsort_expand("${node}")
      ans(successors)
      # visit successors
      foreach(successor ${successors})
        topsort_visit("${result}" "${visited}" "${successor}")
        ans(cycle)
        if(cycle)
      #    message("cycle found")
          return(true)
        endif()
      endforeach()

      #mark permanently
      map_set("${visited}" "${hash}" permanent)

      # add to front of result
      address_push_front("${result}" "${node}")
    endif()
    return(false)
  endfunction()


  map_new()
  ans(visited)
  address_new()
  ans(result)

  # select unmarked node and visit
  foreach(node ${ARGN})
    # get hash for node
    __topsort_get_hash("${node}")
    ans(hash)
    
    # get marking      
    map_tryget("${visited}" "${hash}")
    ans(mark)
    if(NOT mark)
      topsort_visit("${result}" "${visited}" "${node}")
      ans(cycle)
      if(cycle)
       # message("stopping with cycle")
        return()
      endif()
    endif()

  endforeach()
#  message("done")
  address_get(${result})

  return_ans()
endfunction()