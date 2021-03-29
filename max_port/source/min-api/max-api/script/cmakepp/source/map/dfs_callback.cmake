
# emits events parsing a list of map type elements 
# expects a callback function that takes the event type string as a first argument
# follwowing events are called (available context variables are listed as subelements: 
# value
#   - list_length (may be 0 or 1 which is good for a null check)
#   - content_length (contains the length of the content)
#   - node (contains the value)
# list_begin
#   - list_length (number of elements the list contains)
#   - content_length (accumulated length of list elements + semicolon separators)
#   - node (contains all values of the lsit)
# list_end
#   - list_length(number of elements in list)
#   - node (whole list)
#   - list_char_length (length of list content)
#   - content_length (accumulated length of list elements + semicolon separators)
# list_element_begin
#   - list_length(number of elements in list)
#   - node (whole list)
#   - list_char_length (length of list content)
#   - content_length (accumulated length of list elements + semicolon separators)
#   - list_element (contains current list element)
#   - list_element_index (contains current index )   
# list_element_end
#   - list_length(number of elements in list)
#   - node (whole list)
#   - list_char_length (length of list content)
#   - content_length (accumulated length of list elements + semicolon separators)
#   - list_element (contains current list element)
#   - list_element_index (contains current index )
# visited_reference
#   - node (contains ref to revisited map)
# unvisited_reference
#   - node (contains ref to unvisited map)
# map_begin
#   - node( contains ref to map)
#   - map_keys (contains all keys of map)
#   - map_length (contains number of keys of map)
# map_end
#   - node( contains ref to map)
#   - map_keys (contains all keys of map)
#   - map_length (contains number of keys of map)
# map_element_begin
#   - node( contains ref to map)
#   - map_keys (contains all keys of map)
#   - map_length (contains number of keys of map)
#   - map_element_key (current key)
#   - map_element_value (current value)
#   - map_element_index (current index)
# map_element_end
#   - node( contains ref to map)
#   - map_keys (contains all keys of map)
#   - map_length (contains number of keys of map)
#   - map_element_key (current key)
#   - map_element_value (current value)
#   - map_element_index (current index)
function(dfs_callback callback)
  # inner function
  function(dfs_callback_inner node)
 

    is_map("${node}")
    ans(ismap)
    if(NOT ismap)
      list(LENGTH node list_length)
      string(LENGTH "${node}" content_length)
      if(${list_length} LESS 2)
        dfs_callback_emit(value)
      else()
        dfs_callback_emit(list_begin) 
        set(list_element_index 0)
        foreach(list_element ${node})
          list_push_back(path "${list_element_index}")
          dfs_callback_emit(list_element_begin)
          dfs_callback_inner("${list_element}")
          dfs_callback_emit(list_element_end)
          list_pop_back(path)
          math(EXPR list_element_index "${list_element_index} + 1")
        endforeach()
        dfs_callback_emit(list_end)
      endif()
      return()
    endif()

    map_tryget(${visited} "${node}")
    ans(was_visited)

    if(was_visited)
      dfs_callback_emit("visited_reference")
      return()
    else()
      dfs_callback_emit("unvisited_reference")
    endif()


    map_set(${visited} "${node}" true)

    map_keys("${node}")
    ans(map_keys)

    list(LENGTH map_keys map_length)

    dfs_callback_emit(map_begin)

    
    set(map_element_index 0)
    foreach(map_element_key ${map_keys})
      map_tryget("${node}" ${map_element_key})
      ans(map_element_value)
      list_push_back(path "${map_element_key}")
      dfs_callback_emit(map_element_begin)

      dfs_callback_inner("${map_element_value}")

      dfs_callback_emit(map_element_end)
      list_pop_back(path)

      math(EXPR map_element_index "${map_element_index} + 1")
    endforeach()


    dfs_callback_emit(map_end "${node}" )
  endfunction()

  function(dfs_callback callback)
#    curry3(dfs_callback_emit => "${callback}"(/0) as dfs_callback_emit)
    # faster
    eval("
function(dfs_callback_emit)
  ${callback}(\${ARGN})
endfunction()
")
    map_new()
    ans(visited)

   # foreach(arg ${ARGN})
   set(path)
    dfs_callback_inner("${ARGN}")
   # endforeach()
    return()
  endfunction()
  dfs_callback("${callback}" ${ARGN})
  return_ans()
endfunction()