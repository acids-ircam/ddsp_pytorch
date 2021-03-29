## 
## permutates the specified input map 
## takes every key of the input map and treats the value as a list
## the result is n maps which contain one value per key

function(map_permutate input)
  
  map_keys("${input}")
  ans(keys)

  map_new()
  ans(result)
  foreach(key ${keys})
    map_get("${input}" "${key}")
    ans(values)

    set(currentList ${result})
    set(result)
    foreach(current ${currentList})
      foreach(value IN LISTS values)
        map_clone_shallow("${current}")
        ans(current)
        map_set(${current} ${key} ${value})
        list(APPEND result ${current})
      endforeach()      
    endforeach()
  endforeach()
  return_ref(result)
endfunction()