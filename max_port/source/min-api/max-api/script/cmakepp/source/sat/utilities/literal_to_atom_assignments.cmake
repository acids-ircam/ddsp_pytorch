

## takes a literal assignment model 
## returns the atom assignments
function(literal_to_atom_assignments f literal_assignments)
  map_tryget(${f} l_last)
  ans(l_last)
  map_tryget(${f} literal_negated_map)
  ans(literal_negated_map)
  map_tryget(${f} literal_atom_map)
  ans(literal_atom_map)
  map_tryget(${f} atom_map)
  ans(atom_map)
  map_new()
  ans(atom_assignments)
  foreach(i RANGE 0 ${l_last})
    map_tryget(${literal_assignments} ${i})
    ans(value)
  #  print_vars(i value)
    if(NOT "${value}_" STREQUAL "_")
      map_tryget(${literal_atom_map} ${i})
      ans(ai)

      map_tryget(${atom_map} ${ai})
      ans(atom_name)

      #print_vars(atom_map atom_name ai)

      map_tryget(${literal_negated_map} ${i})
      ans(negated)
      #message("value ${atom_name} ${i} ${value}")
      if(negated)
        eval_truth(NOT value)
        ans(value)
      endif()

      map_set(${atom_assignments} ${atom_name} ${value})
    endif()
  endforeach()  
  return_ref(atom_assignments)
endfunction()

