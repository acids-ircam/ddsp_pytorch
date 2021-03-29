## 
## parsers the spefied type definitions
function(typed_value_definitions name)
  regex_cmake()
  string(REGEX MATCHALL "(^|;)<.*>($|;)" positionals "${ARGN}")
  string(REGEX MATCHALL "(^|;)\\[.*\\]($|;)" nonpositionals "${ARGN}")
  string(REGEX REPLACE "((^|;)<.*>($|;))|(^|;)\\[.*\\]($|;)" "" comments "${ARGN}")
  string(REGEX REPLACE "(^|[\n])[ \t]*#([^\n]*)" "\\2\n" comments "${comments}")

  map_new()
  ans(def)  
  map_set(${def} name "${name}")
  
  if(comments)
    map_set(${def} description "${comments}")
  endif()  
  foreach(positional ${positionals})
  typed_value_definition("${positional}")
    ans(d)
    map_append(${def} positionals ${d})
  endforeach()

  foreach(nonpositional ${nonpositionals})
    typed_value_definition("${nonpositional}")
    ans(d)
    map_append(${def} nonpositionals ${d})
  endforeach()
  return(${def})
endfunction()