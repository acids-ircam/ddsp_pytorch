
function(function_import_dispatcher function_name)
    string(REPLACE ";" "\n" content "${ARGN}")

    string(REGEX REPLACE "([^\n]+)" "elseif(command STREQUAL \"\\1\")\n \\1(\${ARGN})\nreturn_ans()\n" content "${content}")
    eval("
        function(${function_name} command)
          if(false)
          ${content}
            endif()
          return()
        endfunction()

      ")
      return()
endfunction()


function(function_import_global_dispatcher function_name)
    get_cmake_property(commands COMMANDS)       
    list(REMOVE_ITEM commands else if elseif endif while function endwhile endfunction macro endmacro foreach endforeach)
    function_import_dispatcher("${function_name}" ${commands})
    return()
endfunction()