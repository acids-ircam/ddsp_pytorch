function(print_commands)

get_cmake_property(_variableNames COMMANDS)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}")
endforeach()

endfunction()