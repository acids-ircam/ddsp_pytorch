# imports the specified map as a function table which is callable via <function_name>
# whis is a performance enhancement 
function(function_import_table map function_name)
  map_keys(${map} )
  ans(keys)
  set("ifs" "if(false)\n")
  foreach(key ${keys})
    map_get(${map}  ${key})
    ans(command_name)
    set(ifs "${ifs}elseif(\"${key}\" STREQUAL \"\${switch}\" )\n${command_name}(\"\${ARGN}\")\nreturn_ans()\n")
  endforeach()
  set(ifs "${ifs}endif()\n")
set("evl" "function(${function_name} switch)\n${ifs}\nreturn()\nendfunction()")
   # message(${evl})
  set_ans("")
   
    eval("${evl}")
endfunction()

