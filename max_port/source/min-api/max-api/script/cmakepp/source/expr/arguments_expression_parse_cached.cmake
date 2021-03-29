
macro(arguments_expression_parse_cached type arguments argn start end)
    string(MD5 __cache_key "${ARGN}${arguments}${type}${${argn}}${start}${end}")
    map_tryget(cache "${__cache_key}")
    if (NOT __ans)
        arguments_expression_parse("${type}" "${arguments}" "${start}" "${end}")
        map_set(cache "${__cache_key}" "${__ans}")
    endif ()
endmacro()

