
function(cmake_write path )
    cmake_serialize(${ARGN})
    ans(serialized)
    fwrite("${path}" "${serialized}")
    return_ans()
endfunction()