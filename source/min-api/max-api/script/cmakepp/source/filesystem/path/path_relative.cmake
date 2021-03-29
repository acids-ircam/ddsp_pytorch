# returns the path specified by path_rel relative to 
# path_base using parent dir path syntax (../../path/to/x)
# if necessary
# e.g. path_rel(c:/dir1/dir2 c:/dir1/dir3/dir4)
# will result in ../dir3/dir4
# returns nothing if transformation is not possible
function(path_relative path_base path_rel)
    set(args ${ARGN})

    path_qualify(path_base)
    path_qualify(path_rel)

    if ("${path_base}" STREQUAL "${path_rel}")
        return(".")
    endif ()

    path_split("${path_base}")
    ans(base_parts)

    path_split("${path_rel}")
    ans(rel_parts)

    set(result_base)

    set(first true)

    while (true)
        list_peek_front(base_parts)
        ans(current_base)
        list_peek_front(rel_parts)
        ans(current_rel)

        if (NOT "${current_base}" STREQUAL "${current_rel}")
            if (first)
                return_ref(path_rel)
            endif ()
            break()
        endif ()
        set(first false)

        path_combine("${result_base}" "${current_base}")
        ans(result_base)
        list_pop_front(base_parts)
        list_pop_front(rel_parts)
    endwhile ()

    set(result_path)

    foreach (base_part ${base_parts})
        path_combine(${result_path} "..")
        ans(result_path)
    endforeach ()

    path_combine(${result_path} ${rel_parts})
    ans(result_path)

    if ("${result_path}" MATCHES "^\\/")
        string_substring("${result_path}" 1)
        ans(result_path)
    endif ()

    return_ref(result_path)
endfunction()

# transforms a path to a path relative to base_dir
#function(path_relative base_dir path)
#  path("${base_dir}")
#  ans(base_dir)
#  path("${path}")
#  ans(path)
#  string_take(path "${base_dir}")
#  ans(match)
#
#  if(NOT match)
#    return_ref(path)
#    #message(FATAL_ERROR "${path} is  not relative to ${base_dir}")
#  endif()
#
#  if("${path}" MATCHES "^\\/")
#    string_substring("${path}" 1)
#    ans(path)
#  endif()
#
#
#  if(match AND NOT path)
#    set(path ".")
#  endif()
#
#  return_ref(path)
#endfunction()
#