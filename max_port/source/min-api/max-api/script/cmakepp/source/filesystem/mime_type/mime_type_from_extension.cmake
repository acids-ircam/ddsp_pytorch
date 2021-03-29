## mime_type_from_extension()->
##
## returns the mime type or types matching the specified file extension
##
function(mime_type_from_extension extension)

    if (${extension} MATCHES "\\.(.*)")
        set(extension "${CMAKE_MATCH_1}")
    endif ()

    string(TOLOWER "${extension}" extension)

    mime_type_map()
    ans(mime_types)

    map_tryget("${mime_types}" "${extension}")
    ans(mime_types)

    set(mime_type_names)
    foreach (mime_type ${mime_types})
        map_tryget("${mime_type}" name)
        ans(mime_type_name)
        list(APPEND mime_type_names "${mime_type_name}")
    endforeach ()

    return_ref(mime_type_names)
endfunction()

