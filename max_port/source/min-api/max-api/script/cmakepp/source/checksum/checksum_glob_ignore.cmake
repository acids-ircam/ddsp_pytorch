## `(<glob ignore expressions...> [--algorithm <hash algorithm> = "MD5"])-><checksum>`
## 
## calculates the checksum for the specified glob ignore expressIONS
## uses checksum_files internally. the checksum is unique to file content
## and relative file structure
## 
function(checksum_glob_ignore)
    set(args ${ARGN})
    list_extract_labelled_keyvalue(args --algorithm)
    ans(algorithm)
    glob_ignore(${args})
    ans(files)


    pwd()
    ans(pwd)
    set(normalized_files)
    foreach(file ${files})
        path_qualify(file)
        path_relative("${pwd}" "${file}")
        ans(file)
        list(APPEND normalized_files ${file})      
    endforeach()



    checksum_files("${pwd}" ${normalized_files})
    return_ans()
endfunction()