## `(<directory> [--algorithm <hash algorithm> "MD5"])-><checksum>`
## 
## this method generates the checksum for the specified directory
## it is done by taking every file's relative path into consideration
## and generating the hash.  The file's content does not influence the hash
## 
function(checksum_layout dir)
    path_qualify(dir)

    set(args ${ARGN})

    list_extract_labelled_keyvalue(args --algorithm)
    ans(algorithm)

    file(GLOB_RECURSE files RELATIVE "${dir}" "${dir}/**")

    if(files)
        ## todo sort. normalize paths, remove directories
        string(REPLACE "//" "/" files "${files}")
        list(SORT files)
    endif()
    
    checksum_string("${files}" ${algorithm})
    ans(checksum_dir)

    return_ref(checksum_dir)
endfunction()

