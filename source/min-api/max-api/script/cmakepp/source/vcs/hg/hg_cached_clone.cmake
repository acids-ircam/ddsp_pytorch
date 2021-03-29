## hg_cached_clone(<remote_uri:<~uri>> <?target_dir> [--ref <ref>] [--readonly] [--file <rel path>] [--read <rel path>])
##
## performs a cached clone from the specified remote uri
##
function(hg_cached_clone remote_uri)
    set(args ${ARGN})

    ### extract options
    list_extract_labelled_value(args --ref)
    ans(ref)

    list_extract_flag(args --readonly)
    ans(readonly)

    list_extract_labelled_value(args --file)
    ans(file)

    list_extract_labelled_value(args --read)
    ans(read)

    if (file AND read)
        message(FATAL_ERROR "--file and --read is not allowed")
    endif ()

    list_pop_front(args)
    ans(target_dir)

    path_qualify(target_dir)

    ## create a cache directory for the uri
    cmakepp_config(cache_dir)
    ans(cache_dir)

    string(MD5 cache_key "${remote_uri}")

    set(repo_cache_dir "${cache_dir}/hg_cache/${cache_key}")

    ## initial clone of repo
    if (NOT EXISTS "${repo_cache_dir}")
        hg(clone "${remote_uri}" "${repo_cache_dir}" --exit-code)#
        ans(error)
        if (error)
            rm("${repo_cache_dir}")
            message(FATAL_ERROR "hg could not clone ${remote_uri} - failed with ${error}")
        endif ()
    endif ()

    # update cached repo then copy it over to the target dir
    # where the correct revision is checked out
    pushd("${repo_cache_dir}")

    hg(update)

    if (file OR read)
        set(path ${file} ${read})
        if (${ref})
            set(ref "-r${ref}")
        endif ()

        hg(cat ${ref} "${path}" --process-handle)
        ans(hg_result)
        assign(error = hg_result.exit_code)
        if (NOT error)
            assign(result = hg_result.stdout)
            if (file)
                fwrite("${target_dir}/${path}" "${result}")
                set(result "${target_dir}/${path}")
            endif ()
        endif ()
    elseif (readonly)
        set(result ${repo_cache_dir})
    else ()
        cp_dir("${repo_cache_dir}" "${target_dir}")
        pushd("${target_dir}")
        if (${ref})
            hg(checkout "${ref}")
        endif ()
        popd()
        set(result "${target_dir}")
    endif ()
    popd()

    return_ref(result)
endfunction()