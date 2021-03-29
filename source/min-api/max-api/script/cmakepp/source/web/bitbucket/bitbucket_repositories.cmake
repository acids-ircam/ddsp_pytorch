

function(bitbucket_repositories user)
    set(result)

    set(api_uri "https://api.bitbucket.org/2.0")
    set(current_uri "${api_uri}/repositories/${user}")
    set(names)
    while (true)
        http_get("${current_uri}" --response)
        ans(response)
        assign(error = response.client_status)
        assign(current_result = response.content)
        if (error)
            error("failed to query ${current_uri} http client said: {response.client_message} ({response.client_status})}")
            return()
        endif ()
        json_extract_string_value(next "${current_result}")
        ans(current_uri)
        json_extract_string_value(name "${current_result}")
        ans_append(names)

        if (NOT current_uri)
            break()
        endif ()
    endwhile ()

    list_remove_duplicates(names)

    ## hack because of the way that json_extract_string_value works I have to remove other names
    # Doesn't work properly, but who uses bitbucket in 2020...
    list_remove(names ssh https master)

    return_ref(names)
endfunction()