function(uri_parse_path uri)
    map_get("${uri}" path)
    ans(path)

    set(segments)
    set(encoded_segments)
    set(last_segment)
    string_take_regex(path "${segment_separator_char}")
    ans(slash)
    set(leading_slash ${slash})

    while (true)
        string_take_regex(path "${segment_char}+")
        ans(segment)

        if ("${segment}_" STREQUAL "_")
            break()
        endif ()

        string_take_regex(path "${segment_separator_char}")
        ans(slash)

        list(APPEND encoded_segments "${segment}")

        uri_decode("${segment}")
        ans(segment)
        list(APPEND segments "${segment}")
        set(last_segment "${segment}")
    endwhile ()

    set(trailing_slash "${slash}")
    set(normalized_segments)
    set(current_segments ${segments})

    while (true)
        list_pop_front(current_segments)
        ans(segment)

        if ("${segment}_" STREQUAL "_")
            break()
        elseif ("${segment}" STREQUAL ".")

        elseif ("${segment}" STREQUAL "..")
            list(LENGTH normalized_segments len)

            list_pop_back(normalized_segments)
            ans(last)
            if ("${last}" STREQUAL "..")
                list(APPEND normalized_segments .. ..)
            elseif ("${last}_" STREQUAL "_")
                list(APPEND normalized_segments ..)
            endif ()
        else ()
            list(APPEND normalized_segments "${segment}")
        endif ()
    endwhile ()

    if (("${segments}_" STREQUAL "_") AND leading_slash)
        set(trailing_slash "")
    endif ()

    map_capture(${uri} segments encoded_segments last_segment trailing_slash leading_slash normalized_segments)
    return()
endfunction()