

  function(dns_parse input)
    regex_uri()

    string_take_regex_replace(input "${dns_user_info_regex}" "\\1")
    ans(user_info)
    
    set(host_port "${input}")


    string_take_regex_replace(input "${dns_host_regex}" "\\1")
    ans(host)

    string_take_regex(input "${dns_port_regex}")
    ans(port)


    if(port AND NOT "${port}" LESS 65536)
      return()
    endif()
    set(rest ${input})

    set(input "${host}")
    string_take_regex(input "${ipv4_regex}")
    ans(ip)

    set(top_label)
    set(labels)
    if(NOT ip)
      while(true)
        string_take_regex(input "${dns_domain_label_regex}")
        ans(label)
        if("${label}_" STREQUAL "_")
          break()

        endif()
        set(top_label "${label}")
        list(APPEND labels "${label}")
        string_take_regex(input "${dns_domain_label_separator}")
        ans(separator)
        if(NOT separator)
          break()
        endif()

      endwhile()


    endif()

    list(LENGTH labels len)
    set(domain)
    if("${len}" GREATER 1)
      list_slice(labels -3 -1)
      ans(domain)
      string_combine("." ${domain} )
      ans(domain)
    else()
      set(domain "${top_label}")
    endif()

    string_split_at_first(user_name password "${user_info}" ":")


    set(normalized_host "${host}")
    if("${normalized_host}_" STREQUAL "_" )
      set(normalized_host localhost)
    endif()

    map_capture_new(
      user_info
      user_name
      password
      host_port
      host
      normalized_host
      labels
      top_label
      domain
      ip
      port
      rest
      )
    return_ans()
  endfunction()