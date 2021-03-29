
# contains common regular expression 
macro(regex_uri)
  if(NOT __regex_uri_included)
    set(__regex_uri_included true)
    set(lowalpha_range "a-z")
    set(lowalpha "[${lowalpha_range}]")
    set(upalpha_range "A-Z")
    set(upalpha "[${upalpha_range}]")
    set(digit_range "0-9")
    set(digit "[${digit_range}]")
    set(alpha_range "${lowalpha_range}${upalpha_range}")
    set(alpha "[${alpha_range}]")
    set(alphanum_range "${alpha_range}${digit_range}")
    set(alphanum "[${alphanum_range}]")

    set(reserved_no_slash_range "\;\\?:@&=\\+\\$,")
    set(reserved_no_slash "[${reserved_no_slash_range}]")
    set(reserved_range "\\/${reserved_no_slash_range}")
    set(reserved "[${reserved_range}]")
    set(mark_range "\\-_\\.!~\\*'\\(\\)")
    set(mark "[${mark_range}]")
    set(unreserved_range "${alpha_range}${mark_range}")
    set(unreserved "[${unreserved_range}]")
    set(hex_range "${0-9A-Fa-f}") 
    set(hex "[${hex_range}]")
    set(escaped "%${hex}${hex}")


    #set(uric "(${reserved}|${unreserved}|${escaped})")
    set(uric "[^ ]")
    set(uric_so_slash "${unreserved}|${reserved_no_slash}|${escaped}")


    set(scheme_mark_range "\\+\\-\\.")
    set(scheme_mark "[${scheme_mark_range}]")
    set(scheme_delimiter ":")

    set(scheme_regex "${alpha}[${alphanum_range}${scheme_mark_range}]*")
    
    set(net_root_regex "//")
    set(abs_root_regex "/")

    set(abs_path "\\/${path_segments}")
    set(net_path "\\/\\/${authority}(${abs_path})?")

    set(authority_char "[^/\\?#]" )
    set(authority_regex "${authority_char}+")

    set(segment_char "[^\\?#/ ]")
    set(segment_separator_char "/")


    set(path_char_regex "[^\\?#]")
    set(query_char_regex "[^#]")
    set(query_delimiter "\\?")
    set(query_regex "${query_delimiter}${query_char_regex}*")
    set(fragment_char_regex "[^ ]")
    set(fragment_delimiter_regex "#")
    set(fragment_regex "${fragment_delimiter_regex}${fragment_char_regex}*")

  #  ";" | ":" | "&" | "=" | "+" | "$" | "," 
    set(dns_user_info_char "(${unreserved}|${escaped}|[;:&=+$,])")
    set(dns_user_info_separator "@")
    set(dns_user_info_regex "(${dns_user_info_char}+)${dns_user_info_separator}")

    set(dns_port_seperator :)
    set(dns_port_regex "[0-9]+")
    set(dns_host_regex_char "[^:]")
    set(dns_host_regex "(${dns_host_regex_char}+)${dns_port_seperator}?")
      set(dns_domain_toplabel_regex "${alpha}(${alphanum}|\\-)*")
      set(dns_domain_label_separator "[.]")
    set(dns_domain_label_regex "[^.]+")
    set(ipv4_group_regex "(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])")
    set(ipv4_regex "${ipv4_group_regex}[\\.]${ipv4_group_regex}[\\.]${ipv4_group_regex}[\\.]${ipv4_group_regex}")
  endif()
endmacro()