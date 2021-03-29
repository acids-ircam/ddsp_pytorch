## naive way of parsing xml tags
## returns a list of all matched xml nodes
## warning: does not supported nested nodes of same name!! and no tag whithout closing tag: <test/>
## {
##  value:"content",
##  attrs:{
##    key:"val",
##    key:"val",
##    ...
##  }
## }
function(xml_parse_tags xml tag)
  set(regex_str "\\\"([^\\\"]*)\\\"")
  set(regex_attrs "([a-zA-Z_-][a-zA-Z0-9_-]*) *= *${regex_str}")
  set(regex "< *${tag}([^>]*)>(.*)</ *${tag} *>")
  string(REGEX MATCHALL "${regex}"  output "${xml}")

  set(res)
  foreach(match ${output})
    string(REGEX REPLACE "${regex}" "\\1" attrs "${match}") 
    string(REGEX REPLACE "${regex}" "\\2" match "${match}") 


    map()
      kv(tag "${tag}")
      kv(value "${match}")    
      map(attrs)
        string(REGEX MATCHALL "${regex_attrs}" attrs "${attrs}")
        foreach(attr ${attrs})
          string(REGEX REPLACE "${regex_attrs}" "\\1" key "${attr}")
          string(REGEX REPLACE "${regex_attrs}" "\\2" val "${attr}")
          kv("${key}" "${val}")
        endforeach()
      end()
    end()
    ans(t)
    list(APPEND res ${t})

  endforeach()

  return_ref(res)

endfunction()