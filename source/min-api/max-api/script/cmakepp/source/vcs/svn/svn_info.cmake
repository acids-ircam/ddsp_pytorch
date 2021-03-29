## returns an info object for the specified svn url
## {
##    path:"path",
##    revision:"revision",
##    kind:"kind",
##    url:"url",
##    root:"root",
##    uuid:"uuid",
## }
## todo: cached?
function(svn_info uri)
    svn_uri("${uri}")
    ans(uri)


    svn(info ${uri} --process-handle --xml ${ARGN})
    ans(res)
    map_tryget(${res} exit_code)
    ans(error)
    if(error)
      return()
    endif()

    map_tryget(${res} stdout)
    ans(xml)

    xml_parse_attrs("${xml}" entry path)    
    ans(path)
    xml_parse_attrs("${xml}" entry revision)    
    ans(revision)
    xml_parse_attrs("${xml}" entry kind)    
    ans(kind)
    xml_parse_values("${xml}" url)
    ans(url)
    xml_parse_values("${xml}" root)
    ans(root)
    xml_parse_values("${xml}" relative-url)
    ans(relative_url)

    string(REGEX REPLACE "^\\^/" "" relative_url "${relative_url}")

    xml_parse_values("${xml}" uuid)
    ans(uuid)
    map()
      var(path revision kind url root uuid relative_url)
    end()
    ans(res)
    return_ref(res)
endfunction()