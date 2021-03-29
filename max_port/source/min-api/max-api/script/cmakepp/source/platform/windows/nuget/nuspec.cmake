
    function(nuspec package_handle)
        assign(package_handle.package_descriptor.tags = ['a','b'])
        assign(package_handle.package_descriptor.copyright = '2017')
        template_run(
            #<authors>@package_handle.package_descriptor.authors</authors>
    #         <authors>myauthor</authors>
    # <owners>owner</owners>
    # <licenseUrl>http://mylicense</licenseUrl>
    # <projectUrl>http://myproject@package_handle.package_descriptor.project_url</projectUrl>
    # <iconUrl>@package_handle.package_descriptor.icon_url</iconUrl>
#    <description>@package_handle.package_descriptor.description</description>
"<package xmlns=\"http://schemas.microsoft.com/packaging/2012/06/nuspec.xsd\">
  <metadata>
    <id>@package_handle.package_descriptor.id</id>
    <version>@package_handle.package_descriptor.version</version>
    <authors>tobitob</authors>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
    <description>some description</description>
    <summary>@package_handle.package_descriptor.summary</summary>
    <releaseNotes>@package_handle.package_descriptor.release_notes</releaseNotes>
    <copyright>@package_handle.package_descriptor.copyright</copyright>
    <tags>@string_combine(' ', $package_handle.package_descriptor.tags)</tags>
  </metadata>  
  <files>
    <file src=\"@package_handle.content_dir/**\" target=\"native\"/>
  </files>
 </package>"
)
        ans(result)
        return_ref(result)        
endfunction()



function(package_source_push_nuget)
     if("${ARGN}" MATCHES "(.*);=>;?(.*)")
        set(source_args "${CMAKE_MATCH_1}")
        set(args "${CMAKE_MATCH_2}")
    else()
        set(source_args ${ARGN})
        set(args)
    endif()

    list_pop_front(source_args)
    ans(source)
        
    ## get target dir
    ## or target uri...
    list_pop_front(args)
    ans(target_dir)
    if(NOT target_dir)
        pwd()
        ans(target_dir)        
    endif()

    path_qualify(target_dir)

    pushtmp()
    ans(tmpdir)

    assign(package_handle = source.pull(${source_args} "${tmpdir}"))


    if(NOT package_handle)
        error("could not pull `${source_args}` ")        
        poptmp()
        return()
    endif()

    nuspec(${package_handle})
    ans(nuspec)
    message("tempdir ${tmpdir}")
    message("${nuspec}")
    fwrite("package.nuspec" "${nuspec}")
    message("pack package.nuspec -BasePath \"${tmpdir}\" -OutputDirectory \"${target_dir}\" --passthru")
    nuget(pack package.nuspec -BasePath "${tmpdir}" -OutputDirectory "${target_dir}" --passthru)



    # package handle set uri to nuget file?    



    #poptmp()
popd()

    return_ref(package_handle)

endfunction()
