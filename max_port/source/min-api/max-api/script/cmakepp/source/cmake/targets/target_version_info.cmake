## adds version info to the specified target
## heavily inspired by https://github.com/halex2005/CMakeHelpers
function(target_version_info)
  arguments_extract_typed_values(0 ${ARGC} 
        <target:<string>>       
        [--icon:<path>=project.ico]
        [--version:<semver>=1.0.0]
        [--revision:<int>=0]
        [--company:<string>]
        [--description:<string>]
        [--internal_name:<string>]
        [--original_file_name:<string>]
        [--bundle:<string>]
        [--copyright:<int>]   #year
        [--verbose]            #verbose output
      )

if(NOT WIN32)
  message(WARNING "currently only supported under windows")
  return()
endif()

if(NOT EXISTS "${icon}")
  set(icon "")
endif()



  if(NOT comment)
    semver_format(${version})
    ans(formatted)
    set(comment "${target} in version ${formatted}")
  endif()


map()
    kv(version "${version}")
    kv(product_name "${target}")
    kv(icon_path "${icon}")
    kv(revision "${revision}")
    kv(comments "${comment}")
    kv(company "${company}")
    kv(description "${description}")
    kv(internal_name "${internal_name}")
    kv(original_file_name "${original_file_name}")
    kv(bundle "${bundle}")
    kv(copyright "(c) ${copyright}")
  end()
  ans(productInfo)





# format the include file
  format("#pragma once

#ifndef PRODUCT_VERSION_MAJOR
#define PRODUCT_VERSION_MAJOR {productInfo.version.major}
#endif

#ifndef PRODUCT_VERSION_MINOR
#define PRODUCT_VERSION_MINOR {productInfo.version.minor}
#endif

#ifndef PRODUCT_VERSION_PATCH
#define PRODUCT_VERSION_PATCH {productInfo.version.patch}
#endif

#ifndef PRODUCT_VERSION_BUILD
#define PRODUCT_VERSION_BUILD {productInfo.revision}
#endif

#ifndef FILE_VERSION_MAJOR
#define FILE_VERSION_MAJOR {productInfo.version.major}
#endif

#ifndef FILE_VERSION_MINOR
#define FILE_VERSION_MINOR {productInfo.version.minor}
#endif

#ifndef FILE_VERSION_PATCH
#define FILE_VERSION_PATCH {productInfo.version.patch}
#endif

#ifndef FILE_VERSION_BUILD
#define FILE_VERSION_BUILD {productInfo.revision}
#endif

#ifndef __TO_STRING
#define __TO_STRING_IMPL(x) #x
#define __TO_STRING(x) __TO_STRING_IMPL(x)
#endif

#define PRODUCT_VERSION_MAJOR_MINOR_STR        __TO_STRING(PRODUCT_VERSION_MAJOR) \".\" __TO_STRING(PRODUCT_VERSION_MINOR)
#define PRODUCT_VERSION_MAJOR_MINOR_PATCH_STR  PRODUCT_VERSION_MAJOR_MINOR_STR \".\" __TO_STRING(PRODUCT_VERSION_PATCH)
#define PRODUCT_VERSION_FULL_STR               PRODUCT_VERSION_MAJOR_MINOR_PATCH_STR \".\" __TO_STRING(PRODUCT_VERSION_BUILD)
#define PRODUCT_VERSION_RESOURCE               PRODUCT_VERSION_MAJOR,PRODUCT_VERSION_MINOR,PRODUCT_VERSION_PATCH,PRODUCT_VERSION_BUILD
#define PRODUCT_VERSION_RESOURCE_STR           PRODUCT_VERSION_FULL_STR \"\\0\"

#define FILE_VERSION_MAJOR_MINOR_STR        __TO_STRING(FILE_VERSION_MAJOR) \".\" __TO_STRING(FILE_VERSION_MINOR)
#define FILE_VERSION_MAJOR_MINOR_PATCH_STR  FILE_VERSION_MAJOR_MINOR_STR \".\" __TO_STRING(FILE_VERSION_PATCH)
#define FILE_VERSION_FULL_STR               FILE_VERSION_MAJOR_MINOR_PATCH_STR \".\" __TO_STRING(FILE_VERSION_BUILD)
#define FILE_VERSION_RESOURCE               FILE_VERSION_MAJOR,FILE_VERSION_MINOR,FILE_VERSION_PATCH,FILE_VERSION_BUILD
#define FILE_VERSION_RESOURCE_STR           FILE_VERSION_FULL_STR \"\\0\"

#ifndef PRODUCT_ICON
#define PRODUCT_ICON \"{productInfo.icon_path}\"
#endif

#ifndef PRODUCT_COMMENTS
#define PRODUCT_COMMENTS           \"{productInfo.comments}\\0\"
#endif

#ifndef PRODUCT_COMPANY_NAME
#define PRODUCT_COMPANY_NAME       \"{productInfo.company}\\0\"
#endif

#ifndef PRODUCT_COMPANY_COPYRIGHT
#define PRODUCT_COMPANY_COPYRIGHT  \"{productInfo.copyright}\\0\"
#endif

#ifndef PRODUCT_FILE_DESCRIPTION
#define PRODUCT_FILE_DESCRIPTION   \"{productInfo.description}\\0\"
#endif

#ifndef PRODUCT_INTERNAL_NAME
#define PRODUCT_INTERNAL_NAME      \"{productInfo.internal_name}\\0\"
#endif

#ifndef PRODUCT_ORIGINAL_FILENAME
#define PRODUCT_ORIGINAL_FILENAME  \"{productInfo.original_file_name}\\0\"
#endif

#ifndef PRODUCT_BUNDLE
#define PRODUCT_BUNDLE             \"{productInfo.bundle}\\0\"
#endif
")
ans(versionInfoTemplate)

set(iconComment "//")
if(icon)
  set(iconComment "")
endif()

# format the resource file
format("
#include \"VersionInfo.h\"
#include \"winres.h\"

${iconComment}IDI_ICON1               ICON                    PRODUCT_ICON

LANGUAGE LANG_INVARIANT, SUBLANG_NEUTRAL

VS_VERSION_INFO VERSIONINFO
    FILEVERSION FILE_VERSION_RESOURCE
    PRODUCTVERSION PRODUCT_VERSION_RESOURCE
    FILEFLAGSMASK 0x3fL
#ifdef _DEBUG
    FILEFLAGS 0x1L
#else
    FILEFLAGS 0x0L
#endif
    FILEOS 0x4L
    FILETYPE 0x1L
    FILESUBTYPE 0x0L
BEGIN
    BLOCK \"StringFileInfo\"
    BEGIN
        BLOCK \"041904b0\"
        BEGIN
            VALUE \"Comments\", PRODUCT_COMMENTS
            VALUE \"CompanyName\", PRODUCT_COMPANY_NAME
            VALUE \"FileDescription\", PRODUCT_FILE_DESCRIPTION
            VALUE \"FileVersion\", FILE_VERSION_RESOURCE_STR
            VALUE \"InternalName\", PRODUCT_INTERNAL_NAME
            VALUE \"LegalCopyright\", PRODUCT_COMPANY_COPYRIGHT
            VALUE \"OriginalFilename\", PRODUCT_ORIGINAL_FILENAME
            VALUE \"ProductName\", PRODUCT_BUNDLE
            VALUE \"ProductVersion\", PRODUCT_VERSION_RESOURCE_STR
        END
    END
    BLOCK \"VarFileInfo\"
    BEGIN
        VALUE \"Translation\", 0x419, 1200
    END
END
")
ans(resourceTemplate)

  path("${CMAKE_CURRENT_BINARY_DIR}/VersionInfo.h")
  ans(versionInfoHeaderFile)

  path("${CMAKE_CURRENT_BINARY_DIR}/version.rc")
  ans(versionResourceFile)

  fwrite("${versionInfoHeaderFile}" "${versionInfoTemplate}")
  fwrite("${versionResourceFile}" "${resourceTemplate}")

  target_sources(${target} PRIVATE "${versionInfoHeaderFile}" "${versionResourceFile}")


if(verbose)
  message(INFO "added version information to target")
  message(INFO "  version header file @ ${versionInfoHeaderFile}")
  message(INFO "  resource file       @ ${versionResourceFile}")
  json_print("${productInfo}")
endif()

endfunction()