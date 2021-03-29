
function(target_export_header
  targetName
  apiName
  )


target_get("${targetName}" TYPE)
ans(targetType)


string(TOUPPER "${apiName}" targetNameUpper)


if("${targetType}" STREQUAL SHARED_LIBRARY)
  target_compile_definitions("${targetName}" PRIVATE "-D${targetNameUpper}_EXPORTS")
else()
  target_compile_definitions("${targetName}${staticSuffix}" PRIVATE "-D${targetNameUpper}_STATIC")
endif()



format("#pragma once
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the {targetNameUpper}_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// ISFREFLECTION_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef {targetNameUpper}_EXPORTS
#define {targetNameUpper}_API __declspec(dllexport)
#else
#ifdef {targetNameUpper}_STATIC
#define {targetNameUpper}_API
#else
#define {targetNameUpper}_API __declspec(dllimport)
#endif
#endif
//#define override  takes 10 seconds off compile time because of warnings generated (non standard extension)



#define {targetNameUpper}_NAMESPACE_BEGIN namespace ${targetName} {
#define {targetNameUpper}_NAMESPACE_END }


#define ${targetName}_namespace ${targetName}")
ans(formatted)
fwrite("${CMAKE_CURRENT_BINARY_DIR}/${targetName}/config.h" "${formatted}")

endfunction()
