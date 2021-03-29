## 
## goes through all of cmakepp's README.md.in files and generates them
function(cmakepp_compile_docs)
  cmakepp_config(base_dir)
  ans(base_dir)
  file(GLOB_RECURSE template_paths "${base_dir}/**README.md.in")
  
  foreach(template_path ${template_paths})
      get_filename_component(template_dir "${template_path}" PATH)
      set(output_file "${template_dir}/README.md")
      message("generating ${output_file}")
      template_run_file("${template_path}")
      ans(generated_content)
      fwrite("${output_file}" "${generated_content}")
  endforeach()

endfunction()