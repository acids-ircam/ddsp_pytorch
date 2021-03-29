function(markdown_template_link file)
    path_qualify(file)
    path_relative("${root_template_dir}" "${file}")
    ans(relative_path)

    fread_lines("${file}" --limit-count 1 --regex "#+ .*")
    ans(title)

    string(REGEX REPLACE "#+ *(.*)" "\\1" title "${title}")
    return("[${title}](${relative_path})")
endfunction()