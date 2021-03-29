/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    BEGIN_USING_C_LINKAGE


    /**	The size you should use when allocating strings for full paths.
        @ingroup files
    */
    #define MAX_PATH_CHARS		2048

    /**	The size you should use when allocating strings for filenames.
        At the time of this writing it supports up to 256 UTF chars
        @ingroup files
    */
    #define MAX_FILENAME_CHARS	512


    /**	Constants that determine the output of path_nameconform().
        @ingroup files
        @see #e_max_path_types
        @see path_nameconform()
    */
    enum e_max_path_styles {
        PATH_STYLE_MAX = 0,			///< use PATH_STYLE_MAX_PLAT
        PATH_STYLE_NATIVE,			///< use PATH_STYLE_NATIVE_PLAT
        PATH_STYLE_COLON,			///< ':'  sep, "vol:"   volume, ":"   relative, "^:" boot
        PATH_STYLE_SLASH,			///< '/'  sep, "vol:/"  volume, "./"  relative, "/"  boot
        PATH_STYLE_NATIVE_WIN		///< '\\' sep, "vol:\\" volume, ".\\" relative, "\\" boot
    };

    #ifdef WIN_VERSION
        #define PATH_STYLE_MAX_PLAT			PATH_STYLE_SLASH
        #define PATH_STYLE_NATIVE_PLAT		PATH_STYLE_NATIVE_WIN
    #else
        #define PATH_STYLE_MAX_PLAT			PATH_STYLE_SLASH
        #define PATH_STYLE_NATIVE_PLAT		PATH_STYLE_SLASH	// PATH_STYLE_COLON path changes for Max 5-jkc
    #endif

    /**	Constants that determine the output of path_nameconform().
        @ingroup files
        @see #e_max_path_styles
        @see path_nameconform()
    */
    enum e_max_path_types {
        PATH_TYPE_IGNORE = 0,		///< ignore
        PATH_TYPE_ABSOLUTE,			///< absolute path
        PATH_TYPE_RELATIVE,			///< relative path
        PATH_TYPE_BOOT,				///< boot path
        PATH_TYPE_C74,				///< Cycling '74 folder
        PATH_TYPE_PATH,				///< path
        PATH_TYPE_DESKTOP,			///< desktop
        PATH_TYPE_TILDE,			///< "home"
        PATH_TYPE_TEMPFOLDER,		///< /tmp
        PATH_TYPE_MAXDB				///< combi: try PATH_TYPE_C74, PATH_TYPE_TILDE, PATH_TYPE_RELATIVE, PATH_TYPE_ABSOLUTE in that order
    };


    static const int TYPELIST_SIZE = 32; //maximum number of types returned
    typedef enum {
        TYPELIST_MAXFILES		= 1,
        TYPELIST_EXTERNS		= 2,
        TYPELIST_COLLECTIVES	= 4,
        TYPELIST_MAXFORLIVE		= 8,
        TYPELIST_SNAPSHOTS		= 16,
        TYPELIST_GENPATCHERS	= 32,
        TYPELIST_SNIPPETS		= 64
    } e_max_typelists;

    void typelist_make(t_fourcc *types, long include, short *numtypes);



    /**	Flags used to represent properties of a file in a #t_fileinfo struct.
        @ingroup files
    */
    enum e_max_fileinfo_flags {
        PATH_FILEINFO_ALIAS = 1,	///< alias
        PATH_FILEINFO_FOLDER = 2,	///< folder
        PATH_FILEINFO_PACKAGE = 4	///< package (Mac-only)
    };


    /** Flags used by functions such as path_foldernextfile() and path_openfolder().
        @ingroup files
    */
    enum e_max_path_folder_flags {
        PATH_REPORTPACKAGEASFOLDER = 1,	///< if not true, then a Mac OS package will be reported as a file rather than a folder.
        PATH_FOLDER_SNIFF = 2,			///< sniff
        PATH_NOALIASRESOLUTION = 4		///< no alias resolution
    };


    /**	Information about a file.
        @ingroup files
    */
    struct t_fileinfo {
        t_fourcc type;			///< type (four-char-code)
        t_fourcc creator;		///< Mac-only creator (four-char-code)
        t_uint32 unused;		///< this was date but it wasn't populated and it wasn't used
        t_int32 flags;			///< One of the values defined in #e_max_fileinfo_flags
    };


    /**	Retrieve the Path ID of the Max application.
        @ingroup files
        @return	The path id.
    */
    short path_getapppath(void);

    short path_getsupportpath(void);

    void path_namefrompathname(char* pathname, char* name);


    /**	Find a Max document by name in the search path.
        This is the preferred method for file searching since its introduction in Max version 4.

        This routine performs the same function as the routine path_getdefault().
        locatefile() searches through the directories specified by the user for
        Patcher files and tables in the File Preferences dialog as well as the
        current default path (see path_getdefault) and the directory
        containing the Max application

        @ingroup	files
        @version	4.0

        @param	name			The file name for the search, receives actual filename.
        @param	outvol			The Path ID of the file (if found).
        @param	outtype			The file type of the file (if found).
        @param	filetypelist	The file type(s) that you are searching for.
        @param	numtypes		The number of file types in the typelist array (1 if a single entry).

        @return			If a file is found with the name
                        specified by filename, locatefile returns 0, otherwise it returns non-zero.

        @remark			The old file search routines locatefile() and locatefiletype()
                        are still supported in Max 4, but the use of a new routine
                        locatefile_extended() is highly recommended. However,
                        locatefile_extended() has an important difference from
                        locatefile() and locatefiletype() that may require some rewriting
                        of your code. <em>It modifies its name parameter</em> in certain cases, while
                        locatefile() and locatefiletype() do not. The two cases where it
                        could modify the incoming filename string are 1) when an alias is
                        specified, the file pointed to by the alias is returned; and 2) when a full
                        path is specified, the output is the filename plus the path number of the
                        folder it's in.

                        This is important because many people pass the s_name field of a
                        #t_symbol to locatefile(). If the name field of a #t_symbol were to be
                        modified, the symbol table would be corrupted. To avoid this problem,
                        use strncpy_zero() to copy the contents of a #t_symbol to a character string first,
                        as shown below:
        @code
        char filename[MAX_FILENAME_CHARS];
        strncpy_zero(filename,str->s_name, MAX_FILENAME_CHARS);
        result = locatefile_extended(filename,&path,&type,typelist,1);
        @endcode
    */
    short locatefile_extended(char* name, short* outvol, t_fourcc* outtype, const t_fourcc* filetypelist, short numtypes);


    /**	Resolve a Path ID plus a (possibly extended) file name
        into a path that identifies the file's directory and a filename.
        This routine converts a name and Path ID to a standard form in which
        the name has no path information and does not refer to an aliased file.

        @ingroup files
        @param	name	A file name (which may be fully or partially qualified),
                        will contain the file name on return.
        @param	path	The Path ID to be resolved.
        @param	outpath	The Path ID of the returned file name.

        @return		 	Returns 0 if successful.
    */
    short path_resolvefile(char* name, const short path, short* outpath);


    /**	Retrive a #t_fileinfo structure from a file/path combination.
        @ingroup files
        @param	name	The file name to be queried.
        @param	path	The Path ID of the file.
        @param	info	The address of a #t_fileinfo structure to contain the file information.

        @return		 	Returns 0 if successful, otherwise it returns an OS-specific error code.
    */
    short path_fileinfo(const char* name, const short path, t_fileinfo* info);

    short path_tempfolder(void);
    short path_desktopfolder(void);
    short path_userdocfolder(void);
    short path_usermaxfolder(void);
    short preferences_path(const char *name, short create, short *path);

    short path_createfolder(const short path, const char* name, short* newpath);
    short path_copyfile(short srcpath, char* srcname, short dstpath, char* dstname);
    short path_copyfolder(short srcpath, short dstpath, char* dstname, long recurse, short* newpath);
    short C74_MUST_CHECK path_getpath(short path, const char* name, short* outpath);
    short path_getname(short path, char* name, short* outpath);


    /**	Create a fully qualified file name from a Path ID/file name combination.
        Unlike path_topotentialname(), this routine will only convert a
        pathname pair to a valid path string if the path exists.

        @ingroup files
        @param	path	The path to be used.
        @param	file	The file name to be used.
        @param	name	Loaded with the fully extended file name on return.
        @return		 	Returns 0 if successful, otherwise it returns an OS-specific error code.
    */
    short path_topathname(const short path, const char* file, char* name);


    /**
        Create a filename and Path ID combination from a fully qualified file name.
        Note that path_frompathname() does not require that the file actually exist.
        In this way you can use it to convert a full path you may have received as an
        argument to a file writing message to a form appropriate to provide to
        a routine such as path_createfile().

        @ingroup files
        @param	name		The extended file path to be converted.
        @param	path		Contains the Path ID on return.
        @param	filename	Contains the file name on return.
        @return		 		Returns 0 if successful, otherwise it returns an OS-specific error code.
    */
    short path_frompathname(const char* name, short* path, char* filename);


    short path_frompotentialpathname(const char* name, short* path, char* filename);


    /**	Determine the modification date of the selected path.

        @ingroup files
        @param	path	The Path ID of the directory to check.
        @param	date	The last modification date of the directory.
        @return			An error code.
    */
    short path_getmoddate(short path, t_ptr_uint* date);


    /**	Determine the modification date of the selected file.

        @ingroup files
        @param	filename	The name of the file to query.
        @param	path		The Path ID of the file.
        @param	date		The last modification date of the file upon return.
        @return				An error code.
    */
    short path_getfilemoddate(const char* filename, const short path, t_ptr_uint* date);


    short path_getfiledatesandsize(const char* filename, short path, t_uint64* create, t_uint64* mod, t_uint64* access, t_uint64* size);
    short path_getfilecreationdate(const char* filename, const short path, t_ptr_uint* date);
    short path_getfilesize(char* filename, short path, t_ptr_size* size);

    short path_nameinpath(const char* name, const short path, short* ref);


    /**	Prepare a directory for iteration.
        @ingroup files
        @param	path	The directory Path ID to open.
        @return			The return value of this routine is an internal "folder state" structure
                        used for further folder manipulation. It should be saved and used for
                        calls to path_foldernextfile() and path_closefolder().
                        If the folder cannot be found or accessed, path_openfolder() returns 0.
    */
    void* path_openfolder(short path);


    /**	Get the next file in the directory.
        In conjunction with path_openfolder() and path_closefolder(),
        this routine allows you to iterate through all of the files in a path.

        @ingroup files
        @param	xx			The "folder state" value returned by path_openfolder().
        @param	filetype	Contains the file type of the file type on return.
        @param	name		Contains the file name of the next file on return.
        @param	descend		Unused.
        @return				Returns non-zero if successful, and zero when there are no more files.
        @see				#e_max_path_folder_flags
    */
    short path_foldernextfile(void* xx, t_fourcc* filetype, char* name, short descend);


    /**	Complete a directory iteration.
        @ingroup files
        @param	x	The "folder state" value originally returned by path_openfolder().
    */
    void path_closefolder(void* x);


    /**	Convert a source path string to destination path string using the specified style and type.

        @ingroup files
        @param	src		A pointer to source character string to be converted.
        @param	dst		A pointer to destination character string.
        @param	style	The destination filepath style, as defined in #e_max_path_styles
        @param	type	The destination filepath type, as defined in #e_max_path_types
        @return			An error code.

        @see #MAX_PATH_CHARS
    */
    short path_nameconform(const char* src, char* dst, long style, long type);


    /**	Create a fully qualified file name from a Path ID/file name combination,
        regardless of whether or not the file exists on disk.

        @ingroup files
        @param	path	The path to be used.
        @param	file	The file name to be used.
        @param	name	Loaded with the fully extended file name on return.
        @param	check	Flag to check if a file with the given path exists.
        @return		 	Returns 0 if successful, otherwise it returns an OS-specific error code.

        @see path_topathname()
    */
    short path_topotentialname(const short path, const char* file, char* name, short check);


    /**	Translates a Max path+filename combo into a correct absolutepath that can be used to pass to libraries
        expecting system-native paths (i.e. POSIX on the Mac) that also handle multiple volumes correctly.

        @ingroup files
        @param	in_path			The Max path reference
        @param	in_filename		The name of the file in that path.
        @param	out_filepath	A string that is MAX_PATH_CHARS in length, which will receive the formatted absolute path upon return.
        @return					Returns 0 if successful.

        @see path_topotentialname()
        @see path_nameconform()
    */
    t_max_err path_toabsolutesystempath(const short in_path, const char* in_filename, char* out_filepath);


    /**	 Convert a path to an absolutepath as done by the absolutepath object in Max.

         @ingroup files
         @param	returned_path	Address to a symbol pointer which will be filled-in upon successful return.
         @param	s				Incoming (non-absolute) path.
         @param	filetypelist	The first of an array of file types to limit the search.
         @param	numtypes		The number of entries in the filetypelist parameter.
         @return				A Max error code.

         @see path_topathname()
         @see locatefile_extended()
     */
    t_max_err path_absolutepath(t_symbol** returned_path, const t_symbol* s, const t_fourcc* filetypelist, short numtypes);


    END_USING_C_LINKAGE

}} // namespace c74::max
