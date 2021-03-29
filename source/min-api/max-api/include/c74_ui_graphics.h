/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    BEGIN_USING_C_LINKAGE

    // opaque types for the users of jgraphics
    struct t_jgraphics;
    struct t_jpath;
    struct t_jpattern;
    struct t_jfont;
    struct t_jtextlayout;
    struct t_jtransform;
    struct t_jsurface;
    struct t_jdesktopui;
    struct t_jpopupmenu;
    struct t_jsvg;
    struct t_jsvg_remap;


    // misc utilities

    enum t_jgraphics_line_join {
        JGRAPHICS_LINE_JOIN_MITER,
        JGRAPHICS_LINE_JOIN_ROUND,
        JGRAPHICS_LINE_JOIN_BEVEL
    };

    enum t_jgraphics_line_cap {
        JGRAPHICS_LINE_CAP_BUTT,
        JGRAPHICS_LINE_CAP_ROUND,
        JGRAPHICS_LINE_CAP_SQUARE
    };

    enum t_jgraphics_bubble_side {
        JGRAPHICS_BUBBLE_SIDE_TOP,
        JGRAPHICS_BUBBLE_SIDE_LEFT,
        JGRAPHICS_BUBBLE_SIDE_BOTTOM,
        JGRAPHICS_BUBBLE_SIDE_RIGHT
    };

    enum t_jgraphics_path_type {
        JGRAPHICS_PATH_STARTNEWSUBPATH,
        JGRAPHICS_PATH_LINETO,
        JGRAPHICS_PATH_QUADRATICTO,
        JGRAPHICS_PATH_CUBICTO,
        JGRAPHICS_PATH_CLOSEPATH
    };


    /** Utility for rounding a double to an int.
        @ingroup jgraphics
        @param	d	floating-point input.
        @return		rounded int output.	*/
    int jgraphics_round(double d);



    // surfaces

    /**	Enumeration of color formats used by jgraphics surfaces.
        @ingroup	jgraphics			*/
    enum t_jgraphics_format {
        JGRAPHICS_FORMAT_ARGB32,		///< Color is represented using 32 bits, 8 bits each for the components, and including an alpha component.
        JGRAPHICS_FORMAT_RGB24,			///< Color is represented using 32 bits, 8 bits each for the components.  There is no alpha component.
        JGRAPHICS_FORMAT_A8				///< The color is represented only as an 8-bit alpha mask.
    //	JGRAPHICS_FORMAT_A1				// not supported
    };


    /**	Enumeration of file formats usable for jgraphics surfaces.
        @ingroup	jgraphics			*/
    enum t_jgraphics_fileformat {
        JGRAPHICS_FILEFORMAT_PNG,		///< Portable Network Graphics (PNG) format
        JGRAPHICS_FILEFORMAT_JPEG		///< JPEG format
    };


    /**	Create an image surface.
        Use jgraphics_surface_destroy() to free it when you are done.
        @ingroup		jsurface
        @param	format	Defines the color format for the new surface.
        @param	width	Defines the width of the new surface.
        @param	height	Defines the height of the new surface.
        @return			A pointer to the new surface.		*/
    t_jsurface* jgraphics_image_surface_create(t_jgraphics_format format, int width, int height);


    /**	Create an image surface, filling it with the contents of a file, and get a reference to the surface.
        Use jgraphics_surface_destroy() to release your reference to the surface when you are done.
        @ingroup			jsurface
        @param	filename	The name of the file.
        @param	path		The path id of the file.
        @return				A pointer to the new surface.		*/
    t_jsurface* jgraphics_image_surface_create_referenced(const char* filename, short path);


    /**	Create an image surface, filling it with the contents of a file.
        Use jgraphics_surface_destroy() to free it when you are done.
        @ingroup			jsurface
        @param	filename	The name of the file.
        @param	path		The path id of the file.
        @return				A pointer to the new surface.		*/
    t_jsurface* jgraphics_image_surface_create_from_file(const char* filename, short path);


    /**	Create an image surface from given pixel data.
        Data should point to start of top line of bitmap, stride tells how to get to next line.
        For upside down windows bitmaps, data = (pBits-(height-1)*stride) and stride is a negative number.
        @ingroup			jsurface
        @param	data		The data.  For example, an RGBA image loaded in memory.
        @param	format		The format of the data.
        @param	width		The width of the new surface.
        @param	height		The height of the new surface.
        @param	stride		The number of bytes between the start of rows in the dat buffer.
        @param	freefun		If not NULL, freefun will be called when the surface is destroyed
        @param	freearg		This will be passed to freefun if/when freefun is called.
        @return				A pointer to the new surface.	*/
    t_jsurface* jgraphics_image_surface_create_for_data(unsigned char* data, t_jgraphics_format format,
                                                        int width, int height, int stride,
                                                        method freefun, void* freearg);

    // Internal Use Only
    t_jsurface* jgraphics_image_surface_create_for_data_premult(unsigned char* data, t_jgraphics_format format,
                                                        int width, int height, int stride,
                                                        method freefun, void* freearg);

    /**	Create a new surface from file data.
        @ingroup		jsurface
        @param	data	A pointer to the raw PNG or JPG bits.
        @param	datalen	The number of bytes in data.
        @return			The new surface.
        @see			jgraphics_write_image_surface_to_filedata()	*/
    t_jsurface* jgraphics_image_surface_create_from_filedata(const void* data, unsigned long datalen);


    /**	Create a new surface from a resource in your external.
        @ingroup			jsurface
        @param	moduleRef	A pointer to your external's module,
                            which is passed to your external's main() function when the class is loaded.
        @param	resname		The name of the resource in the external.
        @remark				The following example shows an example of how this might be used in an external.
        @code
        static s_my_surface = NULL;

        int main(void* moduleRef)
        {
            // (Do typical class initialization here)

            // now create the surface from a resource that we added to the Xcode/VisualStudio project
            s_my_surface = jgraphics_image_surface_create_from_resource(moduleRef, "myCoolImage");

            return 0;
        }
        @endcode	*/
    t_jsurface* jgraphics_image_surface_create_from_resource(const void* moduleRef, const char* resname);

    /**	Low-level routine to access an object's resource data.
        @ingroup	jsurface
        @param	moduleRef	A pointer to your external's module, which is passed to your external's main() function when the class is loaded.
        @param	resname		Base name of the resource data (without an extension)
        @param	extcount	Count of possible extensions (ignored on Windows)
        @param	exts		Array of symbol atoms containing possible filename extensions (ignored on Windows)
        @param	data		Returned resource data assigned to a pointer you supply
        @param	datasize	Size of the data returned
        @remark				You are responsible for freeing any data returned in the data pointer
        @return		A Max error code.	*/
    t_max_err jgraphics_get_resource_data(const void* moduleRef, const char* resname, long extcount, t_atom* exts, void** data, unsigned long* datasize);

    /**	Create a reference to an existing surface.
        Use jgraphics_surface_destroy() to release your reference to the surface when you are done.
        @ingroup	jsurface
        @param	s	The surface to reference.
        @return		The new reference to the surface.	*/
    t_jsurface*	jgraphics_surface_reference(t_jsurface* s);


    /**	Release or free a surface.
        @ingroup	jsurface
        @param	s	The surface to release.		*/
    void		jgraphics_surface_destroy(t_jsurface* s);


    /**	Export a PNG file of the contents of a surface.
        @ingroup			jsurface
        @param	surface		The surface to export.
        @param	filename	Specify the name of the file to create.
        @param	path		Specify the path id for where to create the file.
        @param	dpi			Define the resolution of the image (e.g. 72).
        @return				A Max error code.	*/
    t_max_err	jgraphics_image_surface_writepng(t_jsurface* surface, const char* filename, short path, long dpi);

    /**	Export a JPEG file of the contents of a surface.
        @ingroup			jsurface
        @param	surface		The surface to export.
        @param	filename	Specify the name of the file to create.
        @param	path		Specify the path id for where to create the file.
        @return				A Max error code.	*/
    t_max_err	jgraphics_image_surface_writejpeg(t_jsurface* surface, const char* filename, short path);

    /** Get a writable bitmap of a surface.
        After you are done reading/writing to the bitmap, you should call jgraphics_image_surface_unlockpixels().
        @ingroup jsurface
        @param  s      The surface.
        @param  x      The rect horizontal-origin for the raw bitmap.
        @param  y      The rect vertical-origin for the raw bitmap.
        @param  width    The rect width for the bitmap.
        @param  height    The rect height for the bitmap.
        @param  linestride  The line stride for the bitmap.
        @param  pixelstride  The pixel stride for the bitmap.
        @return        A pointer to the raw bitmap.    */
    unsigned char* jgraphics_image_surface_lockpixels(t_jsurface *s,
                                  int x, int y, int width, int height,
                                  int *linestride, int *pixelstride);

    /** Unlock a surface locked by jgraphics_image_surface_lockpixels().
        @ingroup jsurface
        @param  s    The surface.
        @param  data  The pointer returned by jgraphics_image_surface_lockpixels().  */
    void    jgraphics_image_surface_unlockpixels(t_jsurface *s, const unsigned char *data);

    // Not used by any C74 code...
    void		jgraphics_surface_set_device_offset(t_jsurface* s, double x_offset, double y_offset);
    void		jgraphics_surface_get_device_offset(t_jsurface* s, double* x_offset, double* y_offset);


    /**	Retrieve the width of a surface.
        @ingroup		jsurface
        @param	s		The surface to query.
        @return			The width of the surface.	*/
    int			jgraphics_image_surface_get_width(t_jsurface* s);

    /**	Retrieve the height of a surface.
        @ingroup		jsurface
        @param	s		The surface to query.
        @return			The height of the surface.	*/
    int			jgraphics_image_surface_get_height(t_jsurface* s);

    /**	Set the color of an individual pixel in a surface.
        @ingroup		jsurface
        @param	s		The surface.
        @param	x		The horizontal coordinate of the pixel.
        @param	y		The vertical coordinate of the pixel.
        @param	color	The color of the pixel.		*/
    void		jgraphics_image_surface_set_pixel(t_jsurface* s, int x, int y, t_jrgba color);

    /**	Retrieve the color of an individual pixel in a surface.
        @ingroup		jsurface
        @param	s		The surface.
        @param	x		The horizontal coordinate of the pixel.
        @param	y		The vertical coordinate of the pixel.
        @param	color	The address of a valid #t_jrgba struct
                        whose values will be filled in with the color of the pixel upon return. 	*/
    void		jgraphics_image_surface_get_pixel(t_jsurface* s, int x, int y, char* color);

    /**
        @ingroup		jsurface
        @param	s		The surface to scroll.
        @param	x		The origin of the rect to scroll.
        @param	y		The origin of the rect to scroll.
        @param	width	The width of the rect to scroll.
        @param	height	The height of the rect to scroll.
        @param	dx		The amount to scroll the surface horizontally.
        @param	dy		The amount to scroll the surface vertically.
        @param	path	Can pass NULL if you are not interested in this info.
                        Otherwise pass a pointer and it will be returned with a path containing the invalid region.
    */
    void		jgraphics_image_surface_scroll(t_jsurface* s,
                                               int x, int y, int width, int height,
                                               int dx, int dy,
                                               t_jpath** path);		//


    /**	Draw an image surface.
        This not in cairo, but, it seems silly to have to make a brush to just draw an image.
        This doesn't support rotations, however.
        @ingroup			jsurface
        @param	g			The graphics context in which to draw the surface.
        @param	s			The surface to draw.
        @param	srcRect		The rect within the surface that should be drawn.
        @param	destRect	The rect in the context to which to draw the srcRect.
        @see				jgraphics_image_surface_draw_fast()		*/
    void		jgraphics_image_surface_draw(t_jgraphics* g, t_jsurface* s, t_rect srcRect, t_rect destRect);


    /**	Draw an image surface quickly.
        The draw_fast version won't scale based on zoom factor or user transforms so make sure that this is what you want!
        Draws entire image, origin *can* be shifted via zoom and user transforms
        (even though image is not scaled based on those same transforms)
        @ingroup			jsurface
        @param	g			The graphics context in which to draw the surface.
        @param	s			The surface to draw.
        @see	jgraphics_image_surface_draw						*/
    void		jgraphics_image_surface_draw_fast(t_jgraphics* g, t_jsurface* s);


    /**	Get surface data ready for manually writing to a file.
        @ingroup		jsurface
        @param	surf	The surface whose data will be retrieved.
        @param	fmt		The format for the data.  This should be a selection from #t_jgraphics_fileformat.
        @param	data	The address of a pointer that will be allocated and filled.
                        When you are done with this data you should free it using sysmem_freeptr().
        @param	size	The address of a variable to hold the size of the data upon return.

        @remark			A good example of this is to embed the surface as a PNG in a patcher file.
        @code
        long size = 0;
        void* data = NULL;

        jgraphics_write_image_surface_to_filedata(x->j_surface, JGRAPHICS_FILEFORMAT_PNG, &data, &size);
        if (size) {
            x->j_format = gensym("png");
            binarydata_appendtodictionary(data, size, gensym("data"), x->j_format, d);
            x->j_imagedata = data;
            x->j_imagedatasize = size;
        }
        @endcode
        @see			jgraphics_image_surface_create_from_filedata()	*/
    void jgraphics_write_image_surface_to_filedata(t_jsurface* surf, long fmt, void** data, long* size);

    t_jsurface* jgraphics_image_surface_create_from_base64(const char* base64, unsigned long datalen);
    void jgraphics_write_image_surface_to_base64(t_jsurface* surf, long fmt, char** base64, long* size);


    /** Set all pixels in rect to 0.
        @ingroup		jsurface
        @param	s		The surface to clear.
        @param	x		The horizontal origin of the rect to clear.
        @param	y		The vertical origin of the rect to clear.
        @param	width	The width of the rect to clear.
        @param	height	The height of the rect to clear.	*/
    void jgraphics_image_surface_clear(t_jsurface* s, int x, int y, int width, int height);



    // SVG Stuff

    /**	Read an SVG file, return a #t_jsvg object.
        @ingroup			jsvg
        @param	filename	The name of the file to read.
        @param	path		The path id of the file to read.
        @return				A new SVG object.	*/
    t_jsvg*		jsvg_create_from_file(const char* filename, short path);


    /**	Read an SVG file from a resource.
        @ingroup			jsvg
        @param	moduleRef	The external's moduleRef.
        @param	resname		The name of the SVG resource.
        @return				A new SVG object.
        @see				jgraphics_image_surface_create_from_resource()	*/
    t_jsvg*		jsvg_create_from_resource(const void* moduleRef, const char* resname);


    /**	Create an SVG object from a string containing the SVG's XML.
        @ingroup		jsvg
        @param	svgXML	The SVG source.
        @return			A new SVG object.	*/
    t_jsvg*		jsvg_create_from_xmlstring(const char* svgXML);


    /**	Retrieve the size of an SVG object.
        @ingroup		jsvg
        @param	svg		An SVG object.
        @param	width	The address of a variable that will be set to the width upon return.
        @param	height	The address of a variable that will be set to the width upon return.	*/
    void		jsvg_get_size(t_jsvg* svg, double* width, double* height);


    /**	Free a #t_jsvg object.
        @ingroup		jsvg
        @param	svg		The object to free.	*/
    void		jsvg_destroy(t_jsvg* svg);


    /**	Render an SVG into a graphics context.
        @ingroup		jsvg
        @param	svg		The SVG object to render.
        @param	g		The graphics context in which to render.	*/
    void		jsvg_render(t_jsvg* svg, t_jgraphics* g);

    void jsvg_load_cached(t_symbol* name, t_jsvg* *psvg);
    t_jsvg_remap* jsvg_remap_create(t_jsvg* svg);
    void jsvg_remap_addcolor(t_jsvg_remap* r, char* src, char* dst);
    void jsvg_remap_perform(t_jsvg_remap* r, t_jsvg** remapped);
    void jsvg_remap_destroy(t_jsvg_remap* r);

    void jgraphics_draw_jsvg(t_jgraphics* g, t_jsvg* svg, t_rect* r, int flags, double opacity);


    /**	Create a context to draw on a particular surface.
        When you are done, call jgraphics_destroy().
        @ingroup		jsurface
        @param	target	The surface to which to draw.
        @return			The new graphics context.	*/
    t_jgraphics*	jgraphics_create(t_jsurface* target);


    /**	Get a reference to a graphics context.
        When you are done you should release your reference with jgraphics_destroy().
        @ingroup	jgraphics
        @param	g	The context you wish to reference.
        @return		A new reference to the context. */
    t_jgraphics*	jgraphics_reference(t_jgraphics* g);


    /**	Release or free a graphics context.
        @ingroup	jgraphics
        @param	g	The context to release.		*/
    void			jgraphics_destroy(t_jgraphics* g);



    // Paths

    struct t_jgraphics_path_elem {
        float x1;
        float y1;
        float x2;
        float y2;
        float x3;
        float y3;
        t_jgraphics_path_type type;
    };

    /**	Begin a new path.
        This action clears any current path in the context.
        @ingroup	jgraphics
        @param	g	The graphics context.	*/
    void		jgraphics_new_path(t_jgraphics* g);


    /**	Get a copy of the current path from a context.
        @ingroup	jgraphics
        @param g	the graphics context containing the current path
        @return		A copy of the current path.
     */
    t_jpath*	jgraphics_copy_path(t_jgraphics* g);

    /** Create a new path consisting of the original path stroked with a given thickness
        @ingroup			jgraphics
        @param p			the path to be stroked
        @param thickness	thickness of the stroke
        @param join			the style to join segments together at corners
        @param cap			the style of end cap to use
        @return				the new path, which must be freed with jgraphics_path_destroy() when done
     */
    t_jpath*	jgraphics_path_createstroked(t_jpath* p, double thickness, t_jgraphics_line_join join, t_jgraphics_line_cap cap);

    /**	Release/free a path.
        @ingroup		jgraphics
        @param	path	The path to release.	*/
    void		jgraphics_path_destroy(t_jpath* path);


    /** Add a path to a graphics context.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	path	The path to add.	*/
    void		jgraphics_append_path(t_jgraphics* g, t_jpath* path);


    /**	Close the current path in a context.
        This will add a line segment to close current subpath.
        @ingroup	jgraphics
        @param	g	The graphics context.	*/
    void		jgraphics_close_path(t_jgraphics* g);


    /**	Round out any corners in a path.
        This action clears any current path in the context.
        @ingroup				jgraphics
        @param	g				The graphics context.
        @param	cornerRadius	The amount by which to round corners.	*/
    void		jgraphics_path_roundcorners(t_jgraphics* g, double cornerRadius);

    /** Test if the path contains the point x,y.
        @ingroup				jgraphics
        @param path				the path
        @param x				the x-coordinate of the point to test
        @param y				the y-coordinate of the point to test     */
    long		jgraphics_path_contains(t_jpath* path, double x, double y);

    /** Test if the path intersects the line defined by x1,y1 and x2,y2.
        @ingroup				jgraphics
        @param path				the path
        @param x1				the x-coordinate of the first point on the line
        @param y1				the y-coordinate of the first point on the line
        @param x2				the x-coordinate of the second point on the line
        @param y2				the y-coordinate of the second point on the line
     */
    long		jgraphics_path_intersectsline(t_jpath* path, double x1, double y1, double x2, double y2);

    /** Return the length of a path
        @ingroup				jgraphics
        @param path				the path
        @return					the length of the path
     */
    double		jgraphics_path_getlength(t_jpath* path);

    /** Return a point that lies a given distance from the start of the path
        @ingroup					jgraphics
        @param path					the path
        @param distancefromstart	distance from the start point
        @param x					pointer to double to receive the x position of the point
        @param y					pointer to double to receive the y position of the point
     */
    void		jgraphics_path_getpointalongpath(t_jpath* path, double distancefromstart, double* x, double* y);

    /** Finds the point on the path that is nearest to the point x,y passed in
        @ingroup					jgraphics
        @param path					the path to search
        @param x					x position of the target point
        @param y					y position of the target point
        @param path_x				pointer to double to receive the x position of closest point on path
        @param path_y				pointer to double to receive the y position of the closest point on path
        @return						returns the distance along the path from the path start position to the found point on the path
    */
    double		jgraphics_path_getnearestpoint(t_jpath* path, double x, double y, double* path_x, double* path_y);

    /** Get the path elements and return number of path elements
        @ingroup				jgraphics
        @param path				the path
        @param elems			pointer to array of path elements
        @return					the number of path elements
     */
    long		jgraphics_path_getpathelems(t_jpath* path, t_jgraphics_path_elem **elems);

    /**	Get the current location of the cursor in a graphics context.
        @ingroup	jgraphics
        @param	g	The graphics context.
        @param	x	The address of a variable that will be set to the horizontal cursor location upon return.
        @param	y	The address of a variable that will be set to the vertical cursor location upon return.	*/
    void		jgraphics_get_current_point(t_jgraphics* g, double* x, double* y);


    /**	Add a circular, clockwise, arc to the current path.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	xc		The horizontal coordinate of the arc's center.
        @param	yc		The vertical coordinate of the arc's center.
        @param	radius	The radius of the arc.
        @param	angle1	The starting angle of the arc in radians.
                        Zero radians is center right (positive x axis).
        @param	angle2	The terminal angle of the arc in radians.
                        Zero radians is center right (positive x axis).		*/
    void		jgraphics_arc(t_jgraphics* g,
                              double xc, double yc,
                              double radius,
                              double angle1, double angle2);

    // used by the dial object
    void jgraphics_piesegment(t_jgraphics* g,
                              double xc, double yc,
                              double radius,
                              double angle1, double angle2,
                              double innercircleproportionalsize);

    /**	Add a non-circular arc to the current path.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	xc		The horizontal coordinate of the arc's center.
        @param	yc		The vertical coordinate of the arc's center.
        @param	radiusx	The horizontal radius of the arc.
        @param	radiusy	The vertical radius of the arc.
        @param	angle1	The starting angle of the arc in radians.
                        Zero radians is center right (positive x axis).
        @param	angle2	The terminal angle of the arc in radians.
                        Zero radians is center right (positive x axis).		*/
    void		jgraphics_ovalarc(t_jgraphics* g,
                       double xc, double yc,
                       double radiusx, double radiusy,
                       double angle1, double angle2);


    /**	Add a circular, counter-clockwise, arc to the current path.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	xc		The horizontal coordinate of the arc's center.
        @param	yc		The vertical coordinate of the arc's center.
        @param	radius	The radius of the arc.
        @param	angle1	The starting angle of the arc in radians.
                        Zero radians is center right (positive x axis).
        @param	angle2	The terminal angle of the arc in radians.
                        Zero radians is center right (positive x axis).		*/
    void		jgraphics_arc_negative(t_jgraphics* g,
                              double xc, double yc,
                              double radius,
                              double angle1, double angle2);


    /**	Add a cubic Bezier spline to the current path.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x1		The first control point.
        @param	y1		The first control point.
        @param	x2		The second control point.
        @param	y2		The second control point.
        @param	x3		The destination point.
        @param	y3		The destination point.	*/
    void		jgraphics_curve_to(t_jgraphics* g,
                                   double x1, double y1,
                                   double x2, double y2,
                                   double x3, double y3);


    /**	Add a cubic Bezier spline to the current path, using coordinates relative to the current point.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x1		The first control point.
        @param	y1		The first control point.
        @param	x2		The second control point.
        @param	y2		The second control point.
        @param	x3		The destination point.
        @param	y3		The destination point.	*/
    void		jgraphics_rel_curve_to(t_jgraphics* g,
                                       double x1, double y1,
                                       double x2, double y2,
                                       double x3, double y3);


    /**	Add a line segment to the current path.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x		The destination point.
        @param	y		The destination point.	*/
    void		jgraphics_line_to(t_jgraphics* g, double x, double y);


    /**	Add a line segment to the current path, using coordinates relative to the current point.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x		The destination point.
        @param	y		The destination point.	*/
    void		jgraphics_rel_line_to(t_jgraphics* g, double x, double y);


    /** Move the cursor to a new point and begin a new subpath.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x		The new location.
        @param	y		The new location.	*/
    void		jgraphics_move_to(t_jgraphics* g, double x, double y);


    /** Move the cursor to a new point and begin a new subpath, using coordinates relative to the current point.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x		The new location.
        @param	y		The new location.	*/
    void		jgraphics_rel_move_to(t_jgraphics* g, double x, double y);


    /** Add a closed rectangle path in the context.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x		The horizontal origin.
        @param	y		The vertical origin.
        @param	width	The width of the rect.
        @param	height	The height of the rect.	*/
    void		jgraphics_rectangle(t_jgraphics* g,
                                    double x, double y,
                                    double width, double height);


    /** Deprecated -- do not use.  Adds a closed oval path in the context, however, it does not scale appropriately.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x		The horizontal origin.
        @param	y		The vertical origin.
        @param	width	The width of the oval.
        @param	height	The height of the oval.	*/
    void		jgraphics_oval(t_jgraphics* g,
                                    double x, double y,
                                    double width, double height);


    /** Add a closed rounded-rectangle path in the context.
        @ingroup			jgraphics
        @param	g			The graphics context.
        @param	x			The horizontal origin.
        @param	y			The vertical origin.
        @param	width		The width of the rect.
        @param	height		The height of the rect.
        @param	ovalwidth	The width of the oval used for the round corners.
        @param	ovalheight	The height of the oval used for the round corners.	*/
    void		jgraphics_rectangle_rounded(t_jgraphics* g,
                                            double x, double y,
                                            double width, double height,
                                            double ovalwidth, double ovalheight);


    /** Add a closed elliptical path in the context.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x		The horizontal origin.
        @param	y		The vertical origin.
        @param	width	The width of the rect.
        @param	height	The height of the rect.	*/
    void		jgraphics_ellipse(t_jgraphics* g,
                                  double x, double y,
                                  double width, double height);

    /** Add a closed bubble path in the context.
     @ingroup		jgraphics
     @param	g		The graphics context.
     @param	bodyx	Horizontal body origin.
     @param	bodyy	The vertical origin.
     @param	bodywidth	The width of the rect.
     @param	bodyheight	The height of the rect.
     @param cornersize	Body rounded corners
     @param arrowtipx	X position of arrow tip
     @param arrowtipy	Y position of arrow tip
     @param whichside	side to connect arrow, 0 = top, 1 = left, 2 = bottom, 3 = right,
     @param arrowedgeprop	Arrow proportion along edge (0-1)
     @param arrowwidth	Arrow base width
    */
    void		jgraphics_bubble(t_jgraphics* g,
                                  double bodyx, double bodyy,
                                  double bodywidth, double bodyheight,
                                  double cornersize, double arrowtipx, double arrowtipy,
                                  t_jgraphics_bubble_side whichside, double arrowedgeprop, double arrowwidth);


    /** Add a closed triangular path in the context.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	x1		Coordinate for the first point.
        @param	y1		Coordinate for the first point.
        @param	x2		Coordinate for the second point.
        @param	y2		Coordinate for the second point.
        @param	x3		Coordinate for the third point.
        @param	y3		Coordinate for the third point.
    */
    void jgraphics_triangle(t_jgraphics* g, double x1, double y1, double x2, double y2, double x3, double y3);


    // Internal use only
    void jgraphics_diagonal_line_fill(t_jgraphics* g, double pixels, double x, double y, double width, double height);


    /**	Enumeration of slanting options for font display.
        @ingroup	jfont			*/
    enum class t_jgraphics_font_slant {
        JGRAPHICS_FONT_SLANT_NORMAL,		///< Normal slanting (typically this means no slanting)
        JGRAPHICS_FONT_SLANT_ITALIC 		///< Italic slanting
        // JGRAPHICS_FONT_SLANT_OBLIQUE
    };


    /**	Enumeration of font weight options for font display.
        @ingroup	jfont			*/
    enum class t_jgraphics_font_weight {
        JGRAPHICS_FONT_WEIGHT_NORMAL,		///< Normal font weight
        JGRAPHICS_FONT_WEIGHT_BOLD			///< Bold font weight
    };


    /**	Specify a font for a graphics context.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	family	The name of the font family (e.g. "Arial").
        @param	slant	Define the slant to use for the font.
        @param	weight	Define the weight to use for the font.	*/
    void		jgraphics_select_font_face(t_jgraphics* g,
                                           const char* family,
                                           t_jgraphics_font_slant slant,
                                           t_jgraphics_font_weight weight);


    /**	Specify a font for a graphics context by passing a #t_jfont object.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	jfont	A jfont object whose attributes will be copied to the context.	*/
    void		jgraphics_select_jfont(t_jgraphics* g, t_jfont* jfont);


    /**	Specify the font size for a context.
        @ingroup		jgraphics
        @param	g		The graphics context.
        @param	size	The font size.	*/
    void		jgraphics_set_font_size(t_jgraphics* g, double size);


    /**	Turn underlining on/off for text in a context.
        @ingroup			jgraphics
        @param	g			The graphics context.
        @param	underline	Pass true or false to set the appropriate behavior.	*/
    void		jgraphics_set_underline(t_jgraphics* g, char underline);


    /**	Display text at the current position in a context.
        @ingroup			jgraphics
        @param	g			The graphics context.
        @param	utf8		The text to display.	*/
    void		jgraphics_show_text(t_jgraphics* g, const char* utf8);

    /**	Add a path of text to the current path.
     @ingroup			jgraphics
     @param	g			The graphics context.
     @param	utf8		The text to generate path for.	*/
    void		jgraphics_text_path(t_jgraphics* g, const char* utf8);


    /**	A structure for holding information related to how much space the rendering of a given font will use.
        The units for these measurements is in pixels.
        @ingroup	jgraphics
    */
    struct t_jgraphics_font_extents {
        double ascent;			///< The ascent.
        double descent;			///< The descent.
        double height;			///< The hieght.
        double max_x_advance;	///< Unused / Not valid.
        double max_y_advance; 	///< Unused / Not valid.
    };


    /**	Return the extents of the currently selected font for a given graphics context.
        @ingroup			jgraphics
        @param	g			Pointer to a jgraphics context.
        @param	extents		The address of a #t_jgraphics_font_extents structure to be filled with the results.
    */
    void jgraphics_font_extents(t_jgraphics* g, t_jgraphics_font_extents *extents);


    /**	Return the height and width of a string given current graphics settings in a context.
        @ingroup			jgraphics
        @param	g			Pointer to a jgraphics context.
        @param	utf8		A string containing the text whose dimensions we wish to find.
        @param	width		The address of a variable to be filled with the width of the rendered text.
        @param	height		The address of a variable to be filled with the height of the rendered text.
    */
    void jgraphics_text_measure(t_jgraphics* g, const char* utf8, double* width, double* height);


    /**	Return the height, width, and number of lines that will be used to render a given string.
        @ingroup					jgraphics
        @param	g					Pointer to a jgraphics context.
        @param	utf8				A string containing the text whose dimensions we wish to find.
        @param	wrapwidth			The number of pixels in width at which the text should be wrapped if it is too long.
        @param	includewhitespace	Set zero to not include white space in the calculation, otherwise set this parameter to 1.
        @param	width				The address of a variable to be filled with the width of the rendered text.
        @param	height				The address of a variable to be filled with the height of the rendered text.
        @param	numlines			The address of a variable to be filled with the number of lines required to render the text.
    */
    void jgraphics_text_measuretext_wrapped(t_jgraphics* g, const char* utf8, double wrapwidth, long includewhitespace,
                                        double* width, double* height, long* numlines);


    // Internal Use Only
    double jgraphics_getfontscale(void);



    // Working with fonts directly

    // Internal Use Only
    t_jfont*	jfont_create_from_maxfont(short number, short size);


    /**	Create a new font object.
        @ingroup jfont
        @param	family	The name of the font family (e.g. Arial).
        @param	slant	The type of slant for the font.
        @param	weight	The type of weight for the font.
        @param	size	The size of the font.
        @return			The new font object.	*/
    t_jfont*	jfont_create(const char* family,
                             t_jgraphics_font_slant slant,
                             t_jgraphics_font_weight weight,
                             double size);


    /**	Create new reference to an existing font object.
        @ingroup jfont
        @param	font	The font object for which to obtain a reference.
        @return			The new font object reference.	*/
    t_jfont*	jfont_reference(t_jfont* font);


    /**	Release or free a font object.
        @ingroup jfont
        @param	font	The font object to release.		*/
    void		jfont_destroy(t_jfont* font);


    /** Compare two fonts to see if they are equivalent.
        @ingroup jfont
        @param font		The first font object that is being compared.
        @param other	The second font object that is being compared.
        @return			Nonzero value if the two fonts are equivalent.   */
    long		jfont_isequalto(t_jfont* font, t_jfont* other);

    /** Set the name of the font family (e.g. Arial).
        @ingroup    jfont
        @param      font    The font object.
        @param      family  A t_symbol containing the name of the desired font family.  */
    void        jfont_set_family(t_jfont* font, t_symbol* family);

    /** Get the name of the font family (e.g. Arial).
        @ingroup    jfont
        @param      font    The font object.
        @return             A t_symbol representing the name of the font family. */
    t_symbol* jfont_get_family(t_jfont* font);

    /** Set the slant of the font.
        @ingroup    jfont
        @param      font    The font object
        @param      slant   The desired slant. */
    void jfont_set_slant(t_jfont* font, t_jgraphics_font_slant slant);

    /** Get the slant of the font.
        @ingroup    jfont
        @param      font    The font object.
        @return             The current slant setting for the font. */
    t_jgraphics_font_slant jfont_get_slant(t_jfont* font);

    /** Set the weight of the font.
     @ingroup    jfont
     @param      font    The font object
     @param      weight   The desired weight (e.g. bold). */
    void jfont_set_weight(t_jfont* font, t_jgraphics_font_weight weight);

    /** Get the weight of the font.
     @ingroup    jfont
     @param      font    The font object.
     @return             The current weight setting for the font. */
    t_jgraphics_font_weight jfont_get_weight(t_jfont* font);

    /** Set the size of a font object.
        @ingroup		jfont
        @param	font	The font object.
        @param	size	The new size for the font object.	*/
    void		jfont_set_font_size(t_jfont* font, double size);

    /** Get the size of a font object.
        @ingroup        jfont
        @param  font    The font object.
        @return         The size of the font. */
    double      jfont_get_font_size(t_jfont* font);

    /** Set the underlining of a font object.
        @ingroup		jfont
        @param	font	The font object.
        @param	ul		Pass true to underline, or false for no underlining.	*/
    void		jfont_set_underline(t_jfont* font, char ul);

    /** Get the underline state of a font object.
        @ingroup        jfont
        @param	font	The font object.
        @return         Nonzero value if the font will be underlined. */
    char        jfont_get_underline(t_jfont* font);

    double		jfont_get_heighttocharheightratio(t_jfont* font);


    /**	Get extents of this font
        @ingroup		jfont
        @param	font	The font object.
        @param	extents	The font extents upon return/	*/
    void		jfont_extents(t_jfont* font, t_jgraphics_font_extents *extents);


    /**	Given a font, find out how much area is required to render a string of text.
        @ingroup		jfont
        @param	font	The font object.
        @param	utf8	The text whose rendering will be measured.
        @param	width	The address of a variable to hold the width upon return.
        @param	height	The address of a variable to hold the height upon return.	*/
    void		jfont_text_measure(t_jfont* font, const char* utf8, double* width, double* height);


    /**	Given a font, find out how much area is required to render a string of text,
        provided a horizontal maximum limit at which the text is wrapped.
        @ingroup					jfont
        @param	font				The font object.
        @param	utf8				The text whose rendering will be measured.
        @param	wrapwidth			The maximum width, above which text should wrap onto a new line.
        @param	includewhitespace	If non-zero, include whitespace in the measurement.
        @param	width				The address of a variable to hold the width upon return.
        @param	height				The address of a variable to hold the height upon return.
        @param	numlines			The address of a variable to hold the number of lines of text after wrapping upon return.	*/
    void		jfont_text_measuretext_wrapped(t_jfont* font, const char* utf8, double wrapwidth, long includewhitespace,
                                           double* width, double* height, long* numlines);

    /** Given a font, find out the width and height of the 'M' character.
        This is equivalent to jfont_text_measure(font, "M", width, height) but is faster.
        @ingroup					jfont
        @param	font				The font object.
        @param	width	The address of a variable to hold the width upon return.
        @param	height	The address of a variable to hold the height upon return.	*/
    void		jfont_get_em_dimensions(t_jfont* font, double* width, double* height);

    /**	Get a list of font names.
        @ingroup		jfont
        @param	count	The addres of a variable to hold the count of font names in list upon return.
        @param	list	The address of a #t_symbol** initialized to NULL.
                        Upon return this will be set to an array of count #t_symbol pointers.
                        This array should be freed using sysmem_freeptr() when you are done with it.
        @return			A Max error code.	*/
    t_max_err	jfont_getfontlist(long* count, t_symbol*** list);

    long jfont_isfixedwidth(const char* name);

    const char* jfont_get_default_fixedwidth_name(void);

    // Internal Use Only -- not exported
    void		jfont_set_juce_default_fontname(char* s);
    void		jfont_copy_juce_default_fontname(char* s, long maxlen);
    void		jfont_copy_juce_platform_fontname(char* s, long maxlen);
    void		jfont_set_juce_fallback_fontname(char* s);
    void		jfont_copy_juce_fallback_fontname(char* s, long maxlen);


    /**	Determine if you can anti-alias text to a transparent background.
         You might want to call this and then disable "useimagebuffer" if false *and*
         you are rendering text on a transparent background.
        @ingroup	jgraphics
        @return		Non-zero if you can anti-alias text to a transparent background.	*/
    long jgraphics_system_canantialiastexttotransparentbg();


    long jgraphics_fontname_hasglyph(char* name, long code);


    /**	Create a new textlayout object.
        @ingroup	textlayout
        @return		The new textlayout object.	*/
    t_jtextlayout*	jtextlayout_create();


    /**	Create a new textlayout object.
         This gives a hint to the textlayout as to what the text bgcolor will be.
        It won't actually paint the bg for you.
        But, it does let it do a better job.
        @ingroup		textlayout
        @param	g		The graphics context for the textlayout.
        @param	bgcolor	The background color for the textlayout.
        @return			The new textlayout object.	*/
    t_jtextlayout*	jtextlayout_withbgcolor(t_jgraphics* g, t_jrgba* bgcolor);


    /**	Release/free a textlayout object.
        @ingroup			textlayout
        @param	textlayout	The textlayout object to release.	*/
    void			jtextlayout_destroy(t_jtextlayout* textlayout);


    /**	Enumeration of text justification options, which are specified as a bitmask.
        @ingroup	jgraphics			*/
    enum t_jgraphics_text_justification {
        JGRAPHICS_TEXT_JUSTIFICATION_LEFT = 1,			///< Justify left
        JGRAPHICS_TEXT_JUSTIFICATION_RIGHT = 2,			///< Justify right
        JGRAPHICS_TEXT_JUSTIFICATION_HCENTERED = 4,		///< Centered horizontally
        JGRAPHICS_TEXT_JUSTIFICATION_TOP = 8, 			///< Justified to the top
        JGRAPHICS_TEXT_JUSTIFICATION_BOTTOM = 16,		///< Justified to the bottom
        JGRAPHICS_TEXT_JUSTIFICATION_VCENTERED = 32,	///< Centered vertically
        JGRAPHICS_TEXT_JUSTIFICATION_HJUSTIFIED = 64,	///< Horizontally justified
        JGRAPHICS_TEXT_JUSTIFICATION_CENTERED = JGRAPHICS_TEXT_JUSTIFICATION_HCENTERED + JGRAPHICS_TEXT_JUSTIFICATION_VCENTERED	///< Shortcut for Centering both vertically and horizontally
    };


    /**	Flags for setting text layout options.
        @ingroup	textlayout			*/
    enum t_jgraphics_textlayout_flags {
        JGRAPHICS_TEXTLAYOUT_NOWRAP = 1,		///< disable word wrapping
        JGRAPHICS_TEXTLAYOUT_USEELLIPSIS = 3	///< show ... if a line doesn't fit (implies NOWRAP too)
    };


    /**	Set the text and attributes of a textlayout object.
        @ingroup				textlayout
        @param	textlayout		The textlayout object.
        @param	utf8			The text to render.
        @param	jfont			The font with which to render the text.
        @param	x				The text is placed within rect specified by x, y, width, height.
        @param	y				The text is placed within rect specified by x, y, width, height.
        @param	width			The text is placed within rect specified by x, y, width, height.
        @param	height			The text is placed within rect specified by x, y, width, height.
        @param	justification	How to justify the text within the rect.
        @param	flags			Additional flags to control behaviour.	*/
    void	  jtextlayout_set(t_jtextlayout* textlayout,
                              const char* utf8,
                              t_jfont* jfont,
                              double x, double y,
                              double width,
                              double height,
                              t_jgraphics_text_justification justification,
                              t_jgraphics_textlayout_flags flags);

    /**	Set the text of a textlayout object.
        @ingroup				textlayout
        @param	textlayout		The textlayout object.
        @param	utf8			The text to render.
        @param	jfont			The font with which to render the text.*/
    void	  jtextlayout_settext(t_jtextlayout* textlayout,
                            const char* utf8,
                            t_jfont* jfont);

    /**	Set the color to render text in a textlayout object.
        @ingroup			textlayout
        @param	textlayout	The textlayout object for which to set the color.
        @param	textcolor	The color for the text.			*/
    void	  jtextlayout_settextcolor(t_jtextlayout* textlayout, t_jrgba* textcolor);


    /**	Return a measurement of how much space will be required to draw the text of a textlayout.
        @ingroup					textlayout
        @param	textlayout			The textlayout object to query.
        @param	startindex			You can measure a subset of the characters.
                                    This defines the character from which to start.
        @param	numchars			Pass -1 for all characters from startindex to end
        @param	includewhitespace	Define whether to measure with or without whitespace truncated from edges.
        @param	width				Returns the width of text not including any margins.
        @param	height				Returns the height of text not including any margins.
        @param	numlines			Returns the number of lines of text.	*/
    void	  jtextlayout_measuretext(t_jtextlayout* textlayout,
                                  long startindex,
                                  long numchars,
                                  long includewhitespace,
                                  double* width, double* height,
                                  long* numlines);


    /**	Draw a textlayout in a given graphics context.
        @ingroup		textlayout
        @param	tl		The textlayout object to query.
        @param	g		The graphics context in which to draw the text.		*/
    void	  jtextlayout_draw(t_jtextlayout* tl, t_jgraphics* g);


    /** Retrieve a count of the number of characters in a textlayout object.
        @ingroup		textlayout
        @param	tl		The textlayout object to query.
        @return			The number of characters.		*/
    long	  jtextlayout_getnumchars(t_jtextlayout* tl);


    /**	Retrieve the #t_rect containing a character at a given index.
        @ingroup		textlayout
        @param	tl		The textlayout object to query.
        @param	index	The index from which to fetch the unicode character.
        @param	rect	The address of a valid #t_rect which will be filled in upon return.
        @return			A Max error code.				*/
    t_max_err jtextlayout_getcharbox(t_jtextlayout* tl, long index, t_rect* rect);


    /**	Retrieve the unicode character at a given index.
        @ingroup		textlayout
        @param	tl		The textlayout object to query.
        @param	index	The index from which to fetch the unicode character.
        @param	pch		The address of a variable to hold the unicode character value upon return.
        @return			A Max error code.				*/
    t_max_err jtextlayout_getchar(t_jtextlayout* tl, long index, long* pch);


    /** Create a t_jpath object representing the text layout.
        @ingroup        textlayout
        @param  tl      The textlayout object to retrieve a path for.
        @return         A t_jpath. When finished with the path free it with jgraphics_path_destroy. */
    t_jpath* jtextlayout_createpath(t_jtextlayout* tl);


    /** An affine transformation (such as scale, shear, etc).
        @ingroup jmatrix	*/
    struct t_jmatrix {
        double xx;		///< xx component
        double yx;		///< yx component
        double xy;		///< xy component
        double yy;		///< yy component
        double x0;		///< x translation
        double y0;		///< y translation
    };

    /**	Set a #t_jmatrix to an affine transformation.
        @ingroup	jmatrix
        @param	x	x
        @param	xx	xx
        @param	yx	yx
        @param	xy	xy
        @param	yy	yy
        @param	x0	x0
        @param	y0	y0
        @remark		given x,y the matrix specifies the following transformation:
        @code
        xnew = xx * x + xy * y + x0;
        ynew = yx * x + yy * y + y0;
        @endcode
    */
    void jgraphics_matrix_init(t_jmatrix* x, double xx, double yx, double xy, double yy, double x0, double y0);

    /** Modify a matrix to be an identity transform.
        @ingroup	jmatrix
        @param	x	The #t_jmatrix.
    */
    void jgraphics_matrix_init_identity(t_jmatrix* x);

    /**	Initialize a #t_jmatrix to translate (offset) a point.
        @ingroup	jmatrix
        @param	x	The #t_jmatrix.
        @param	tx	The amount of x-axis translation.
        @param	ty	The amount of y-axis translation.
    */
    void jgraphics_matrix_init_translate(t_jmatrix* x, double tx, double ty);

    /**	Initialize a #t_jmatrix to scale (offset) a point.
        @ingroup	jmatrix
        @param	x	The #t_jmatrix.
        @param	sx	The horizontal scale factor.
        @param	sy	The vertical scale factor.
    */
    void jgraphics_matrix_init_scale(t_jmatrix* x, double sx, double sy);

    /**Initialize a #t_jmatrix to rotate (offset) a point.
        @ingroup		jmatrix
        @param	x		The #t_jmatrix.
        @param	radians	The angle or rotation in radians.
    */
    void jgraphics_matrix_init_rotate(t_jmatrix* x, double radians);


    /**	Apply a translation to an existing matrix.
        @ingroup	jmatrix
        @param	x	The #t_jmatrix.
        @param	tx	The amount of x-axis translation.
        @param	ty	The amount of y-axis translation.
    */
    void jgraphics_matrix_translate(t_jmatrix* x, double tx, double ty);

    /** Apply a scaling to an existing matrix.
        @ingroup	jmatrix
        @param	x	The #t_jmatrix.
        @param	sx	The horizontal scale factor.
        @param	sy	The vertical scale factor.
    */
    void jgraphics_matrix_scale(t_jmatrix* x, double sx, double sy);

    /** Apply a rotation to an existing matrix.
        @ingroup	jmatrix
        @param	x	The #t_jmatrix.
        @param	radians	The angle or rotation in radians.
    */
    void jgraphics_matrix_rotate(t_jmatrix* x, double radians);

    /**	Invert an existing matrix.
        @ingroup	jmatrix
        @param	x	The #t_jmatrix.
    */
    void jgraphics_matrix_invert(t_jmatrix* x);

    /** Multiply two matrices: resulting matrix has effect of first applying a and then applying b.
        @ingroup		jmatrix
        @param	result	The resulting product #t_jmatrix.
        @param	a		The first operand.
        @param	b		The second operand.
    */
    void jgraphics_matrix_multiply(t_jmatrix* result, const t_jmatrix* a, const t_jmatrix* b);


    /** Transform a point using a #t_jmatrix transormation.
        @ingroup		jmatrix
        @param	matrix	The #t_jmatrix.
        @param	x		The address of the variable holding the x coordinate.
        @param	y		The address of the variable holding the y coordinate.
    */
    void jgraphics_matrix_transform_point(const t_jmatrix* matrix, double* x, double* y);




    // Patterns
    t_jpattern*	jgraphics_pattern_create_rgba(double red,
                                              double green,		// colors between 0 and 1
                                              double blue,
                                              double alpha);	// solid, set alpha to 1.0

    t_jpattern*	jgraphics_pattern_create_for_surface(t_jsurface* surface);

    t_jpattern* jgraphics_pattern_create_linear(double x0, double y0, double x1, double y1);

    /*
        cx0 : x coordinate for the center of the start circle
        cy0 : y coordinate for the center of the start circle
        radius0 : radius of the start circle
        cx1 : x coordinate for the center of the end circle
        cy1 : y coordinate for the center of the end circle
        radius1 : radius of the end circle
    */
    t_jpattern* jgraphics_pattern_create_radial(double cx0, double cy0, double radius0, double cx1, double cy1, double radius1);

    void jgraphics_pattern_add_color_stop_rgba(t_jpattern* pattern, double offset, double red, double green, double blue, double alpha);

    void jgraphics_pattern_add_color_for_proportion(t_jpattern* pattern, double proportion);

    t_jpattern* jgraphics_pattern_reference(t_jpattern* pattern);
    void		jgraphics_pattern_destroy(t_jpattern* pattern);

    enum t_jgraphics_pattern_type {
        JGRAPHICS_PATTERN_TYPE_SOLID,
        JGRAPHICS_PATTERN_TYPE_SURFACE,
        JGRAPHICS_PATTERN_TYPE_LINEAR,
        JGRAPHICS_PATTERN_TYPE_RADIAL
    };

    t_jgraphics_pattern_type jgraphics_pattern_get_type(t_jpattern* pattern);

    enum t_jgraphics_extend {
        JGRAPHICS_EXTEND_NONE,
        JGRAPHICS_EXTEND_REPEAT,
        JGRAPHICS_EXTEND_REFLECT,
        JGRAPHICS_EXTEND_PAD
    };

    // rbs -- JGRAPHICS_EXTEND_NONE for images isn't actually supported yet
    static const int JGRAPHICS_EXTEND_GRADIENT_DEFAULT = JGRAPHICS_EXTEND_PAD;
    static const int JGRAPHICS_EXTEND_SURFACE_DEFAULT = JGRAPHICS_EXTEND_NONE;

    // These functions are placeholders for Cairo functionality, but for which there is no direct JUCE implementation.
    // They may or may not be implemented in the future.
    void jgraphics_pattern_set_extend(t_jpattern* pattern, t_jgraphics_extend extend);
    t_jgraphics_extend jgraphics_pattern_get_extend(t_jpattern* pattern);

    void jgraphics_pattern_set_matrix(t_jpattern* pattern, const t_jmatrix* matrix);
    void jgraphics_pattern_get_matrix(t_jpattern* pattern, t_jmatrix* matrix);
    // pattern matrix convenience functions
    void jgraphics_pattern_translate(t_jpattern* pattern, double tx, double ty);
    void jgraphics_pattern_scale(t_jpattern* pattern, double sx, double sy);
    void jgraphics_pattern_rotate(t_jpattern* pattern, double angle);
    void jgraphics_pattern_transform(t_jpattern* pattern, const t_jmatrix* matrix);
    void jgraphics_pattern_identity_matrix(t_jpattern* pattern);
    t_jsurface* jgraphics_pattern_get_surface(t_jpattern* pattern);

    // Transforms
    void		jgraphics_translate(t_jgraphics* g, double tx, double ty);
    void		jgraphics_scale(t_jgraphics* g, double sx, double sy);
    void		jgraphics_rotate(t_jgraphics* g, double angle);
    void		jgraphics_transform(t_jgraphics* g, const t_jmatrix* matrix);
    void		jgraphics_set_matrix(t_jgraphics* g, const t_jmatrix* matrix);
    void		jgraphics_get_matrix(t_jgraphics* g, t_jmatrix* matrix);
    void		jgraphics_identity_matrix(t_jgraphics* g);


    /**	User coordinates are those passed to drawing functions in a given #t_jgraphics context.
          Device coordinates refer to patcher canvas coordinates, before any zooming.
        @ingroup jgraphics		*/
    void		jgraphics_user_to_device(t_jgraphics* g,
                                         double* x,
                                         double* y);

    /**	User coordinates are those passed to drawing functions in a given #t_jgraphics context.
          Device coordinates refer to patcher canvas coordinates, before any zooming.
        @ingroup jgraphics		*/
    void		jgraphics_device_to_user(t_jgraphics* g,
                                         double* x,
                                         double* y);

    // Graphics

    void		jgraphics_save(t_jgraphics* g);				// doesn't save/restore the path
    void		jgraphics_restore(t_jgraphics* g);

    t_jsurface* jgraphics_get_target(t_jgraphics* g);


    // Pushing and Popping groups is not currently exported or supported by Max.
    void jgraphics_push_group(t_jgraphics* g);
    t_jpattern* jgraphics_pop_group(t_jgraphics* g);
    void jgraphics_pop_group_to_source(t_jgraphics* g);
    t_jsurface* jgraphics_get_group_target(t_jgraphics* g);
    // jgraphics_pop_group_surface is not in cairo, but equivalent to the following sequence
    // jgraphics_get_group_target(), jgraphics_surface_reference(), jgraphics_restore()
    t_jsurface* jgraphics_pop_group_surface(t_jgraphics* g);


    // if a t_jpattern source was previously selected, it is removed
    void		jgraphics_set_source_rgba(t_jgraphics* g,
                                          double red,
                                          double green,
                                          double blue,
                                          double alpha);

    void		jgraphics_set_source_jrgba(t_jgraphics* g, t_jrgba* rgba);

    void		jgraphics_set_source_rgb(t_jgraphics* g,
                                         double red,
                                         double green,
                                         double blue);
    // if NULL source is passed in will revert to prior solid color.
    void		jgraphics_set_source(t_jgraphics* g, t_jpattern* source);

    // convenience function for creating pattern from surface and making it the source for g
    void		jgraphics_set_source_surface(t_jgraphics* g,
                                             t_jsurface* surface,
                                             double x, double y);


    enum t_jgraphics_pattern_shared {
        JGRAPHICS_PATTERN_GRAY = 0,
        JGRAPHICS_NUM_SHARED_PATTERNS
    };


    // We create some standard patterns which are "owned" by the jgraphics library.
    // You can use these as a source for filling or stroking paths.
    // The gray patter above is what is used to put dotted lines around the comment box, and others.
    void		jgraphics_set_source_shared(t_jgraphics* g, t_jgraphics_pattern_shared patindex);

    // color transforms
    // each component (rgba) has a scale and offset value as part of the graphics context.
    // this is saved and restored with the jgraphics_save and jgraphics_restore calls.

    void		jgraphics_scale_source_rgba(t_jgraphics* g,
                                            double redscale,
                                            double greenscale,
                                            double bluescale,
                                            double alphascale);

    void		jgraphics_translate_source_rgba(t_jgraphics* g,
                                                double redoffset,
                                                double greenoffset,
                                                double blueoffset,
                                                double alphaoffset);

    // example use of this function is in the gswitch object
    void		jgraphics_set_dash(t_jgraphics* g,
                                   double* dashes,
                                   int numdashes,
                                   double offset);		// offset not supported yet


    enum t_jgraphics_fill_rule {
        JGRAPHICS_FILL_RULE_WINDING,
        JGRAPHICS_FILL_RULE_EVEN_ODD
    };

    void					jgraphics_set_fill_rule(t_jgraphics* g, t_jgraphics_fill_rule fill_rule);
    t_jgraphics_fill_rule	jgraphics_get_fill_rule(t_jgraphics* g);

    void					jgraphics_set_line_cap(t_jgraphics* g, t_jgraphics_line_cap line_cap);
    t_jgraphics_line_cap	jgraphics_get_line_cap(t_jgraphics* g);

    void		jgraphics_set_line_join(t_jgraphics* g,
                                        t_jgraphics_line_join line_join);
    t_jgraphics_line_join	jgraphics_get_line_join(t_jgraphics* g);

    void		jgraphics_set_line_width(t_jgraphics* g,
                                         double width);
    double		jgraphics_get_line_width(t_jgraphics* g);

    // perhaps someday we could expose something like this
    //			jgraphics_set_operator()

    void		jgraphics_fill(t_jgraphics* g);
    void		jgraphics_fill_preserve(t_jgraphics* g);
    void		jgraphics_fill_with_alpha(t_jgraphics* g, double alpha);
    void		jgraphics_fill_preserve_with_alpha(t_jgraphics* g, double alpha);
    // Note: you can use jgraphics_image_surface_create with a 1x1 offscreen to do path stuff
    //       that isn't actually going to be used for drawing.
    int jgraphics_in_fill(t_jgraphics* g, double x, double y);		// hit test
    int	jgraphics_path_intersects_line(t_jgraphics* g, double x1, double y1, double x2, double y2);		// not in cairo

    // various utilities
    int jgraphics_ptinrect(t_pt pt, t_rect rect);

    int jgraphics_lines_intersect(double l1x1, double l1y1, double l1x2, double l1y2, double l2x1, double l2y1, double l2x2, double l2y2, double* ix, double* iy);
    int jgraphics_line_intersects_rect(double linex1, double liney1, double linex2, double liney2, t_rect r, double* ix, double* iy);

    // jgraphics_ptaboveline: returns non-zero if the given point is above the line
    // specified by the two points: lx1, ly1 to ending point lx2, ly2
    int jgraphics_ptaboveline(t_pt pt, double lx1, double ly1, double lx2, double ly2);

    // return nonzero if points a and b are on the same side of the line specified by lx1, ly1 -> lx2, ly2
    int jgraphics_points_on_same_side_of_line(t_pt a, t_pt b, double lx1, double ly1, double lx2, double ly2);

    /*
        Note that the functions jgraphics_rectangle_rounded() and jgraphics_ptinroundedrect()
        need different size arguments as ovalsize and ovalwidth.
    */
    int jgraphics_ptinroundedrect(t_pt pt, t_rect rect, double ovalwidth, double ovalheight);

    // get extents of current path in device coordinates (after transform matrix)
    void jgraphics_fill_extents(t_jgraphics* g,
                                double* x1, double* y1,
                                double* x2, double* y2);

    // paints current source using alpha of pattern
    //void		jgraphics_mask(t_jgraphics* x,
    //						   t_jpattern* pattern);

    // paints current source using alpha of surface
    //void		jgraphics_mask_surface(t_jgraphics* g,
    //								   t_jsurface* surface,
    //								   double surface_x,	// surface origin
    //								   double surface_y);	// surface origin

    // paint current source in entire current clip region
    void		jgraphics_paint(t_jgraphics* g);
    void		jgraphics_paint_with_alpha(t_jgraphics* g,
                                           double alpha);

    void		jgraphics_stroke(t_jgraphics* g);
    void		jgraphics_stroke_preserve(t_jgraphics* g);
    void		jgraphics_stroke_with_alpha(t_jgraphics* g, double alpha);
    void		jgraphics_stroke_preserve_with_alpha(t_jgraphics* g, double alpha);

    // fast non antialiasing/rotating versions
    void jgraphics_rectangle_fill_fast(t_jgraphics* g, double x, double y, double width, double height);
    void jgraphics_rectangle_draw_fast(t_jgraphics* g, double x, double y, double width, double height, double border);
    void jgraphics_line_draw_fast(t_jgraphics* g, double x1, double y1, double x2, double y2, double linewidth);

    // desktopui API: so externals can create transparent popup windows, draw to them, and receive mouse events

    enum t_jdesktopui_flags {
        JDESKTOPUI_FLAGS_FIRSTFLAG = 1				// no flags defined yet, but this is a placeholder
    };

    t_jdesktopui* jdesktopui_new(t_object* owner, t_jdesktopui_flags flags, t_rect rect);
    void jdesktopui_destroy(t_jdesktopui *x);
    void jdesktopui_setvisible(t_jdesktopui *x, long way);
    void jdesktopui_setalwaysontop(t_jdesktopui *x, long way);
    void jdesktopui_setrect(t_jdesktopui *x, t_rect rect);
    void jdesktopui_getrect(t_jdesktopui *x, t_rect* rect);
    void jdesktopui_setposition(t_jdesktopui *x, t_pt pt);
    void jdesktopui_setfadetimes(t_jdesktopui *x, int fade_in_ms, int fade_out_ms);
    t_jgraphics* jdesktopui_get_jgraphics(t_jdesktopui *x);
    void jdesktopui_redraw(t_jdesktopui *x);
    void jdesktopui_redrawrect(t_jdesktopui *x, t_rect rect);
    double jdesktopui_getopacity(t_jdesktopui *x);
    void* jdesktopui_createtimer(t_jdesktopui *x, t_symbol* msg, void* arg);
    void jdesktopui_starttimer(void* ref, int interval);
    void jdesktopui_stoptimer(void* ref, int alsodelete);
    void jdesktopui_destroytimer(void* ref);

    // color transformations

    t_jrgba jgraphics_jrgba_contrasting(char* c, double amount);
    t_jrgba jgraphics_jrgba_contrastwith(char* c1, char* c2);
    t_jrgba jgraphics_jrgba_darker(char* c, double amount);
    t_jrgba jgraphics_jrgba_brighter(char* c, double amount);
    t_jrgba jgraphics_jrgba_overlay(char* c1, char* c2);
    t_jrgba jgraphics_jrgba_interpolate(char* c1, char* c2, double proportion);
    void jgraphics_jrgba_gethsb(char* c, double* h, double* s, double* b);
    t_jrgba jgraphics_jrgba_fromhsb(double h, double s, double b, double a);

    long jcolor_getcolor(t_symbol* name, char* on, char* off);

    // popup menu API so externals can create popup menus that can also be drawn into

    /**	Create a pop-up menu.
        Free this pop-up menu using jpopupmenu_destroy().
        @ingroup	jpopupmenu
        @return		A pointer to the newly created jpopupmenu object. */
    t_jpopupmenu* jpopupmenu_create();

    /**	Free a pop-up menu created with jpopupmenu_create().
        @ingroup		jpopupmenu
        @param	menu	The pop-up menu to be freed.		*/
    void jpopupmenu_destroy(t_jpopupmenu* menu);

    /**	Clear the conents of a pop-up menu.
        @ingroup		jpopupmenu
        @param	menu	The pop-up menu whose contents will be cleared.	*/
    void jpopupmenu_clear(t_jpopupmenu* menu);

    // Internal use only
    void jpopupmenu_setitemcallback(method fun, void* arg);

    /**	Set the colors used by a pop-up menu.
        @ingroup				jpopupmenu
        @param	menu			The pop-up menu to which the colors will be applied.
        @param	text			The text color for menu items.
        @param	bg				The background color for menu items.
        @param	highlightedtext	The text color for the highlighted menu item.
        @param	highlightedbg	The background color the highlighted menu item.		*/
    void jpopupmenu_setcolors(t_jpopupmenu* menu,
                                         t_jrgba text,
                                         t_jrgba bg,
                                         t_jrgba highlightedtext,
                                         t_jrgba highlightedbg);

    // Internal use only (header functions are not exported)
    void jpopupmenu_setheadercolor(t_jpopupmenu* menu, char* hc);

    /**	Set the font used by a pop-up menu.
        @ingroup				jpopupmenu
        @param	menu			The pop-up menu whose font will be set.
        @param	font			A pointer to a font object, whose font info will be copied to the pop-up menu.	*/
    void jpopupmenu_setfont(t_jpopupmenu* menu, t_jfont* font);

    /**	Add an item to a pop-up menu.
        @ingroup			jpopupmenu
        @param	menu		The pop-up menu to which the item will be added.
        @param	itemid		Each menu item should be assigned a unique integer id using this parameter.
        @param	utf8Text	The text to display in for the menu item.
        @param	textColor	The color to use for the menu item, or NULL to use the default color.
        @param	checked		A non-zero value indicates that the item should have a check-mark next to it.
        @param	disabled	A non-zero value indicates that the item should be disabled.
        @param	icon		A #t_jsurface will be used as an icon for the menu item if provided here.
                            Pass NULL for no icon.		*/
    void jpopupmenu_additem(t_jpopupmenu* menu,
                                       int itemid,
                                       const char* utf8Text,
                                       char* textColor,
                                       int checked,
                                       int disabled,
                                       t_jsurface* icon);

    /**	Add a pop-menu to another pop-menu as a submenu.
        @ingroup			jpopupmenu
        @param	menu		The pop-up menu to which a menu will be added as a submenu.
        @param	utf8Name	The name of the menu item.
        @param	submenu		The pop-up menu which will be used as the submenu.
        @param	disabled	Pass a non-zero value to disable the menu item. */
    void jpopupmenu_addsubmenu(t_jpopupmenu* menu,
                                          const char* utf8Name,
                                          t_jpopupmenu* submenu,
                                          int disabled);

    /**	Add a separator to a pop-menu.
        @ingroup			jpopupmenu
        @param	menu		The pop-up menu to which the separator will be added.	*/
    void jpopupmenu_addseperator(t_jpopupmenu* menu);

    // Internal use only (header functions are not exported)
    void jpopupmenu_addheader(t_jpopupmenu* menu, const char* utf8Text);

    // Internal use only
    // ownerdraw: give a t_object to the menu.
    // it will be sent a paint message to draw itself.
    // it will be sent a getsize message to find out the size.
    void jpopupmenu_addownerdrawitem(t_jpopupmenu* menu,
                                                int itemid,
                                                t_object* owner);

    /**	Tell a menu to display at a specified location.
        @ingroup			jpopupmenu
        @param	menu		The pop-up menu to display.
        @param	screen		The point at which to display in screen coordinates.
        @param	defitemid	The initially choosen item id.
        @return				The item id for the item in the menu choosen by the user.	*/
    int jpopupmenu_popup(t_jpopupmenu* menu,
                                     t_pt screen,
                                     int defitemid);		// initial item id


    /**	Tell a menu to display near a given box in a patcher.
        @ingroup			jpopupmenu
        @param	menu		The pop-up menu to display.
        @param	box			The box above which to display the menu.
        @param	view		The patcherview for the box in which to display the menu.
        @param	defitemid	The initially choosen item id.
        @return				The item id for the item in the menu choosen by the user.	*/
    int jpopupmenu_popup_nearbox(t_jpopupmenu* menu,
                                             t_object* box, t_object* view,
                                             int defitemid);


    /**	Tell a menu to display below a given rectangle in a patcher.
        @ingroup			jpopupmenu
        @param	menu		The pop-up menu to display.
        @param	rect		The rectangle below which to display the menu.
        @param	defitemid	The initially choosen item id.
        @return				The item id for the item in the menu choosen by the user.	*/
    int jpopupmenu_popup_belowrect(t_jpopupmenu* menu, t_rect rect, int defitemid);

    /**	Tell a menu to display above a given rectangle in a patcher.
        @ingroup			jpopupmenu
        @param	menu		The pop-up menu to display.
        @param	rect		The rectangle above which to display the menu.
        @param	defitemid	The initially choosen item id.
        @return				The item id for the item in the menu choosen by the user.	*/
    int jpopupmenu_popup_aboverect(t_jpopupmenu* menu, t_rect rect, int defitemid);


    /**	Get the slant box's font.
        @ingroup	jfont
        @param	b	An object's box.
        @return		A value from the #t_jgraphics_font_weight enum.	*/
    long jbox_get_font_weight(t_object* b);

    /**	Get the slant box's font.
        @ingroup	jfont
        @param	b	An object's box.
        @return		A value from the #t_jgraphics_font_slant enum.	*/
    long jbox_get_font_slant(t_object* b);


    /**
        Create a color (#t_jrgba) attribute and add it to a Max class.

        @ingroup	attr
        @param	c				The class pointer.
        @param	attrname		The name of this attribute as a C-string.
        @param	flags			Any flags you wish to declare for this attribute, as defined in #e_max_attrflags.
        @param	structname		The C identifier for the struct (containing a valid #t_object header) representing an instance of this class.
        @param	structmember	The C identifier of the member in the struct that holds the value of this attribute.
    */
    #define CLASS_ATTR_RGBA(c,attrname,flags,structname,structmember) \
        {	CLASS_ATTR_DOUBLE_ARRAY(c,attrname,flags,structname,structmember,4); \
            CLASS_ATTR_ACCESSORS(c,attrname,NULL,jgraphics_attr_setrgba); \
            CLASS_ATTR_PAINT(c,attrname,0); }


    /**
        Retrieves the value of a color attribute, given its parent object and name.

        @ingroup attr
        @param 	ob		The attribute's parent object
        @param 	s		The attribute's name
        @param	c		The address of a #t_jrgba struct that will be filled with the attribute's color component values.
        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_getjrgba(void* ob, t_symbol* s, t_jrgba* c);


    /**
        Sets the value of a color attribute, given its parent object and name.
        The function will call the attribute's <tt>set</tt> method, using the data provided.

        @ingroup attr
        @param 	ob		The attribute's parent object
        @param 	s		The attribute's name
        @param	c		The address of a #t_jrgba struct that contains the new color.
        @return 		This function returns the error code #MAX_ERR_NONE if successful,
                         or one of the other error codes defined in #e_max_errorcodes if unsuccessful.
    */
    t_max_err object_attr_setjrgba(void* ob, t_symbol* s, t_jrgba* c);


    /**	Get the components of a color in an array of pre-allocated atoms.

        @ingroup color
        @param	argv	The address to the first of an array of atoms that will hold the result.
                        At least 4 atoms must be allocated, as 4 atoms will be set by this function
                        for the red, green, blue, and alpha components.
        @param	c		The address of a #t_jrgba struct from which the color components will be fetched.
    */
    void jrgba_to_atoms(t_jrgba* c, t_atom* argv);


    /**	Set the components of a color by providing an array of atoms.
        If it is an array of 3 atoms, then the atoms provided should define the
        red, green, and blue components (in this order) in a range of [0.0, 1.0].
        If a 4th atom is provided, it will define the alpha channel.
        If the alpha channel is not defined then it is assumed to be 1.0.

        @ingroup color
        @param	argc	The number of atoms in the array provided in argv.
                        This should be 3 or 4 depending on whether or not the alpha channel is being provided.
        @param	argv	The address to the first of an array of atoms that define the color.
        @param	c		The address of a #t_jrgba struct for which the color will be defined.
        @return			A Max error code.		*/
    t_max_err atoms_to_jrgba(long argc, t_atom* argv, t_jrgba* c);


    /**	Set the components of a color.
        @ingroup color
        @param	prgba	The address of a #t_jrgba struct for which the color will be defined.
        @param	r		The value of the red component in a range of [0.0, 1.0].
        @param	g		The value of the green component in a range of [0.0, 1.0].
        @param	b		The value of the blue component in a range of [0.0, 1.0].
        @param	a		The value of the alpha component in a range of [0.0, 1.0].	*/
    void jrgba_set(t_jrgba* prgba, double r, double g, double b, double a);


    /**	Copy a color.
        @ingroup color
        @param	dest	The address of a #t_jrgba struct to which the color will be copied.
        @param	src		The address of a #t_jrgba struct from which the color will be copied.	*/
    void jrgba_copy(t_jrgba* dest, t_jrgba* src);


    /**	Compare two colors for equality.
        @ingroup color
        @param	rgba1	The address of a #t_jrgba struct to compare.
        @param	rgba2	The address of another #t_jrgba struct to compare.
        @return 		returns 1 if rgba1 == rgba2.	*/
    long jrgba_compare(t_jrgba* rgba1, t_jrgba* rgba2);


    /**	Get a list of of filetypes appropriate for use with jgraphics surfaces.
        @ingroup			jgraphics
        @param	dummy		Unused.
        @param	count		The address of a variable to be set with the number of types in filetypes upon return.
        @param	filetypes	The address of a variable that will represent the array of file types upon return.
        @param	alloc		The address of a char that will be flagged with a 1 or a 0 depending on whether or not
                            memory was allocated for the filetypes member.

        @remark This example shows a common usage of jgraphics_getfiletypes().
        @code
        char       filename[MAX_PATH_CHARS];
        t_fourcc   *type = NULL;
        long       ntype;
        long       outtype;
        t_max_err  err;
        char       alloc;
        short      path;
        t_jsurface* surface;

        if (want_to_show_dialog) {
            jgraphics_getfiletypes(x, &ntype, &type, &alloc);
            err = open_dialog(filename, &path,(void* )&outtype, (void* )type, ntype);
            if (err)
                goto out;
        }
        else {
            strncpy_zero(filename, s->s_name, MAX_PATH_CHARS);
            err = locatefile_extended(filename, &path, &outtype, type, ntype);
            if (err)
                goto out;
        }
        surface = jgraphics_image_surface_create_referenced(filename, path);
    out:
        if (alloc)
            sysmem_freeptr((char* )type);
        @endcode
    */
    void jgraphics_getfiletypes(void* dummy, long* count, t_symbol** filetypes, char* alloc);


    // boxlayer stuff

    /**	Invalidate a layer, indicating that it needs to be re-drawn.
        @ingroup		boxlayer
        @param	b		The object/box to invalidate.
        @param	view	The patcherview for the object which should be invalidated, or NULL for all patcherviews.
        @param	name	The name of the layer to invalidate.
        @return			A Max error code.	*/
    t_max_err jbox_invalidate_layer(t_object* b, t_object* view, t_symbol* name);

    /**	Create a layer, and ready it for drawing commands.
        The layer drawing commands must be wrapped with a matching call to jbox_end_layer()
        prior to calling jbox_paint_layer().

        @ingroup		boxlayer
        @param	b		The object/box to which the layer is attached.
        @param	view	The patcherview for the object to which the layer is attached.
        @param	name	A name for this layer.
        @param	width	The width of the layer.
        @param	height	The height of the layer.
        @return			A #t_jgraphics context for drawing into the layer.	*/
    t_jgraphics* jbox_start_layer(t_object* b, t_object* view, t_symbol* name, double width, double height);

    /**	Conclude a layer, indicating that it is complete and ready for painting.
        @ingroup		boxlayer
        @param	b		The object/box for the layer opened by jbox_start_layer().
        @param	view	The patcherview for the object opened by jbox_start_layer().
        @param	name	The name of the layer.
        @return			A Max error code.	*/
    t_max_err jbox_end_layer(t_object* b, t_object* view, t_symbol* name);

    /**	Paint a layer at a given position.
        Note that the current color alpha value is used when painting layers to allow you to blend layers.
        The same is also true for jgraphics_image_surface_draw() and jgraphics_image_surface_draw_fast().

        @ingroup		boxlayer
        @param	b		The object/box to be painted.
        @param	view	The patcherview for the object which should be painted, or NULL for all patcherviews.
        @param	name	The name of the layer to paint.
        @param	x		The x-coordinate for the position at which to paint the layer.
        @param	y		The y-coordinate for the position at which to paint the layer.
        @return			A Max error code.	*/
    t_max_err jbox_paint_layer(t_object* b, t_object* view, t_symbol* name, double x, double y);


    /** Simple utility to test for rectangle intersection.
        @ingroup		jgraphics
        @param	r1		The address of the first rect for the test.
        @param	r2		The address of the second rect for the test.
        @return			Returns true if the rects intersect, otherwise false.	*/
    long jgraphics_rectintersectsrect(t_rect* r1, t_rect* r2);

    /** Simple utility to test for rectangle containment.
        @ingroup		jgraphics
        @param	outer	The address of the first rect for the test.
        @param	inner	The address of the second rect for the test.
        @return			Returns true if the inner rect is completely inside the outer rect,
                         otherwise false.	*/
    long jgraphics_rectcontainsrect(t_rect* outer, t_rect* inner);

    /** Generate a #t_rect according to positioning rules.
        @ingroup							jgraphics
        @param	positioned_rect				The address of a valid #t_rect whose members will be filled in upon return.
        @param	positioned_near_this_rect	A pointer to a rect near which this rect should be positioned.
        @param	keep_inside_this_rect		A pointer to a rect defining the limits within which the new rect must reside.	*/
    void jgraphics_position_one_rect_near_another_rect_but_keep_inside_a_third_rect(
        t_rect* positioned_rect,
        const t_rect* positioned_near_this_rect,
        const t_rect* keep_inside_this_rect);


    /** Clip to a subset of the graphics context; once done, cannot be undone, only further reduced.
        @ingroup		jgraphics

        @param		g		The #t_jgraphics context to be clipped.
        @param		x		x origin of clip region.
        @param		y		y origin of clip region.
        @param		width	width of clip region.
        @param		height	height of clip region.
        */
    void jgraphics_clip(t_jgraphics* g, double x, double y, double width, double height);


    END_USING_C_LINKAGE

}} // namespace c74::max
