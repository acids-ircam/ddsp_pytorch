/// @file
///	@ingroup 	minapi
///	@copyright	Copyright (c) 2016, Cycling '74
///	@license	Usage of this file and its contents is governed by the MIT License

#pragma once

namespace c74::min::ui {

    class target {
    public:
        explicit target(const atoms& args) {
            // assert(args.size() > 1);
            if (args.size() < 2) {
                // this can happen when the args are a single atom with an event inside of it.
                ;
            }
            else if (args.size() == 3) {
                // if there are 3 args then the first arg is the graphics context itself
                // this occurs in the case where we create an image (surface)
                m_graphics_context = reinterpret_cast<max::t_jgraphics*>(static_cast<void*>(args[0]));
                m_rect.width = args[1];
                m_rect.height = args[2];
            }
            else {
                // this is the typical case, where we need to get the context from the object's box
                *this = target(args[0], args[1]);
            }
        }

        target(max::t_object* o, max::t_object* a_patcherview) {
            m_box = (max::t_jbox*)o;
            m_view = a_patcherview;
            m_graphics_context = (max::t_jgraphics*)max::patcherview_get_jgraphics(m_view);
            jbox_get_rect_for_view((max::t_object*)m_box, m_view, &m_rect);
        }



        operator max::t_jgraphics*() const {
            return m_graphics_context;
        }

        max::t_object* view() {
            return m_view;
        }

        number x() const {
            return m_rect.x;
        }

        number y() const {
            return m_rect.y;
        }

        number width() const {
            return m_rect.width;
        }

        number height() const {
            return m_rect.height;
        }

    private:
        max::t_jbox*		m_box;
        max::t_object*		m_view;
        max::t_jgraphics*	m_graphics_context;
        max::t_rect			m_rect {};
    };


    // The classes below appear to have a lot of code duplication.
    // This is true. As an example: position, origin, and destination are essentially identical.
    // We cannot, however, declare one class and then make type aliases to it
    // because doing so results in failures to resolve ambiguity in the variadic template constructor(s) below.


    class position {
    public:
        position(const double x, const double y)
        : m_position {x,y}
        {}

        position(const int x, const int y)
        : m_position { static_cast<double>(x), static_cast<double>(y) }
        {}

        position(const c74::max::t_pt& point)
        : m_position { point.x, point.y }
        {}

        void operator()(max::t_rect& r) const {
            r.x = m_position.x;
            r.y = m_position.y;
        }

    private:
        max::t_pt m_position;
    };

    
    class origin {
    public:
        origin(const double x, const double y)
        : m_position {x,y}
        {}

        origin(const int x, const int y)
        : m_position { static_cast<double>(x), static_cast<double>(y) }
        {}

        origin(const c74::max::t_pt& point)
        : m_position { point.x, point.y }
        {}

        void operator()(max::t_rect& r) const {
            r.x = m_position.x;
            r.y = m_position.y;
        }

    private:
        max::t_pt m_position;
    };


    class destination {
    public:
        destination(const double x, const double y)
        : m_position { x, y }
        {}

        destination(const int x, const int y)
        : m_position { static_cast<double>(x), static_cast<double>(y) }
        {}

        destination(const int x, const double y)
        : m_position { static_cast<double>(x), y }
        {}

        destination(const c74::max::t_pt& point)
        : m_position { point.x, point.y }
        {}

        void operator()(max::t_rect& r) const {
            r.x = m_position.x;
            r.y = m_position.y;
        }

    private:
        max::t_pt m_position;
    };
    

    class size {
    public:
        size(const double width, const double height)
        : m_size { width, height }
        {}

        size(const double a_size)
        : size { a_size, a_size }
        {}

        void operator()(max::t_rect& r) const {
            r.width = m_size.width;
            r.height = m_size.height;
        }

    private:
        max::t_size m_size;
    };


    class corner {
    public:
        corner(const double a_width, const double a_height)
        : m_size { a_width, a_height }
        {}

        corner(const double a_radius)
        : m_size { a_radius, a_radius }
        {}

        void operator()(max::t_rect& r) const {
            r.width = m_size.width;
            r.height = m_size.height;
        }

        private:
            max::t_size m_size;
    };


    class span {
    public:
        span(const double start, const double finish)
        : angle1 { start }
        , angle2 { finish }
        {}

        void operator()(max::t_rect& r) const {
            r.x = angle1;
            r.y = angle2;
        }

    private:
        double angle1 {};
        double angle2 {};
    };


    class line_width {
    public:
        line_width(const number a_width)
        : m_width { a_width }
        {}

        void operator()(const target& g) const {
            max::jgraphics_set_line_width(g, m_width);
        }

    private:
        number m_width;
    };


    /// textual content
    class content {
    public:
        content(const string& str)
        : m_text { str }
        {}

        void operator()(string& s) const {
            s = m_text;
        }

    private:
        string m_text;
    };


    class fontface {
    public:
        fontface(const symbol a_name, const bool bold = false, const bool italic = false)
        : m_name	{ a_name }
		, m_weight {bold ? max::t_jgraphics_font_weight::JGRAPHICS_FONT_WEIGHT_BOLD
						 : max::t_jgraphics_font_weight::JGRAPHICS_FONT_WEIGHT_NORMAL}
		, m_slant {italic ? max::t_jgraphics_font_slant::JGRAPHICS_FONT_SLANT_ITALIC
						  : max::t_jgraphics_font_slant::JGRAPHICS_FONT_SLANT_NORMAL}
        {}

        void operator()(target& g) const {
            max::jgraphics_select_font_face(g, m_name, m_slant, m_weight);
        }

    private:
        symbol							m_name;
        max::t_jgraphics_font_weight	m_weight;
        max::t_jgraphics_font_slant		m_slant;
    };


    // NOTE: this is the model to follow for the other draw commands

    class fontsize {
    public:
        fontsize(const double a_value)
        : m_value { a_value }
        {}

        void operator()(target& g) const {
            max::jgraphics_set_font_size(g, m_value);
        }

    private:
        number m_value;
    };


    /// Apply a rotate matrix transformation to the object

    class rotation {
    public:
        rotation(const number a_value)
//		: m_value { a_value }
        {}

        void apply() {

        }

        void cleanup() {

        }

    private:
//		number m_value;
    };


    class element {
    protected:

        /// constructor utility: target (graphics context)
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, target>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            m_target = std::make_unique<target>(arg);
        }

        /// constructor utility: color
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, color>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            m_color = arg;
        }

        /// constructor utility: position
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, position>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg(m_rect);
        }

        /// constructor utility: size
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, size>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg(m_rect);
        }

        /// constructor utility: origin
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, origin>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg(m_rect);
        }

        /// constructor utility: destination
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, destination>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg(m_misc);
        }

        /// constructor utility: corner
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, corner>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg(m_misc);
        }

        /// constructor utility: span
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, span>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg(m_misc);
        }

        /// constructor utility: fontface
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, fontface>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg( const_cast<target&>(*m_target) );
        }

        /// constructor utility: fontsize
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, fontsize>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg( const_cast<target&>(*m_target) );
        }

        /// constructor utility: line_width
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, line_width>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg( const_cast<target&>(*m_target) );
        }

        /// constructor utility: content
        template<typename argument_type>
        constexpr typename enable_if<is_same<argument_type, content>::value>::type
        assign_from_argument(const argument_type& arg) noexcept {
            arg(m_text);
        }

        /// constructor utility
        constexpr void handle_arguments() noexcept {
            ;
        }

        /// constructor utility
        template <typename FIRST_ARG, typename ...REMAINING_ARGS>
        constexpr void handle_arguments(FIRST_ARG const& first, REMAINING_ARGS const& ...args) noexcept {
            assign_from_argument(first);
            if (sizeof...(args) > 0)
                handle_arguments(args...); // recurse
        }


        void update() {
            if (m_rect.x <= 0.0)
                m_rect.x = 0 + m_rect.x;
            if (m_rect.y <= 0.0)
                m_rect.y = 0 + m_rect.y;
            if (m_rect.width <= 0.0)
                m_rect.width = m_target->width() + m_rect.width;
            if (m_rect.height <= 0.0)
                m_rect.height = m_target->height() + m_rect.height;

            max::jgraphics_set_source_jrgba(*m_target, m_color);
        }


        std::unique_ptr<target>		m_target;
        max::t_rect					m_rect {};
        max::t_rect					m_misc {};
        color						m_color;
        string						m_text;
    };


    enum draw_style {
        stroke,
        fill
    };


    template<draw_style style>
    inline void draw(target& a_target);

    template<>
    inline void draw<stroke>(target& a_target) {
        max::jgraphics_stroke(a_target);
    }

    template<>
    inline void draw<fill>(target& a_target) {
        max::jgraphics_fill(a_target);
    }


    template<draw_style style = stroke>
    class rect : public element {
    public:
        template<typename ...ARGS>
        rect(ARGS... args) {
            handle_arguments(args...);
            update();
            max::jgraphics_rectangle_rounded(*m_target, m_rect.x, m_rect.y, m_rect.width, m_rect.height, m_misc.width, m_misc.height);
            draw<style>(*m_target);
        }
    };


    template<draw_style style = stroke>
    class ellipse : public element {
    public:
        template<typename ...ARGS>
        ellipse(ARGS... args) {
            handle_arguments(args...);
            update();
            max::jgraphics_ellipse(*m_target, m_rect.x, m_rect.y, m_rect.width, m_rect.height);
            draw<style>(*m_target);
        }
    };


    template<draw_style style = stroke>
    class line : public element {
    public:
        template<typename ...ARGS>
        line(ARGS... args) {
            handle_arguments(args...);
            update();
            max::jgraphics_move_to(*m_target, m_rect.x, m_rect.y);
            max::jgraphics_line_to(*m_target, m_misc.x, m_misc.y);
            draw<style>(*m_target);
        }
    };


    template<draw_style style = stroke>
    class arc : public element {
    public:
        template<typename ...ARGS>
        arc(ARGS... args) {
            handle_arguments(args...);
            update();
            // reinterpreting the "rect" coordinates here to be the center point and the width/height (identical) to be the radius
            max::jgraphics_arc(*m_target, m_rect.x, m_rect.y, m_rect.height, m_misc.x, m_misc.y);
            draw<style>(*m_target);
        }
    };


    class text : public element {
    public:
        template<typename ...ARGS>
        text(ARGS... args) {
            handle_arguments(args...);
            update();
            max::jgraphics_move_to(*m_target, m_rect.x, m_rect.y);
            max::jgraphics_show_text(*m_target, m_text.c_str());
        }
    };


    class image {
    public:
        image(object_base* an_owner, const double width, const double height, const function& a_function = nullptr)
        : m_width { width }
        , m_height { height }
        , m_draw_callback { a_function }
        {}

        ~image() {
            if (m_surface) {
                c74::max::jgraphics_surface_destroy(m_surface);
                m_surface = nullptr;
            }
        }

        void redraw(const int width, const int height) {
            auto old_surface = m_surface;
            m_surface = c74::max::jgraphics_image_surface_create(c74::max::JGRAPHICS_FORMAT_ARGB32, width, height);
            m_width = width;
            m_height = height;
            c74::max::t_jgraphics *ctx = jgraphics_create(m_surface);
            atoms a {{ ctx, m_width, m_height }};
            m_draw_callback(a,0);
            c74::max::jgraphics_destroy(ctx);

            if (old_surface)
                c74::max::jgraphics_surface_destroy(old_surface);
        }

        void draw(ui::target& t, const double x, const double y, const double width, const double height) {
            c74::max::jgraphics_image_surface_draw(t, m_surface, {0.0, 0.0, m_width, m_height}, {x, y, width, height});
        }

    private:
        double					m_width;
        double					m_height;
        function        	    m_draw_callback;
        c74::max::t_jsurface*   m_surface { nullptr };
    };


} // namespace c74::min:::graphics
