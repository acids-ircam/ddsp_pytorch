/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_jitter.h"

namespace c74::min {

    using pixel = std::array<uchar, 4>;

    template<class matrix_type, size_t plane_count>
    using cell = std::array<matrix_type, plane_count>;

    enum {
        alpha = 0,
        red,
        green,
        blue
    };


    class matrix_coord {
    public:
        matrix_coord(const long x, const long y) {
            position[0] = x;
            position[1] = y;
        }

        long x() const {
            return position[0];
        }

        long y() const {
            return position[1];
        }

        long position[max::JIT_MATRIX_MAX_DIMCOUNT]{};
    };


    class matrix_info {
    public:
        matrix_info(const max::t_jit_matrix_info* a_in_info, uchar* ip, max::t_jit_matrix_info* a_out_info, uchar* op)
        : m_in_info { a_in_info }
        , m_bip { ip }
        , m_out_info { a_out_info }
        , m_bop { op }
        {}


        long plane_count() const {
            return m_in_info->planecount;
        }

        long dim_count() const {
            return m_in_info->dimcount;
        }

        long width() const {
            return m_in_info->dim[0];
        }

        long height() const {
            return m_in_info->dim[1];
        }


        template<class matrix_type, size_t plane_count>
        const std::array<matrix_type, plane_count> in_cell(const matrix_coord& coord) const {
            auto p = m_bip;

            for (auto j = 0; j < m_in_info->dimcount; ++j)
                p += coord.position[j] * m_in_info->dimstride[j];

            std::array<matrix_type, plane_count> pa;

            auto p2 = reinterpret_cast<matrix_type*>(p);
            for (auto plane = 0; plane < plane_count; ++plane)
                pa[plane] = *(p2 + plane);
            return pa;
        }

        template<class matrix_type, size_t plane_count>
        const std::array<matrix_type, plane_count> in_cell(const int x, const int y) const {
            matrix_coord coord(x, y);
            return in_cell<matrix_type, plane_count>(coord);
        }


        const pixel in_pixel(const matrix_coord& coord) const {
            auto p = m_bip;

            for (auto j = 0; j < m_in_info->dimcount; ++j)
                p += coord.position[j] * m_in_info->dimstride[j];

            const pixel pa = {{*(p), *(p + 1), *(p + 2), *(p + 3)}};
            return pa;
        }


        const pixel in_pixel(const int x, const int y) const {
            matrix_coord coord(x, y);
            return in_pixel(coord);
        }


        pixel out_pixel(const matrix_coord&) {
            // TODO: implement
			pixel a {};
            return a;
        }


        const max::t_jit_matrix_info* m_in_info;
        uchar*                  m_bip;
        max::t_jit_matrix_info* m_out_info;
        uchar*                  m_bop;
    };


    /// The base class for all template specializations of matrix_operator.

    class matrix_operator_base {

    public:
        /// When the matrix is processed a call is made to the subclass calc_cell() method for each cell.
        /// The order in which the cells are iterated will be one of the options provided here.

        enum class  iteration_direction { forward, reverse, bidirectional, enum_count };
        enum_map    iteration_direction_info {"forward", "reverse", "bidirectional"};
    };


    /// Inheriting from matrix_operator extends your class functionality to processing matrices.

    template<placeholder matrix_operator_placeholder_type = placeholder::none>
    class matrix_operator : public matrix_operator_base {
    public:
        /// @param	enable_parallel_breakup Allow the matrix processing engine to break apart the processing into smaller submatrices.
        ///									This can improve the speed calculating your matrix processing, but in some cases may have
        ///									undesired consequences.

        explicit matrix_operator(const bool enable_parallel_breakup = true)
        : m_enable_parallel_breakup { enable_parallel_breakup }
        {}

        template<class matrix_type, size_t planecount>
        friend cell<matrix_type, planecount> calc_cell(cell<matrix_type, planecount> input, const matrix_info& info, matrix_coord& position);


        /// Find out if parallel processing of the matrix is enabled
        ///	@return	True if parallel breakup is enabled. Otherwise false.

        bool parallel_breakup_enabled() const {
            return m_enable_parallel_breakup;
        }


        void direction(const iteration_direction new_direction) {
            m_direction = new_direction;
        }


        iteration_direction direction() const {
            return m_direction;
        }

    private:
        bool                m_enable_parallel_breakup;
        iteration_direction m_direction {};
    };


    // this is for the jitter object (the normal one is used for the max wrapper of that)
    static max::t_class* this_jit_class = nullptr;


    // NOTE: For Jitter, minwrap is the nobox Jitter Object
    // we then generate another wrapper (max_jit_wrapper) around that...

    /// @param s	The name of the object is passed as an argument to support object-mappings.
    ///		In such cases we might not know what the object name is at compile time.
    template<class min_class_type>
    max::t_object* jit_new(const max::t_symbol* s) {
        auto self = static_cast<minwrap<min_class_type>*>(max::jit_object_alloc(this_jit_class));

        self->m_min_object.assign_instance(self->maxobj());
        min_ctor(self, {});

        // NOTE: when instantiated from JS s will be NULL
        if (s)
            self->m_min_object.set_classname(s);
        self->m_min_object.postinitialize();

        self->m_min_object.try_call("setup");

        return self->maxobj();
    }

    template<class min_class_type>
    void jit_free(minwrap<min_class_type>* self) {
        self->cleanup();
        self->m_min_object.~min_class_type();    // placement delete
    }


    struct max_jit_wrapper {
        max::t_object m_ob;
        void*         m_obex;
    };


    template<class min_class_type>
    void* max_jit_mop_new(const max::t_symbol* s, const long argc, const max::t_atom* argv) {
        assert(this_class_name != nullptr);    // required pre-condition

        atom_reference args(argc, argv);
        long           attrstart = attr_args_offset(static_cast<short>(args.size()), args.begin());
        auto           cppname   = this_class_name;
        auto           self      = static_cast<max_jit_wrapper*>(max::max_jit_object_alloc(this_class, cppname));
        auto           o         = max::jit_object_new(cppname, s);
        auto           job       = reinterpret_cast<minwrap<min_class_type>*>(o);

        if (job->m_min_object.has_call("mop_setup")) {
            atoms atomargs(args.begin(), args.begin() + attrstart);
            atomargs.push_back(atom{self});
            job->m_min_object.try_call("mop_setup", atomargs);
        }
        else {
            max_jit_mop_setup_simple(self, o, args.size(), args.begin());
        }

        max_jit_attr_args(self, static_cast<short>(args.size()), args.begin());
        job->m_min_object.try_call("maxob_setup", atoms(args.begin(), args.begin() + attrstart));

        return self;
    }


    template<class min_class_type>
    void max_jit_mop_free(max_jit_wrapper* self) {
        max::max_jit_mop_free(self);
        max::jit_object_free(max::max_jit_obex_jitob_get(self));
        max::max_jit_object_free(self);
    }


    // We are using a C++ template to process a vector of the matrix for any of the given types.
    // Thus, we don't need to duplicate the code for each datatype.

    template<class min_class_type, typename U, enable_if_matrix_operator<min_class_type> = 0>
    void jit_calculate_vector(
        minwrap<min_class_type>* self, const matrix_info& info, const long n, const long i, const max::t_jit_op_info* in, max::t_jit_op_info* out) {
        auto       ip         = in ? static_cast<U*>(in->p) : nullptr;
        auto       op         = static_cast<U*>(out->p);
        auto       is         = in ? in->stride : 0;
        auto       os         = out->stride;
        const auto step       = os / info.plane_count();
        const bool planematch = (info.m_in_info->planecount == info.m_out_info->planecount);
        auto       ip_last    = ip + (is * (n - 1));
        auto       op_last    = op + (os * (n - 1));

        if (planematch && info.plane_count() == 1) {
            // forward or bidirectional
            if (self->m_min_object.direction() != matrix_operator_base::iteration_direction::reverse) {
                for (auto j = 0; j < n; ++j) {
                    matrix_coord           position(j, i);
                    U                      val = ip ? *(ip) : 0;
                    const std::array<U, 1> tmp = {{val}};
                    const std::array<U, 1> out_cell = self->m_min_object.calc_cell(tmp, info, position);

                    *(op) = out_cell[0];
                    if (ip)
                        ip += is;
                    op += os;
                }
            }

            // reverse or bidirectional
            if (self->m_min_object.direction() != matrix_operator_base::iteration_direction::forward) {
                ip = ip_last;
                op = op_last;

                for (auto j = n - 1; j >= 0; --j) {
                    matrix_coord position(j, i);

                    if (self->m_min_object.direction() == matrix_operator_base::iteration_direction::bidirectional) {
                        const std::array<U, 1> tmp = {{*op}};
                        const std::array<U, 1> out_cell = self->m_min_object.calc_cell(tmp, info, position);
                        *op                        = out_cell[0];
                    }
                    else {
                        std::array<U, 1> tmp;
                        if (ip)
                            tmp = {{*ip}};
                        const std::array<U, 1> out_cell = self->m_min_object.calc_cell(tmp, info, position);
                        *op                        = out_cell[0];
                    }
                    if (ip)
                        ip -= is;
                    op -= os;
                }
            }
        }
        else if (planematch && info.plane_count() == 4) {
            if (self->m_min_object.direction() != matrix_operator_base::iteration_direction::reverse) {
                for (auto j = 0; j < n; ++j) {
                    matrix_coord           position(j, i);
                    U                      v1  = ip ? *(ip) : 0;
                    U                      v2  = ip ? *(ip + step) : 0;
                    U                      v3  = ip ? *(ip + step * 2) : 0;
                    U                      v4  = ip ? *(ip + step * 3) : 0;
                    const std::array<U, 4> tmp = {{v1, v2, v3, v4}};
                    const std::array<U, 4> out_cell = self->m_min_object.calc_cell(tmp, info, position);

                    *(op)            = out_cell[0];
                    *(op + step)     = out_cell[1];
                    *(op + step * 2) = out_cell[2];
                    *(op + step * 3) = out_cell[3];

                    if (ip)
                        ip += is;
                    op += os;
                }
            }

            // reverse or bidirectional
            if (self->m_min_object.direction() != matrix_operator_base::iteration_direction::forward) {
                ip = ip_last;
                op = op_last;

                for (auto j = n - 1; j >= 0; --j) {
                    matrix_coord position(j, i);

                    if (self->m_min_object.direction() == matrix_operator_base::iteration_direction::bidirectional) {
                        U                      v1  = ip ? *(op) : 0;
                        U                      v2  = ip ? *(op + step) : 0;
                        U                      v3  = ip ? *(op + step * 2) : 0;
                        U                      v4  = ip ? *(op + step * 3) : 0;
                        const std::array<U, 4> tmp = {{v1, v2, v3, v4}};
                        const std::array<U, 4> out_cell = self->m_min_object.calc_cell(tmp, info, position);
                        *(op)                      = out_cell[0];
                        *(op + step)               = out_cell[1];
                        *(op + step * 2)           = out_cell[2];
                        *(op + step * 3)           = out_cell[3];
                    }
                    else {
                        U                      v1  = ip ? *(ip) : 0;
                        U                      v2  = ip ? *(ip + step) : 0;
                        U                      v3  = ip ? *(ip + step * 2) : 0;
                        U                      v4  = ip ? *(ip + step * 3) : 0;
                        const std::array<U, 4> tmp = {{v1, v2, v3, v4}};
                        const std::array<U, 4> out_cell = self->m_min_object.calc_cell(tmp, info, position);
                        *(op)                      = out_cell[0];
                        *(op + step)               = out_cell[1];
                        *(op + step * 2)           = out_cell[2];
                        *(op + step * 3)           = out_cell[3];
                    }
                    if (ip)
                        ip -= is;
                    op -= os;
                }
            }
        }
        else {
            const auto instep  = is / info.m_in_info->planecount;
            const auto outstep = os / info.m_out_info->planecount;

            // forward or bidirectional
            if (self->m_min_object.direction() != matrix_operator_base::iteration_direction::reverse) {
                for (auto j = 0; j < n; ++j) {
                    matrix_coord                                  position(j, i);
                    std::array<U, max::JIT_MATRIX_MAX_PLANECOUNT> tmp;

                    if (ip) {
                        for (auto k = 0; k < info.m_in_info->planecount; ++k)
                            tmp[k] = *(ip + instep * k);
                    }

                    const std::array<U, max::JIT_MATRIX_MAX_PLANECOUNT> out_cell = self->m_min_object.calc_cell(tmp, info, position);

                    for (auto k = 0; k < info.m_out_info->planecount; ++k)
                        *(op + outstep * k) = out_cell[k];

                    if (ip)
                        ip += is;
                    op += os;
                }
            }

            // reverse or bidirectional
            if (self->m_min_object.direction() != matrix_operator_base::iteration_direction::forward) {
                ip = ip_last;
                op = op_last;

                for (auto j = n - 1; j >= 0; --j) {
                    matrix_coord                                  position(j, i);
                    std::array<U, max::JIT_MATRIX_MAX_PLANECOUNT> tmp;

                    if (ip) {
                        for (auto k = 0; k < info.m_in_info->planecount; ++k)
                            tmp[k] = *(ip + instep * k);
                    }

                    const std::array<U, max::JIT_MATRIX_MAX_PLANECOUNT> out_cell = self->m_min_object.calc_cell(tmp, info, position);

                    for (auto k = 0; k < info.m_out_info->planecount; ++k)
                        *(op + outstep * k) = out_cell[k];

                    if (ip)
                        ip -= is;
                    op -= os;
                }
            }
        }
    }


    // We also use a C+ template for the loop that wraps the call to jit_simple_vector(),
    // further reducing code duplication in jit_simple_calculate_ndim().
    // The calls into these templates should be inlined by the compiler, eliminating concern about any added function call overhead.

    template<class min_class_type, typename U>
    typename enable_if<is_base_of<matrix_operator_base, min_class_type>::value>::type
    jit_calculate_ndim_loop(minwrap<min_class_type>* self, const long n, max::t_jit_op_info* in_opinfo, max::t_jit_op_info* out_opinfo, max::t_jit_matrix_info* in_minfo, max::t_jit_matrix_info* out_minfo, uchar* bip, uchar* bop, long* dim, const long plane_count, const long datasize) {
        matrix_info info((in_minfo ? in_minfo : out_minfo), (bip ? bip : bop), out_minfo, bop);
        for (auto i = 0; i < dim[1]; i++) {
            if (in_opinfo)
                in_opinfo->p = bip + i * in_minfo->dimstride[1];
            out_opinfo->p = bop + i * out_minfo->dimstride[1];
            jit_calculate_vector<min_class_type, U>(self, info, n, i, in_opinfo, out_opinfo);
        }
    }


    template<class min_class_type, enable_if_matrix_operator<min_class_type> = 0>
    void jit_calculate_ndim(minwrap<min_class_type>* self, const long dim_count, long* dim, const long plane_count, max::t_jit_matrix_info* in_minfo, uchar* bip, max::t_jit_matrix_info* out_minfo, uchar* bop) {
        if (dim_count < 1)
            return;    // safety

        max::t_jit_op_info in_opinfo;
        max::t_jit_op_info out_opinfo;

        switch (dim_count) {
            case 1:
                dim[1] = 1;
                // (fall-through to next case is intentional)
            case 2: {
                // if plane_count is the same then flatten planes - treat as single plane data for speed
                auto n            = dim[0];
                in_opinfo.stride  = in_minfo->dim[0] > 1 ? in_minfo->planecount : 0;
                out_opinfo.stride = out_minfo->dim[0] > 1 ? out_minfo->planecount : 0;

                if (in_minfo->type == max::_jit_sym_char)
                    jit_calculate_ndim_loop<min_class_type, uchar>(
                        self, n, &in_opinfo, &out_opinfo, in_minfo, out_minfo, bip, bop, dim, plane_count, 1);
                else if (in_minfo->type == max::_jit_sym_long)
                    jit_calculate_ndim_loop<min_class_type, int>(
                        self, n, &in_opinfo, &out_opinfo, in_minfo, out_minfo, bip, bop, dim, plane_count, 4);
                else if (in_minfo->type == max::_jit_sym_float32)
                    jit_calculate_ndim_loop<min_class_type, float>(
                        self, n, &in_opinfo, &out_opinfo, in_minfo, out_minfo, bip, bop, dim, plane_count, 4);
                else if (in_minfo->type == max::_jit_sym_float64)
                    jit_calculate_ndim_loop<min_class_type, double>(
                        self, n, &in_opinfo, &out_opinfo, in_minfo, out_minfo, bip, bop, dim, plane_count, 8);
            } break;
            default:
                for (auto i = 0; i < dim[dim_count - 1]; i++) {
                    auto ip = bip + i * in_minfo->dimstride[dim_count - 1];
                    auto op = bop + i * out_minfo->dimstride[dim_count - 1];
                    jit_calculate_ndim(self, dim_count - 1, dim, plane_count, in_minfo, ip, out_minfo, op);
                }
        }
    }


    template<class min_class_type, enable_if_matrix_operator<min_class_type> = 0>
    void jit_calculate_ndim_single(
        minwrap<min_class_type>* self, const long dim_count, long* dim, const long plane_count, max::t_jit_matrix_info* out_minfo, uchar* bop) {
        if (dim_count < 1)
            return;    // safety

        max::t_jit_op_info out_opinfo;

        switch (dim_count) {
            case 1:
                dim[1] = 1;
                // (fall-through to next case is intentional)
            case 2: {
                // if plane_count is the same then flatten planes - treat as single plane data for speed
                auto n            = dim[0];
                out_opinfo.stride = out_minfo->dim[0] > 1 ? out_minfo->planecount : 0;

                if (out_minfo->type == max::_jit_sym_char)
                    jit_calculate_ndim_loop<min_class_type, uchar>(
                        self, n, NULL, &out_opinfo, NULL, out_minfo, NULL, bop, dim, plane_count, 1);
                else if (out_minfo->type == max::_jit_sym_long)
                    jit_calculate_ndim_loop<min_class_type, int>(
                        self, n, NULL, &out_opinfo, NULL, out_minfo, NULL, bop, dim, plane_count, 1);
                else if (out_minfo->type == max::_jit_sym_float32)
                    jit_calculate_ndim_loop<min_class_type, float>(
                        self, n, NULL, &out_opinfo, NULL, out_minfo, NULL, bop, dim, plane_count, 1);
                else if (out_minfo->type == max::_jit_sym_float64)
                    jit_calculate_ndim_loop<min_class_type, double>(
                        self, n, NULL, &out_opinfo, NULL, out_minfo, NULL, bop, dim, plane_count, 1);
            } break;
            default:
                for (auto i = 0; i < dim[dim_count - 1]; i++) {
                    auto op = bop + i * out_minfo->dimstride[dim_count - 1];
                    jit_calculate_ndim_single(self, dim_count - 1, dim, plane_count, out_minfo, op);
                }
        }
    }


    template<class min_class_type, enable_if_matrix_operator<min_class_type> = 0>
    void jit_matrix_docalc(minwrap<min_class_type>* self, max::t_object* inputs, max::t_object* outputs) {
        max::t_jit_err err        = max::JIT_ERR_NONE;
        auto           in_matrix  = static_cast<max::t_object*>(max::object_method(inputs, max::_jit_sym_getindex, 0));
        auto           out_matrix = static_cast<max::t_object*>(max::object_method(outputs, max::_jit_sym_getindex, 0));

        if (max::object_classname(in_matrix) != max::_jit_sym_jit_matrix)
            in_matrix = static_cast<max::t_object*>(max::object_method(in_matrix, k_sym_getmatrix));
        if (max::object_classname(out_matrix) != max::_jit_sym_jit_matrix)
            out_matrix = static_cast<max::t_object*>(max::object_method(out_matrix, k_sym_getmatrix));

        if (!self || !in_matrix || !out_matrix)
            err = max::JIT_ERR_INVALID_PTR;
        else {
            auto in_savelock  = max::object_method(in_matrix, max::_jit_sym_lock, reinterpret_cast<void*>(1));
            auto out_savelock = max::object_method(out_matrix, max::_jit_sym_lock, reinterpret_cast<void*>(1));

            max::t_jit_matrix_info in_minfo;
            max::t_jit_matrix_info out_minfo;
            max::object_method(in_matrix, max::_jit_sym_getinfo, &in_minfo);
            max::object_method(out_matrix, max::_jit_sym_getinfo, &out_minfo);

            uchar* in_bp  = nullptr;
            uchar* out_bp = nullptr;
            max::object_method(in_matrix, max::_jit_sym_getdata, &in_bp);
            max::object_method(out_matrix, max::_jit_sym_getdata, &out_bp);

            if (!in_bp)
                err = max::JIT_ERR_INVALID_INPUT;
            else if (!out_bp)
                err = max::JIT_ERR_INVALID_OUTPUT;
            else if (in_minfo.type != out_minfo.type)
                err = max::JIT_ERR_MISMATCH_TYPE;

            if (in_minfo.type == out_minfo.type && in_bp && out_bp) {
                long dim[max::JIT_MATRIX_MAX_DIMCOUNT];
                auto dim_count   = out_minfo.dimcount;
                auto plane_count = out_minfo.planecount;

                for (auto i = 0; i < dim_count; ++i) {
                    // if dimsize is 1, treat as infinite domain across that dimension.
                    // otherwise truncate if less than the output dimsize
                    dim[i] = out_minfo.dim[i];
                    if (in_minfo.dim[i] < dim[i] && in_minfo.dim[i] > 1) {
                        dim[i] = in_minfo.dim[i];
                    }
                }

                if (self->m_min_object.parallel_breakup_enabled()) {
                    max::jit_parallel_ndim_simplecalc2(reinterpret_cast<max::method>(jit_calculate_ndim<min_class_type>), self, dim_count,
                        dim, plane_count, &in_minfo, reinterpret_cast<char*>(in_bp), &out_minfo, reinterpret_cast<char*>(out_bp), 0, 0);
                }
                else {
                    jit_calculate_ndim<min_class_type>(self, dim_count, dim, plane_count, &in_minfo, reinterpret_cast<uchar*>(in_bp),
                        &out_minfo, reinterpret_cast<uchar*>(out_bp));
                }
            }

            max::object_method(out_matrix, max::_jit_sym_lock, out_savelock);
            max::object_method(in_matrix, max::_jit_sym_lock, in_savelock);
        }
        throw err;
    }


    // This is the "matrix_calc" used for processors (both input matrix and an output matrix)

    template<class min_class_type>
    max::t_jit_err jit_matrix_calc(minwrap<min_class_type>* self, max::t_object* inputs, max::t_object* outputs) {
        try {
            jit_matrix_docalc(self, inputs, outputs);
            return 0;
        }
        catch (max::t_jit_err& err) {
            return err;
        }
    }


    // This is the "matrix_calc" used for generators (no input matrix, only an output matrix)

    template<class min_class_type, enable_if_matrix_operator<min_class_type> = 0>
    void min_jit_mop_outputmatrix(max_jit_wrapper* self) {
        auto jitob      = static_cast<minwrap<min_class_type>*>(max::max_jit_obex_jitob_get(self));
        long outputmode = max::max_jit_mop_getoutputmode(self);
        auto mop        = max::max_jit_obex_adornment_get(self, max::_jit_sym_jit_mop);

        if (outputmode && mop && outputmode == 1) {    // always output unless output mode is none
            auto           outputs    = static_cast<max::t_object*>(max::object_method((max::t_object*)mop, max::_jit_sym_getoutputlist));
            max::t_jit_err err        = max::JIT_ERR_NONE;
            auto           out_mop_io = static_cast<max::t_object*>(max::object_method(outputs, max::_jit_sym_getindex, 0));
            auto           out_matrix = static_cast<max::t_object*>(max::object_method(out_mop_io, k_sym_getmatrix));

            if (!self || !out_matrix) {
                err = max::JIT_ERR_INVALID_PTR;
            }
            else {
                auto                   out_savelock = max::object_method(out_matrix, max::_jit_sym_lock, reinterpret_cast<void*>(1));
                max::t_jit_matrix_info out_minfo;
                char*                  out_bp = nullptr;

                max::object_method(out_matrix, max::_jit_sym_getinfo, &out_minfo);
                max::object_method(out_matrix, max::_jit_sym_getdata, &out_bp);

                if (!out_bp)
                    err = max::JIT_ERR_INVALID_OUTPUT;
                else {
                    if (jitob->m_min_object.parallel_breakup_enabled()) {
                        max::jit_parallel_ndim_simplecalc1(reinterpret_cast<max::method>(jit_calculate_ndim_single<min_class_type>), jitob,
                            out_minfo.dimcount, out_minfo.dim, out_minfo.planecount, &out_minfo, out_bp, 0);
                    }
                    else {
                        jit_calculate_ndim_single<min_class_type>(
                            jitob, out_minfo.dimcount, out_minfo.dim, out_minfo.planecount, &out_minfo, reinterpret_cast<uchar*>(out_bp));
                    }
                    max::object_method(out_matrix, max::_jit_sym_lock, out_savelock);
                }
            }
            max::max_jit_mop_outputmatrix(self);
        }
        else {
            max::max_jit_mop_outputmatrix(self);
        }
    }

}    // namespace c74::min
