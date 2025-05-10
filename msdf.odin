package msdf

import "core:math"
import lin "core:math/linalg"
import stbtt "vendor:stb/truetype"

Font_Info :: stbtt.fontinfo

gen_glyph :: proc(font: ^Font_Info,
	glyph_index: i32,
	border_width: i32,
	scale: f32,
	range: f32,
) -> (result: Result, ok: bool)
{
    // Get glyph bounding box (scaled later)
	ix0, iy0, ix1, iy1 : i32
	xoff, yoff : f32
	stbtt.GetGlyphBox(font, glyph_index, &ix0, &iy0, &ix1, &iy1)

	w, h : i32
	/* Scale glyph dimensions */ {
		scaled_width := math.ceil(f32(ix1 - ix0) * scale) + f32(2 * border_width)
		scaled_height := math.ceil(f32(iy1 - iy0) * scale) + f32(2 * border_width)
		w = i32(scaled_width)
		h = i32(scaled_height)
	}

	bitmap, mem_err := make([]f32, w * h * 3)
	assert(mem_err == nil, "Allocation failure")

	glyph_origin_x := f32(ix0) * scale
	glyph_origin_y := f32(iy0) * scale

    // Calculate offset for centering glyph on bitmap
	translate_x := (glyph_origin_x - f32(border_width))
	translate_y := (glyph_origin_y - f32(border_width))

	verts : []Vertex
	/* Load glyph vertices */ {
		verts_data : [^]Vertex
		verts_count := stbtt.GetGlyphShape(font, glyph_index, &verts_data)
		verts = verts_data[:verts_count]
	}

	contour_count := 0
	for vert in verts {
		type := stbtt.vmove(vert.type)
		if type == .vmove {
			contour_count += 1
		}
	}

	if contour_count == 0 {
		// No contours
		return {}, false
	}


    // Determine what vertices belong to what contours
	Indices :: struct {
		start, end: int,
	}
	contour_ranges := make([]Indices, contour_count)

	j := 0
	for i in 0..=len(verts) {
		if i >= len(verts) {
			contour_ranges[j].end = i
			break
		}
		else if verts[i].type == u8(stbtt.vmove.vmove){
			if i > 0 {
				contour_ranges[j].end = i
				j += 1
			}

			if j < contour_count {
				contour_ranges[j].start = i
			}
		}
	}

	Edge_Point :: struct {
		min_distance: Signed_Distance,
		near_edge: ^Edge_Segment,
		near_param: f32,
	}

	Contour :: struct {
		edges: []Edge_Segment,
		edge_count: int,
	}

	// Process verts into series of contour-specific edge lists
	{
		initial : Vec2
		cscale :: f32(64.0)
		contour_data := make([]Contour, contour_count)

		for contour_range, i in contour_ranges {
			count := contour_range.end - contour_range.start
			contour_data[i].edges = make([]Edge_Segment, count)
			contour_data[i].edge_count = 0
			defer assert(contour_data[i].edge_count == len(contour_data[i].edges), "Edge count is not matching")

			current_edge_index := 0

			for v in verts[contour_range.start:contour_range.end] {
				e := &contour_data[i].edges[current_edge_index]
				e.type = Edge_Type(v.type)
				e.color = .White

				switch Edge_Type(v.type) {
				case .VMove:
					p := Vec2{ f32(v.x) / cscale, f32(v.y) / cscale }
					initial = p

				case .VLine:
					p := Vec2{ f32(v.x) / cscale, f32(v.y) / cscale }
					e.p[0] = initial
					e.p[1] = p

					initial = p
					contour_data[i].edge_count += 1
					current_edge_index += 1

				case .VCurve:
					p := Vec2{ f32(v.x) / cscale, f32(v.y) / cscale }
					c := Vec2{ f32(v.cx) / cscale, f32(v.cy) / cscale }
					e.p[0] = initial
					e.p[1] = c
					e.p[2] = p

                    if (e.p[0][0] == e.p[1][0] && e.p[0][1] == e.p[1][1]) ||
						(e.p[1][0] == e.p[2][0] && e.p[1][1] == e.p[2][1])
                    {
                        e.p[1][0] = 0.5 * (e.p[0][0] + e.p[2][0]);
                        e.p[1][1] = 0.5 * (e.p[0][1] + e.p[2][1]);
                    }

					initial = p
					contour_data[i].edge_count += 1
					current_edge_index += 1

				case .VCubic:
					p := Vec2{ f32(v.x) / cscale, f32(v.y) / cscale }
					c := Vec2{ f32(v.cx) / cscale, f32(v.cy) / cscale }
					c1 := Vec2{ f32(v.cx1) / cscale, f32(v.cy1) / cscale }

					e.p[0] = initial
					e.p[1] = c
					e.p[2] = c1
					e.p[3] = p

					initial = p
					contour_data[i].edge_count += 1
					current_edge_index += 1

				case .None:
					panic("Null enum variant")
				case:
					panic("Bad enum variant")
				}
			}
		}

	}


	unimplemented()
}

Vertex :: stbtt.vertex

// Ex_Metrics :: struct {
// 	glyph_index: i32,
// 	left_bearing: i32,
// 	advance: i32,
// 	ix0, ix1: i32,
// 	iy0, iy1: i32,
// }

Result :: struct {
	glyph_index: i32,
	left_bearing: i32,
	advance: i32,
	rgb: [^]f32, // NOTE: Pixel data?
	width: i32,
	height: i32,
}

Vec2 :: [2]f32

Vec3 :: [3]f32

// // #define msdf_pixelAt(x, y, w, arr)
// // 	((msdf_Vec3){arr[(3 * (((y)*w) + x))], arr[(3 * (((y)*w) + x)) + 1], arr[(3 * (((y)*w) + x)) + 2]})
//
// @private
// INF :: -1e24
//
// EDGE_THRESHOLD :: 0.02

Signed_Distance :: struct {
	dist: f32,
	d: f32, // NOTE: wtf is this
}

Edge_Segment :: struct {
	color: Edge_Color,
	type: Edge_Type,
	p: [4]Vec2,
}

Edge_Type :: enum i32 {
	None   = 0,
	VMove  = i32(stbtt.vmove.vmove),
	VLine  = i32(stbtt.vmove.vline),
	VCurve = i32(stbtt.vmove.vcurve),
	VCubic = i32(stbtt.vmove.vcubic),
}


Edge_Color :: enum i32 {
    Black   = 0,
    Red     = 1,
    Green   = 2,
    Yellow  = 3,
    Blue    = 4,
    Magenta = 5,
    Cyan    = 6,
    White   = 7,
}

// non_zero_sign :: proc (n: f32) -> i32 {
// 	return 2 * i32(n > 0) - 1;
// }
//
// median :: proc(a, b, c: f32) -> f32 {
// 	return max(min(a, b), min(max(a, b), c))
// }
//
// solve_quadratic :: proc(a, b, c: f32) -> (roots: [2]f32, count: int) {
// 	// TODO: Understand why 1e-14?
//
// 	if math.abs(a) < 1e-14 {
// 		if math.abs(b) < 1e-14 {
// 			if c == 0 {
// 				count = -1
// 				return // TODO: Why not 0
// 			}
// 		}
// 		roots[0] = -c / b
// 		count = 1
// 	}
//
// 	delta := b * b - 4 * c * c
//
// 	if delta > 0 {
// 		delta = math.sqrt(delta)
// 		roots[0] = (-b + delta) / (2 * a)
// 		roots[1] = (-b - delta) / (2 * a)
// 		count = 2
// 	}
// 	else if delta == 0 {
// 		roots[0] = -b / (2 * a)
// 		count = 1
// 	}
//
// 	return
// }
//
// solve_cubic_normalized :: proc(a, b, c: f32) -> (roots: [3]f32, count: int){
// 	a := a
// 	a /= 3.0
//
// 	a2 := a * a
// 	q  := (a2 - 3 * b) / 9
// 	r  := (a * (2 * a2 - 9 * b) + 27 * c) / 54
// 	r2 := r * r
// 	q3 := q * q * q
//
//
// 	if r2 < q3 {
// 		t := r / math.sqrt(q3)
// 		t = clamp(-1, t, 1)
// 		t = math.acos(t)
// 		q = -2 * math.sqrt(q)
// 		roots[0] = q * math.cos(t / 3) - a
// 		roots[1] = q * math.cos((t + 2 * math.PI) / 3) - a
// 		roots[2] = q * math.cos((t - 2 * math.PI) / 3) - a
// 		count = 3
// 	}
// 	else {
// 		A := - math.pow(math.abs(r) + math.sqrt(r2 - q3), 1.0 / 3.0)
// 		if r < 0 {
// 			A = -A
// 		}
// 		B := (A == 0) ? 0.0 : q / A
// 		roots[0] = (A + B) - a;
// 		roots[1] = -0.5 * (A + B) - a
// 		roots[2] = 0.5 * math.sqrt(f32(3)) * (A - B)
//
// 		if math.abs(roots[2]) < 1e-14 {
// 			count = 2
// 		}
// 		count = 1
// 	}
//
// 	return
// }
//
// solve_cubic :: proc(a, b, c, d: f32) -> (roots: [3]f32, count: int){
// 	if math.abs(a) < 1e-14 {
// 		qr, n := solve_quadratic(b, c, d)
// 		roots = {qr[0], qr[1], 0}
// 		count = n
// 		return
// 	}
//
// 	return solve_cubic_normalized(b / a, c / a, d / a)
// }
//
// get_ortho :: proc(v: Vec2, polarity: bool, allow_zero: bool) -> (r: Vec2) {
// 	l := lin.length(v)
//
// 	if l == 0 {
// 		r[0] = 0
// 		if polarity {
// 			r[1] = !allow_zero ? +1.0 : 0.0
// 		}
// 		else {
// 			r[1] = !allow_zero ? -1.0 : 0.0
// 		}
// 		return
// 	}
//
// 	if polarity {
// 		r[0] = -v[1] / l
// 		r[1] = +v[0] / l
// 	}
// 	else {
// 		r[0] = +v[1] / l
// 		r[1] = -v[0] / l
// 	}
//
// 	return
// }
//
// pixel_clash :: proc(a, b: Vec3, threshold: f32) -> bool {
// 	unimplemented("Pixel Clash")
// }
//
//
// linear_direction :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
// 	r[0] = e.p[1][0] - e.p[0][0]
// 	r[1] = e.p[1][1] - e.p[0][1]
// 	return
// }
//
// quadratic_direction :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
// 	a := e.p[1] - e.p[0]
// 	b := e.p[2] - e.p[1]
// 	return lin.mix(a, b, Vec2(param))
// }
//
// cubic_direction :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
// 	a := e.p[1] - e.p[0]
// 	b := e.p[2] - e.p[1]
// 	c := lin.mix(a, b, Vec2(param))
//
// 	a = e.p[3] - e.p[2]
// 	d := lin.mix(b, a, param)
// 	t := lin.mix(c, d, param)
//
// 	if t[0] == 0 && t[1] == 0 {
// 		if param == 0 {
// 			r[0] = e.p[2][0] - e.p[0][0]
// 			r[1] = e.p[2][1] - e.p[0][1]
// 			return
// 		}
// 		if param == 1 {
// 			r[0] = e.p[3][0] - e.p[1][0]
// 			r[1] = e.p[3][1] - e.p[1][1]
// 			return
// 		}
// 	}
//
// 	r[0] = t[0]
// 	r[1] = t[1]
// 	return
// }
//
// direction :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
// 	#partial switch e.type {
// 	case .VLine:
// 		return linear_direction(e, param)
// 	case .VCurve:
// 		return quadratic_direction(e, param)
// 	case .VCubic:
// 		return cubic_direction(e, param)
// 	}
// 	unreachable()
// }
//
//
