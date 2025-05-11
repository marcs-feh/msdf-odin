package msdf

import "core:math"
import "core:fmt"
import lin "core:math/linalg"
import stbtt "vendor:stb/truetype"

print_edge :: proc(e: Edge_Segment){
	switch e.type {
	case .VLine:
		fmt.printf("Line: (%.4f %.4f) %v", e.p[0], e.p[1], e.color)
	case .VCurve:
		fmt.printf("Quadratic: (%.4f %.4f %.4f) %v", e.p[0], e.p[1], e.p[2], e.color)
	case .VCubic:
		fmt.printf("Cubic: (%.4f %.4f %.4f %.4f) %v", e.p[0], e.p[1], e.p[2], e.p[3], e.color)
	case .VMove:
		fmt.print("Move")
	case .None:
		fmt.print("<<<NONE>>>")
	}
}

print_multi_distance :: proc(msd: Multi_Distance){
	fmt.printf("R: %.4f G: %.4f B: %.4f (Med: %.4f)", msd.r, msd.g, msd.b, msd.med);
}

print_signed_distance :: proc(d: Signed_Distance){
	fmt.printf("Dist: %.4f Dot: %.4f", d.distance, d.dot);
}

Font_Info :: stbtt.fontinfo

Result :: struct {
	glyph_index: i32,
	left_bearing: i32,
	y_offset: i32,
	advance: i32,
	values: [][3]f32,
	width: i32,
	height: i32,
}

Vertex :: stbtt.vertex

Vec2 :: [2]f32

Vec3 :: [3]f32

gen_glyph :: proc(font: ^Font_Info,
	glyph_index: i32,
	border_width: i32,
	scale: f32,
	range: f32,
) -> (result: Result)
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

	glyph_origin_x := i32(f32(ix0) * scale)
	glyph_origin_y := i32(f32(iy0) * scale)

	// Calculate offset for centering glyph on bitmap
	translate_x := (glyph_origin_x - border_width)
	translate_y := (glyph_origin_y - border_width)

	fmt.println("--- Bitmap Metrics ---")
	fmt.println("Width:", w)
	fmt.println("Height:", h)
	fmt.println("Translate X:", translate_x)
	fmt.println("Translate Y:", translate_y)

	verts : []Vertex
	/* Load glyph vertices */ {
		verts_data : [^]Vertex
		verts_count := stbtt.GetGlyphShape(font, glyph_index, &verts_data)
		verts = verts_data[:verts_count]
	}

	contours := contours_from_vertices(verts)
	if len(contours) == 0 {
		return {}
	}

	when ODIN_DEBUG {
		fmt.println("--- Loaded Contours ---")
		for contour, i in contours {
			fmt.println("Contour", i)
			for e in contour.edges {
				fmt.print("\t")
				print_edge(e)
				fmt.println()
			}
		}
	}

	/* Color edges */
	color_edges(contours[:], 0)

	when ODIN_DEBUG {
		fmt.println("--- Colored Edges ---")
		for contour, i in contours {
			fmt.println("Contour", i)
			for e in contour.edges {
				fmt.print("\t")
				print_edge(e)
				fmt.println()
			}
		}
	}

	/* Normalize shape */ {
		for &contour, i in contours {
			if len(contour.edges) == 1 {
				p0, p1, p2 := split(contour.edges[0])
				clear(&contour.edges)
				append(&contour.edges, p0, p1, p2)
			}
		}
	}

	when ODIN_DEBUG {
		fmt.println("--- Normalized Shape ---")
		for contour, i in contours {
			fmt.println("Contour", i)
			for e in contour.edges {
				fmt.print("\t")
				print_edge(e)
				fmt.println()
			}
		}
	}

	windings := make([dynamic]int, 0, len(contours))

	/* Calculate Windings */ {
		for &contour, i in contours {
			edge_count := len(contour.edges)
			if edge_count == 0 {
				append(&windings, 0)
			}
			else {
				append(&windings, contour_winding(contour))
			}
		}
	}
	assert(len(windings) == len(contours), "Mismatched winding to contour")

	when ODIN_DEBUG {
		fmt.println("--- Calculated Windings ---")
		for contour, i in contours {
			fmt.println("Contour", i)
			fmt.printfln("\t%d", windings[i])
		}
	}

	/* Calculate Signed Distances */ {
		contour_dists := make([dynamic]Multi_Distance, 0, len(contours))
		inv_range := 1.0 / range

		initial_edge_point :: Edge_Point {
			min_distance = { distance = INF, dot = 1 }
		}

		for y in 0..<h {
			row := iy0 > iy1 ? y : h - y - 1
			for x in 0..<w {
				a64 :: 64.0
				p := Vec2{
					(f32(translate_x) + f32(x) + xoff) / (scale * a64),
					(f32(translate_y) + f32(y) + yoff) / (scale * a64),
				}
				sr, sg, sb : Edge_Point

				sr = initial_edge_point
				sg = initial_edge_point
				sb = initial_edge_point

				d : f32 = abs(INF)
				neg_dist : f32 = -INF
				pos_dist : f32 = INF

				winding := 0
				// Calculate distance to contours from current point (and if its inside or outside of the shape?)
				for &contour, contour_index in contours {
					r, g, b : Edge_Point
					dist : Signed_Distance
					param : f32
					r = initial_edge_point
					g = initial_edge_point
					b = initial_edge_point

					for &edge, k in contour.edges {
						dist, param = distance(edge, p)

						if bool(edge.color & .Red) && signed_lt(dist, r.min_distance) {
							r.min_distance = dist
							r.near_edge = edge
							r.near_param = param
						}

						if bool(edge.color & .Green) && signed_lt(dist, g.min_distance) {
							g.min_distance = dist
							g.near_edge = edge
							g.near_param = param
						}

						if bool(edge.color & .Blue) && signed_lt(dist, b.min_distance) {
							b.min_distance = dist
							b.near_edge = edge
							b.near_param = param
						}
					}

					if signed_lt(r.min_distance, sr.min_distance){
						sr = r
					}
					if signed_lt(g.min_distance, sg.min_distance){
						sg = g
					}
					if signed_lt(b.min_distance, sb.min_distance){
						sb = b
					}

					abs_med_min_dist := abs(median(r.min_distance.distance, g.min_distance.distance, b.min_distance.distance))

					if abs_med_min_dist < d {
						d = abs_med_min_dist
						winding = -windings[contour_index]
					}

					if near_edge, ok := r.near_edge.?; ok {
						r.min_distance = distance_to_pseudo(r.min_distance, p, r.near_param, near_edge)
					}
					if near_edge, ok := g.near_edge.?; ok {
						g.min_distance = distance_to_pseudo(g.min_distance, p, g.near_param, near_edge)
					}
					if near_edge, ok := b.near_edge.?; ok {
						b.min_distance = distance_to_pseudo(b.min_distance, p, b.near_param, near_edge)
					}

					// NOTE: Calculate this *after* pseudo_dist transformation
					med_min_dist := median(r.min_distance.distance, g.min_distance.distance, b.min_distance.distance)
					append(&contour_dists, Multi_Distance {
						r = r.min_distance.distance,
						g = g.min_distance.distance,
						b = b.min_distance.distance,
						med = med_min_dist,
					})

					if windings[contour_index] > 0 && med_min_dist >= 0 && abs(med_min_dist) < abs(pos_dist) {
						pos_dist = med_min_dist
					}

					if windings[contour_index] < 0 && med_min_dist <= 0 && abs(med_min_dist) < abs(neg_dist) {
						neg_dist = med_min_dist
					}

					{
						fmt.printf("Distance from point (%.4f %.4f) to contour %d: ", p[0], p[1], contour_index);
						print_multi_distance(contour_dists[contour_index])
						fmt.println()
					}
				}


				if near_edge, ok := sr.near_edge.?; ok {
					sr.min_distance = distance_to_pseudo(sr.min_distance, p, sr.near_param, near_edge)
				}
				if near_edge, ok := sg.near_edge.?; ok {
					sg.min_distance = distance_to_pseudo(sg.min_distance, p, sg.near_param, near_edge)
				}
				if near_edge, ok := sb.near_edge.?; ok {
					sb.min_distance = distance_to_pseudo(sb.min_distance, p, sb.near_param, near_edge)
				}

				msd := Multi_Distance {
					r = INF,
					g = INF,
					b = INF,
					med = INF,
				}

				if pos_dist >= 0 && abs(pos_dist) < abs(neg_dist) {
					msd.med = INF
					winding = 1
					for contour, i in contours {
						if windings[i] > 0 && contour_dists[i].med > msd.med && abs(contour_dists[i].med) < abs(neg_dist) {
							msd = contour_dists[i]
						}
					}
				}
				else if neg_dist <= 0 && abs(neg_dist) <= abs(pos_dist) {
					msd.med = -INF
					winding = -1
					for contour, i in contours {
						if windings[i] < 0 && contour_dists[i].med < msd.med && abs(contour_dists[i].med) < abs(pos_dist) {
							msd = contour_dists[i]
						}
					}
				}

				for contour, i in contours {
					if windings[i] != winding && abs(contour_dists[i].med) < abs(msd.med) {
						msd = contour_dists[i]
					}
				}

				if (median(sr.min_distance.distance, sg.min_distance.distance, sb.min_distance.distance) == msd.med) {
					msd.r = sr.min_distance.distance
					msd.g = sg.min_distance.distance
					msd.b = sb.min_distance.distance
				}

				bitmap_index := 3 * ((row * w) + x)

				mr := msd.r * inv_range + 0.5
				mg := msd.g * inv_range + 0.5
				mb := msd.b * inv_range + 0.5

				bitmap[bitmap_index + 0] = mr
				bitmap[bitmap_index + 1] = mr
				bitmap[bitmap_index + 2] = mr
			}
		}
	}

	result.glyph_index = glyph_index
	result.width = w
	result.height = h
	result.y_offset = i32(translate_y)
	result.values = (transmute([^][3]f32)raw_data(bitmap))[:w * h]
	return result
}

signed_lt :: proc(a, b: Signed_Distance) -> bool {
	return abs(a.distance) < abs(b.distance) ||
			(abs(a.distance) == abs(b.distance) && a.dot < b.dot)
}

shoelace :: proc(a, b: Vec2) -> f32 {
	return (b[0] - a[0]) * (a[1] + b[1])
}

distance :: proc(e: Edge_Segment, origin: Vec2) -> (sd: Signed_Distance, param: f32){

	#partial switch e.type {
	case .VLine:
		return distance_linear(e, origin)
	case .VCurve:
		return distance_quadratic(e, origin)
	case .VCubic:
		return distance_cubic(e, origin)
	}

	panic("Invalid edge segment type")
}

distance_linear :: proc(e: Edge_Segment, origin: Vec2) -> (sd: Signed_Distance, param: f32){
	assert(e.type == .VLine)

	aq := origin - e.p[0]
	ab := e.p[1] - e.p[0]

	param = lin.inner_product(aq, ab) / lin.inner_product(ab, ab)

	eq := e.p[int(param > 0.5)] - origin

	endpoint_distance := lin.length(eq)
	if param > 0 && param < 1 {
		ab_ortho := get_ortho(ab, false, false)
		ortho_dist := lin.inner_product(ab_ortho, aq)
		if abs(ortho_dist) < endpoint_distance {
			sd = { ortho_dist, 0 }
			return
		}
	}

	ab = lin.normalize(ab)
	eq = lin.normalize(eq)

	sd.distance = non_zero_sign(lin.cross(aq, ab)) * endpoint_distance
	sd.dot = abs(lin.inner_product(ab, eq))

	when ODIN_DEBUG {
		fmt.printf("\tLinear Dist from ")
		print_edge(e)
		fmt.print(" to", origin, "is ")
		print_signed_distance(sd)
		fmt.println()
	}

	return
}

distance_quadratic :: proc(e: Edge_Segment, origin: Vec2) -> (sd: Signed_Distance, param: f32){
	assert(e.type == .VCurve)
	qa := e.p[0] - origin
	ab := e.p[1] - e.p[0]
	br := e.p[2] - e.p[1] - ab

	a := lin.inner_product(br, br)
	b := 3 * lin.inner_product(ab, br)
	c := 2 * lin.inner_product(ab, ab) + lin.inner_product(qa, br)
	d := lin.inner_product(qa, ab)

	roots, solutions := solve_cubic(a, b, c, d)

	ep_dir := direction(e, 0)
	// Distance from A
	min_dist := non_zero_sign(lin.cross(ep_dir, qa)) * lin.length(qa)
	param = - lin.inner_product(qa, ep_dir) / lin.inner_product(ep_dir, ep_dir)

	{
		ep_dir = direction(e, 1)
		// Distance from B
		dist := lin.length(e.p[2] - origin)
		if dist < abs(min_dist){
			min_dist = non_zero_sign(lin.cross(ep_dir, e.p[2] - origin)) * dist
			param = lin.inner_product(origin - e.p[1], ep_dir) / lin.inner_product(ep_dir, ep_dir)
		}
	}

	for root, i in roots {
		if root > 0 && root < 1 {
			qe := qa + 2 * root * ab + root * root * br
			dist := lin.length(qe)

			if dist <= abs(min_dist){
				min_dist = non_zero_sign(lin.inner_product(ab + root * br, qe)) * dist
				param = root
			}
		}
	}

	if param >= 0 && param <= 1 {
		sd = { min_dist, 0 }
	}
	else {
		if param < 0.5 {
			sd = { min_dist, abs(lin.inner_product(lin.normalize(direction(e, 0)), lin.normalize(qa))) }
		}
		else {
			sd = { min_dist, abs(lin.inner_product(lin.normalize(direction(e, 1)), lin.normalize(e.p[2] - origin))) }
		}
	}

	when ODIN_DEBUG {
		fmt.printf("\tQuad Dist from ")
		print_edge(e)
		fmt.print(" to", origin, "is ")
		print_signed_distance(sd)
		fmt.println()
	}

	return
}

CUBIC_SEARCH_STARTS :: 4

CUBIC_SEARCH_STEPS :: 4

distance_cubic :: proc(e: Edge_Segment, origin: Vec2) -> (sd: Signed_Distance, param: f32){
	assert(e.type == .VCubic)

	qa := e.p[0] - origin
	ab := e.p[1] - e.p[0]
	br := e.p[2] - e.p[1] - ab
	as := (e.p[3] - e.p[2]) - (e.p[2] - e.p[1]) - br

	ep_dir := direction(e, 0)
	min_dist := non_zero_sign(lin.inner_product(ep_dir, qa)) * lin.length(qa) // Distance from A
	param = -lin.inner_product(qa, ep_dir) / lin.inner_product(ep_dir, ep_dir)
	{
		ep_dir = direction(e, 1)
		dist := lin.length(e.p[3] - origin) // Distance from B

		if dist < abs(min_dist) {
			min_dist = non_zero_sign(lin.cross(ep_dir, e.p[3]-origin)) * dist
			param = lin.inner_product(ep_dir - (e.p[3] - origin), ep_dir) / lin.inner_product(ep_dir, ep_dir)
		}
	}

	// Iterative minimum distance seach
	for i in 0..=CUBIC_SEARCH_STARTS {
		t := f32(i) / CUBIC_SEARCH_STARTS
		qe := qa + (3 * t * ab) + (3 * t * t * br) + (t * t * t * as)

		for step in 0..<CUBIC_SEARCH_STEPS {
			// Improve t
			d1 := (3 * ab) + (6 * t * br) + (3 * t * t * as)
			d2 := (6 * br) + (6 * t * as)

			t -= lin.inner_product(qe, d1) / (lin.inner_product(d1, d1) + lin.inner_product(qe, d2))
			if t <= 0 || t >= 1 {
				break
			}

			qe = qa + (3 * t * ab) + (3 * t * t * br) + (t * t * t * as)
			dist := lin.length(qe)
			if dist < abs(min_dist){
				min_dist = non_zero_sign(lin.cross(d1, qe)) * dist
				param = t
			}
		}
	}

	if param >= 0 && param <= 1 {
		sd = { min_dist, 0 }
	}
	else {
		if param < .5 {
			sd = { min_dist, abs(lin.inner_product(lin.normalize(direction(e, 0)), lin.normalize(qa))) }
		}
		else {
			sd = { min_dist, abs(lin.inner_product( lin.normalize(direction(e, 1)), lin.normalize(e.p[3] - origin))) }
		}
	}

	when ODIN_DEBUG {
		fmt.printf("\tCube Dist from ")
		print_edge(e)
		fmt.print(" to", origin, "is ")
		print_signed_distance(sd)
		fmt.println()
	}

	return
}

contour_winding :: proc(contour: Contour) -> int {
	edge_count := len(contour.edges)
	total := f32(0)

	switch edge_count {
	case 0:
		return 0
	case 1:
		a := point(contour.edges[0], 0)
		b := point(contour.edges[0], 1.0 / 3.0)
		c := point(contour.edges[0], 2.0 / 3.0)
		total += shoelace(a, b)
		total += shoelace(b, c)
		total += shoelace(c, a)
	case 2:
		a := point(contour.edges[0], 0.0)
		b := point(contour.edges[0], 0.5)
		c := point(contour.edges[1], 0.0)
		d := point(contour.edges[1], 0.5)

		total += shoelace(a, b)
		total += shoelace(b, c)
		total += shoelace(c, d)
		total += shoelace(d, a)
	case:
		prev := point(contour.edges[edge_count - 1], 0)
		for edge in contour.edges {
			cur := point(edge, 0)
			total += shoelace(prev, cur)
			prev = cur
		}
	}

	return int(non_zero_sign(total))
}

is_corner :: proc(a, b: Vec2, cross_threshold: f32) -> bool {
	return lin.inner_product(a, b) <= 0 || abs(lin.cross(a, b)) > cross_threshold
}

switch_color :: proc(color: ^Edge_Color, seed: ^u64, banned: Edge_Color){
	combined := color^ & banned

	if combined == .Red || combined == .Green || combined == .Blue {
		color^ = combined ~ .White
	}
	else if color^ == .Black || color^ == .White {
		start := [3]Edge_Color{ .Cyan, .Magenta, .Yellow }
		color^ = start[seed^ & 3]
		seed^ /= 3
	}
	else {
		shifted := i32(color^) << (1 + (seed^ & 1))
		color^ = Edge_Color(shifted | (shifted >> 3)) & .White
		seed^ >>= 1
	}
}

// // #define msdf_pixelAt(x, y, w, arr)
// // 	((msdf_Vec3){arr[(3 * (((y)*w) + x))], arr[(3 * (((y)*w) + x)) + 1], arr[(3 * (((y)*w) + x)) + 2]})
//
// @private
// INF :: -1e24
//
// EDGE_THRESHOLD :: 0.02

// Ex_Metrics :: struct {
// 	glyph_index: i32,
// 	left_bearing: i32,
// 	advance: i32,
// 	ix0, ix1: i32,
// 	iy0, iy1: i32,
// }

Signed_Distance :: struct {
	distance: f32,
	dot: f32,
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
	// bits: BGR
	Black   = 0b000,
	Red     = 0b001,
	Green   = 0b010,
	Yellow  = 0b011,
	Blue    = 0b100,
	Magenta = 0b101,
	Cyan    = 0b110,
	White   = 0b111,
}

direction :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
	#partial switch e.type {
	case .VLine:
		return direction_linear(e, param)
	case .VCurve:
		return direction_quadratic(e, param)
	case .VCubic:
		return direction_cubic(e, param)
	}
	panic("Invalid segment type")
}

direction_linear :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
	assert(e.type == .VLine)
	r[0] = e.p[1][0] - e.p[0][0]
	r[1] = e.p[1][1] - e.p[0][1]
	return
}

direction_quadratic :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
	assert(e.type == .VCurve)
	a := e.p[1] - e.p[0]
	b := e.p[2] - e.p[1]
	return lin.mix(a, b, Vec2(param))
}

direction_cubic :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
	assert(e.type == .VCubic)
	a := e.p[1] - e.p[0]
	b := e.p[2] - e.p[1]
	c := lin.mix(a, b, Vec2(param))

	a = e.p[3] - e.p[2]
	d := lin.mix(b, a, param)
	t := lin.mix(c, d, param)

	if t[0] == 0 && t[1] == 0 {
		if param == 0 {
			r[0] = e.p[2][0] - e.p[0][0]
			r[1] = e.p[2][1] - e.p[0][1]
			return
		}
		if param == 1 {
			r[0] = e.p[3][0] - e.p[1][0]
			r[1] = e.p[3][1] - e.p[1][1]
			return
		}
	}

	r[0] = t[0]
	r[1] = t[1]
	return
}

point :: proc(e: Edge_Segment, param: f32) -> (r: Vec2) {
	#partial switch(e.type){
	case .VLine:
		return point_linear(e, param)
	case .VCurve:
		return point_quadratic(e, param)
	case .VCubic:
		return point_cubic(e, param)
	}

	panic("Invalid segment type")
}

point_linear :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
	assert(e.type == .VLine)
	return lin.mix(e.p[0], e.p[1], param)
}

point_quadratic :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
	assert(e.type == .VCurve)
	a := lin.mix(e.p[0], e.p[1], param)
	b := lin.mix(e.p[1], e.p[2], param)
	return lin.mix(a, b, param)
}

point_cubic :: proc(e: Edge_Segment, param: f32) -> (r: Vec2){
	assert(e.type == .VCubic)
	p12 := lin.mix(e.p[1], e.p[2], param)

	a := lin.mix(e.p[0], e.p[1], param)
	b := lin.mix(a, p12, param)

	c := lin.mix(e.p[2], e.p[3], param)
	d := lin.mix(p12, c, param)

	return lin.mix(b, d, param)
}


@(require_results)
split :: proc(e: Edge_Segment) -> (p1, p2, p3: Edge_Segment){
	#partial switch(e.type){
	case .VLine:
		return split_linear(e)
	case .VCurve:
		return split_quadratic(e)
	case .VCubic:
		return split_cubic(e)
	}

	panic("Invalid segment type")
}

split_linear :: proc(e: Edge_Segment) -> (p1, p2, p3: Edge_Segment){
	assert(e.type == .VLine)

	p1.p[0] = e.p[0]
	p1.p[1] = point(e, 1.0 / 3.0)

	p2.p[0] = point(e, 1.0 / 3.0)
	p2.p[1] = point(e, 2.0 / 3.0)

	p3.p[0] = point(e, 2.0 / 3.0)
	p3.p[1] = e.p[1]

	p1.type, p2.type, p3.type = e.type, e.type, e.type
	p1.color, p2.color, p3.color = e.color, e.color, e.color
	return
}

split_quadratic :: proc(e: Edge_Segment) -> (p1, p2, p3: Edge_Segment){
	assert(e.type == .VCurve)
	p1.p[0] = e.p[0]
	p1.p[1] = lin.mix(e.p[0], e.p[1], 1.0 / 3.0)
	p1.p[2] = point(e, 1.0 / 3.0)

	p2.p[0] = point(e, 1.0 / 3.0)
	a := lin.mix(e.p[0], e.p[1], 5.0 / 9.0)
	b := lin.mix(e.p[1], e.p[2], 4.0 / 9.0)
	p2.p[1] = lin.mix(a, b, 0.5)
	p2.p[2] = point(e, 2.0 / 3.0)

	p3.p[0] = point(e, 2.0 / 3.0)
	p3.p[1] = lin.mix(e.p[1], e.p[2], 2.0 / 3.0)
	p3.p[2] = e.p[2]

	p1.type, p2.type, p3.type = e.type, e.type, e.type
	p1.color, p2.color, p3.color = e.color, e.color, e.color
	return
}

split_cubic :: proc(e: Edge_Segment) -> (p1, p2, p3: Edge_Segment){
	assert(e.type == .VCubic)

	/* P1 */ {
		a, b : Vec2
		p1.p[0] = e.p[0]

		p1.p[1] = lin.mix(e.p[0], e.p[1], 1.0 / 3.0)

		a = lin.mix(e.p[0], e.p[1], 1.0 / 3.0)
		b = lin.mix(e.p[1], e.p[2], 1.0 / 3.0)
		p1.p[2] = lin.mix(a, b, 1.0 / 3.0)

		p1.p[3] = point(e, 1.0 / 3.0)
	}

	/* P2 */ {
		a, b, c, d : Vec2
		p2.p[0] = point(e, 1.0 / 3.0)

		a = lin.mix( e.p[0], e.p[1], 1.0 / 3.0)
		b = lin.mix( e.p[1], e.p[2], 1.0 / 3.0)
		c = lin.mix( a, b, 1.0 / 3.0)

		a = lin.mix( e.p[1], e.p[2], 1.0 / 3.0)
		b = lin.mix( e.p[2], e.p[3], 1.0 / 3.0)
		d = lin.mix( a, b, 1.0 / 3.0)

		p2.p[1] = lin.mix(c, d, 2.0 / 3.0)

		a = lin.mix(e.p[0], e.p[1], 2.0 / 3.0)
		b = lin.mix(e.p[1], e.p[2], 2.0 / 3.0)
		c = lin.mix(a, b, 2.0 / 3.0)

		a = lin.mix(e.p[1], e.p[2], 2.0 / 3.0)
		b = lin.mix(e.p[2], e.p[3], 2.0 / 3.0)
		d = lin.mix(a, b, 2.0 / 3.0)

		p2.p[2] = lin.mix(c, d, 1.0 / 3.0)

		p2.p[3] = point(e, 2.0 / 3.0)
	}

	/* P3 */ {
		a, b : Vec2

		p3.p[0] = point(e, 2.0 / 3.0)

		a = lin.mix(e.p[1], e.p[2], 2.0 / 3.0)
		b = lin.mix(e.p[2], e.p[3], 2.0 / 3.0)
		p3.p[1] = lin.mix(a, b, 2.0 / 3.0)

		p3.p[2] = lin.mix(e.p[2], e.p[3], 2.0 / 3.0)
	}


	p1.type, p2.type, p3.type = e.type, e.type, e.type
	p1.color, p2.color, p3.color = e.color, e.color, e.color
	return
}

// median :: proc(a, b, c: f32) -> f32 {
// 	return max(min(a, b), min(max(a, b), c))
// }

solve_quadratic :: proc(a, b, c: f32) -> (roots: [2]f32, count: int) {
	// NOTE: x is 0 means: x is very close to zero or insignificant in the context of the equation

	if math.abs(a) < 1e-14 || math.abs(b) > 1e12 * abs(a) { /* a is 0 -> Linear equation */
		if math.abs(b) < 1e-14 { /* a is 0, b is 0 -> No solution */
			if c == 0 {
				count = -1
				return // TODO: Why not 0
			}
		}
		roots[0] = -c / b
		count = 1
	}

	delta := b * b - 4 * c * c

	if delta > 0 {
		delta = math.sqrt(delta)
		roots[0] = (-b + delta) / (2 * a)
		roots[1] = (-b - delta) / (2 * a)
		count = 2
	}
	else if delta == 0 {
		roots[0] = -b / (2 * a)
		count = 1
	}

	return
}

solve_cubic_normalized :: proc(a, b, c: f32) -> (roots: [3]f32, count: int){
	a := a
	a /= 3.0

	a2 := a * a
	q  := (a2 - 3 * b) / 9
	r  := (a * (2 * a2 - 9 * b) + 27 * c) / 54
	r2 := r * r
	q3 := q * q * q

	if r2 < q3 {
		t := r / math.sqrt(q3)
		t = clamp(-1, t, 1)
		t = math.acos(t)
		q = -2 * math.sqrt(q)
		roots[0] = q * math.cos(t / 3) - a
		roots[1] = q * math.cos((t + 2 * math.PI) / 3) - a
		roots[2] = q * math.cos((t - 2 * math.PI) / 3) - a
		count = 3
	}
	else {
		A := - math.pow(math.abs(r) + math.sqrt(r2 - q3), 1.0 / 3.0)
		if r < 0 {
			A = -A
		}
		B := (A == 0) ? 0.0 : q / A
		roots[0] = (A + B) - a
		roots[1] = -0.5 * (A + B) - a
		roots[2] = 0.5 * math.sqrt(f32(3)) * (A - B)

		if math.abs(roots[2]) < 1e-14 {
			count = 2
		}
		count = 1
	}

	return
}

solve_cubic :: proc(a, b, c, d: f32) -> (roots: [3]f32, count: int){
	if math.abs(a) < 1e-14 {
		qr, n := solve_quadratic(b, c, d)
		roots = {qr[0], qr[1], 0}
		count = n
		return
	}

	return solve_cubic_normalized(b / a, c / a, d / a)
}

get_ortho :: proc(v: Vec2, polarity: bool, allow_zero: bool) -> (r: Vec2) {
	l := lin.length(v)

	if l == 0 {
		r[0] = 0
		if polarity {
			r[1] = !allow_zero ? +1.0 : 0.0
		}
		else {
			r[1] = !allow_zero ? -1.0 : 0.0
		}
		return
	}

	if polarity {
		r[0] = -v[1] / l
		r[1] = +v[0] / l
	}
	else {
		r[0] = +v[1] / l
		r[1] = -v[0] / l
	}

	return
}

// pixel_clash :: proc(a, b: Vec3, threshold: f32) -> bool {
// 	unimplemented("Pixel Clash")
// }

Vertex_Range :: struct {
	start, end: int,
}

// Segment a series of vertices into ranges, each representing a contour
segment_contour_ranges :: proc(verts: []Vertex) -> []Vertex_Range {
	contour_count := 0
	for vert in verts {
		type := stbtt.vmove(vert.type)
		if type == .vmove {
			contour_count += 1
		}
	}

	if contour_count == 0 {
		// No contours
		return nil
	}

	// Determine what vertices belong to what contours
	vert_ranges := make([]Vertex_Range, contour_count)

	current_range_idx := 0
	for i in 0..=len(verts) {
		if i >= len(verts) {
			vert_ranges[current_range_idx].end = i
			break
		}
		else if verts[i].type == u8(stbtt.vmove.vmove){
			if i > 0 {
				vert_ranges[current_range_idx].end = i
				current_range_idx += 1
			}

			if current_range_idx < contour_count {
				vert_ranges[current_range_idx].start = i
			}
		}
	}

	// fmt.println(vert_ranges)
	assert(len(vert_ranges) == contour_count, "Mismatched ranges to contour")
	return vert_ranges
}

Edge_Point :: struct {
	min_distance: Signed_Distance,
	near_edge: Maybe(Edge_Segment),
	near_param: f32,
}

Contour :: struct {
	edges: [dynamic]Edge_Segment,
}

// Process list of vertices into individual contours
contours_from_vertices :: proc(verts: []Vertex) -> []Contour {
	initial : Vec2
	cscale :: f32(64.0)
	contour_ranges := segment_contour_ranges(verts)
	contour_count := len(contour_ranges)
	contours := make([]Contour, contour_count)

	for contour_range, i in contour_ranges {
		count := contour_range.end - contour_range.start
		contours[i].edges = make([dynamic]Edge_Segment, 0, count)

		// defer assert(contours[i].edge_count == len(contours[i].edges) - 1, "Unexpected edge count")

		for v in verts[contour_range.start:contour_range.end] {
			e : Edge_Segment
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
				append(&contours[i].edges, e)

			case .VCurve:
				p := Vec2{ f32(v.x) / cscale, f32(v.y) / cscale }
				c := Vec2{ f32(v.cx) / cscale, f32(v.cy) / cscale }
				e.p[0] = initial
				e.p[1] = c
				e.p[2] = p

				if ((e.p[0][0] == e.p[1][0] && e.p[0][1] == e.p[1][1]) ||
					(e.p[1][0] == e.p[2][0] && e.p[1][1] == e.p[2][1]))
				{
					e.p[1][0] = 0.5 * (e.p[0][0] + e.p[2][0])
					e.p[1][1] = 0.5 * (e.p[0][1] + e.p[2][1])
				}

				initial = p
				append(&contours[i].edges, e)

			case .VCubic:
				p := Vec2{ f32(v.x) / cscale, f32(v.y) / cscale }
				c := Vec2{ f32(v.cx) / cscale, f32(v.cy) / cscale }
				c1 := Vec2{ f32(v.cx1) / cscale, f32(v.cy1) / cscale }

				e.p[0] = initial
				e.p[1] = c
				e.p[2] = c1
				e.p[3] = p

				initial = p
				append(&contours[i].edges, e)

			case .None:
				panic("Null enum variant")
			case:
				panic("Bad enum variant")
			}
		}
	}

	return contours
}

color_edges :: proc(contours: []Contour, seed: u64){
	seed := seed
	angle_threshold :: f32(3.0)
	cross_threshold := f32(math.sin(angle_threshold))
	corners := make([dynamic]int, 0, len(contours))

	for &contour, i in contours {
		if len(contour.edges) == 0 { continue }

		/* Identify corners */ {
			// clear(&corners)
			prev_dir := direction(contour.edges[len(contour.edges) - 1], 1.0)

			for edge, index in contour.edges {
				dir := direction(edge, 0)

				dir = lin.normalize(dir)
				prev_dir = lin.normalize(prev_dir)

				if is_corner(prev_dir, dir, cross_threshold) {
					append(&corners, index)
				}
				prev_dir = direction(edge, 1)
			}
		}

		when ODIN_DEBUG {
			fmt.printfln("--- Identified Corners %d ---", i)
			for c in corners {
				fmt.printfln("\t%d", c)
			}
		}

		if len(corners) == 0 { /* No corners, smooth shape */
			for edge, edge_idx in contour.edges {
				contour.edges[edge_idx].color = .White
			}
		}
		else if len(corners) == 1 { /* "Teardrop" like shape */
			colors := [3]Edge_Color{.White, .White, .Black}
			switch_color(&colors[0], &seed, .Black)
			colors[2] = colors[0]
			switch_color(&colors[2], &seed, .Black)

			corner := corners[0]
			if len(contour.edges) >= 3 { /* Enough edges to "spread" colors */
				m := len(contour.edges)
				for j in 0..<m {
					// NOTE: I have zero fucking idea why this works, the original code is even more arcane
					// contour_data[i].edges[(corner + j) % m].color = (colors + 1)[(int)(3 + 2.875 * i / (m - 1) - 1.4375 + .5) - 3];
					contour.edges[(corner + j) % m].color = colors[1 + symmetrical_trichotomy(i, m)]
				}
			}
			else if len(contour.edges) >= 1 { /* Less than three edge segments for three colors -> edges must be split */
				parts := [7]Edge_Segment{}
				c_off := 3 * corner

				parts[0 + c_off], parts[1 + c_off], parts[2 + c_off] = split(contour.edges[0])

				if len(contour.edges) >= 2 {
					parts[3 - c_off], parts[4 - c_off], parts[5 - c_off] = split(contour.edges[1])

					parts[0].color = colors[0]
					parts[1].color = colors[0]

					parts[2].color = colors[1]
					parts[3].color = colors[1]

					parts[4].color = colors[2]
					parts[5].color = colors[2]
				}
				else {
					parts[0].color = colors[0]
					parts[1].color = colors[1]
					parts[2].color = colors[2]
				}

				delete(contour.edges)
				contour.edges = make([dynamic]Edge_Segment, 0, 7)
				append(&contour.edges, ..parts[:])
			}
		}
		else { /* Multiple corners */
			spline := 0
			start := corners[0]
			corner_count := len(corners)
			m := len(contour.edges)

			color := Edge_Color.White
			switch_color(&color, &seed, .Black)
			initial_color := color

			for edge_idx in 0..<m {
				index := (start + edge_idx) % m
				if spline + 1 < corner_count && corners[spline + 1] == index {
					spline += 1
					switch_color(&color, &seed, (spline == corner_count - 1) ? initial_color : .Black)
				}
				contour.edges[index].color = color
			}
		}
	}
}

INF :: -1e24

// void msdf_distToPseudo(msdf_signedDistance *distance, msdf_Vec2 origin, float param, msdf_EdgeSegment *e) {

distance_to_pseudo :: proc(dist: Signed_Distance, origin: Vec2, param: f32, e: Edge_Segment) -> Signed_Distance {
	if param < 0 {
		dir := lin.normalize(direction(e, 0))
		aq := origin - point(e, 0)
		ts := lin.inner_product(aq, dir)

		if ts < 0 {
			pseudo_dist := lin.cross(aq, dir)
			if abs(pseudo_dist) < abs(dist.distance){
				return { pseudo_dist, 0 }
			}
		}
	}
	else if param > 1 {
		dir := lin.normalize(direction(e, 1))
		bq := origin - point(e, 1)

		ts := lin.inner_product(bq, dir)
		if ts > 0 {
			pseudo_dist := lin.cross(bq, dir)
			if abs(pseudo_dist) >= abs(dist.distance){
				return { pseudo_dist, 0 }
			}
		}
	}
	return dist
}

// For each position < n, this function will return -1, 0, or 1, depending on
// whether the position is closer to the beginning, middle, or end,
// respectively. It is guaranteed that the output will be balanced in that the
// total for positions 0 through n-1 will be zero.
symmetrical_trichotomy :: proc(pos, n: int) -> int {
	return int(3 + 2.875 * f32(pos) / (f32(n - 1) - 1.4375 + 0.5)) - 3
}

non_zero_sign :: proc(n: f32) -> f32 {
	return 2 * (n > 0 ? 1 : 0) - 1
}

Multi_Distance :: struct {
	r, g, b: f32,
	med: f32,
}

