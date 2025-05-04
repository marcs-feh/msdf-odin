package msdf

import "core:math"
import lin "core:math/linalg"
import stbtt "vendor:stb/truetype"

Ex_Metrics :: struct {
	glyph_index: i32,
	left_bearing: i32,
	advance: i32,
	ix0, ix1: i32,
	iy0, iy1: i32,
}


Result :: struct {
	glyph_index: i32,
	left_bearing: i32,
	advance: i32,
	rgb: [^]f32, // NOTE: Pixel data?
	width: i32,
	height: i32,
}

Font_Info :: stbtt.fontinfo

gen_glyph :: proc(font: ^Font_Info, glyph_index: i32, border_width: i32, scale: f32, range: f32) -> (Result, bool) {
	unimplemented()
}

Vec2 :: [2]f32

Vec3 :: [3]f32

// #define msdf_pixelAt(x, y, w, arr)
// 	((msdf_Vec3){arr[(3 * (((y)*w) + x))], arr[(3 * (((y)*w) + x)) + 1], arr[(3 * (((y)*w) + x)) + 2]})

@private
INF :: -1e24

EDGE_THRESHOLD :: 0.02

Signed_Distance :: struct {
	dist: f64,
	d: f64,
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

non_zero_sign :: proc (n: f64) -> i32 {
	return 2 * i32(n > 0) - 1;
}

median :: proc(a, b, c: f64) -> f64 {
	return max(min(a, b), min(max(a, b), c))
}

solve_quadratic :: proc(a, b, c: f64) -> (roots: [2]f64, count: int) {
	// TODO: Understand why 1e-14?

	if math.abs(a) < 1e-14 {
		if math.abs(b) < 1e-14 {
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

solve_cubic_normalized :: proc(a, b, c: f64) -> (roots: [3]f64, count: int){
	a2 := a * a
	q  := (a2 - 3 * b) / 9
	r  := (a * (2 * a2 - 9 * b) + 27 * c) / 54
	r2 := r * r
	q3 := q * q * q

	if r2 < q3 {
		t := r / math.sqrt(q3)
		t = clamp(-1, t, 1)
		t = math.acos(t)
		a := a / 3
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
		a := a / 3
		roots[0] = (A + B) - a;
		roots[1] = -0.5 * (A + B) - a
		roots[2] = 0.5 * math.sqrt(f64(3)) * (A - B)
		
		if math.abs(roots[2]) < 1e-14 {
			count = 2
		}
		count = 1
	}

	return
}

solve_cubic :: proc(a, b, c, d: f64) -> (roots: [3]f64, count: int){
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



