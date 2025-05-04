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



