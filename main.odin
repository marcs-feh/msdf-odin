package msdf

import "core:fmt"
import "core:mem"
import "core:time"
import "core:strings"
import "core:math"
import "core:os"
import stbtt "vendor:stb/truetype"

FONT :: #load("jetbrains.ttf", []byte)

median :: proc(r, g, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b))
}

sigmoid :: proc(x, s: f32) -> f32{
	return 1.0 / (1 + math.exp(-s * (x - 0.5)))
}

// load_ascii :: proc(font: ^stbtt.fontinfo, pixel_height: int) -> map[rune]Field {
// 	chars := make(map[rune]Field)
//
// 	scale := stbtt.ScaleForPixelHeight(font, auto_cast pixel_height)
//
// 	for r in 0..=rune(0x7f) {
// 		field, err := gen_glyph_from_rune(font, r, 4, scale, 1.0)
// 		if err == nil {
// 			chars[r] = field
// 		}
// 	}
//
// 	return chars
// }

foreign import msdf_c "libmsdf.o"
import "core:c"

msdfc_Result :: struct {
	glyphIdx: c.int,
	left_bearing: c.int,
	advance: c.int,
	rgb: [^]f32,
	width: c.int,
	height: c.int,
	yOffset: c.int,
}

foreign msdf_c {
	msdf_genGlyph :: proc(
		result: ^msdfc_Result,
		font: ^stbtt.fontinfo,
		index: c.int,
		border: i32,
		scale: f32,
		range: f32,
		ctx: rawptr,
	) -> c.int ---
}

res_conv :: proc(res: msdfc_Result) -> Result {
	return Result {
		glyph_index = res.glyphIdx,
		left_bearing = res.left_bearing,
		y_offset = res.yOffset,
		advance = res.advance,
		values = (transmute([^][3]f32)res.rgb)[:res.width * res.height],
		width = res.width,
		height = res.height,
	}
}

main :: proc(){
	font : stbtt.fontinfo
	if !stbtt.InitFont(&font, raw_data(FONT), 0) {
		panic("Failed to load font")
	}

	if os.args[1] == "port" {
		test_ppm(&font, "port.ppm", true)
	}
	else {
		test_ppm(&font, "original.ppm", false)
	}
}

test_ppm :: proc(font: ^stbtt.fontinfo, output: string, use_port: bool){
	scale := stbtt.ScaleForPixelHeight(font, 20)
	index := stbtt.FindGlyphIndex(font, 'Á')

	result : Result
	if use_port {
		result = gen_glyph(font, index, 2, scale, 0.5)
	}
	else {
		result_ : msdfc_Result
		msdf_genGlyph(&result_, font, index, 2, scale, 0.5, nil)
		result = res_conv(result_)
	}

	sb : strings.Builder
	strings.builder_init_len_cap(&sb, 0, 256 + int(result.width * result.height * 3))
	defer strings.builder_destroy(&sb)

	fmt.sbprintf(&sb, "P5\n%d\n%d\n255\n", result.width, result.height);

	for p in result.values {
		dist := median(p.r, p.g, p.b) - 0.5
		d := clamp(0.0, dist + 0.5, 1.0)

		opacity := sigmoid(d, 8)

		val := u8(clamp(0, opacity * 255.0, 0xff))
		append(&sb.buf, val)
	}
	// fmt.printfln("Peak: %.2fKiB", f64(_internal_arena.peak_used) / f64(mem.Kilobyte))

	os.write_entire_file(output, sb.buf[:])
}


