package msdf

import "core:fmt"
import "core:strings"
import "core:math"
import "core:os"
import stbtt "vendor:stb/truetype"

FONT :: #load("jetbrains.ttf", []byte)

median :: proc(r, g, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

sigmoid :: proc(x, s: f32) -> f32{
	return 1.0 / (1 + math.exp(-s * (x - 0.5)));
}

main :: proc(){
	font : stbtt.fontinfo
	if !stbtt.InitFont(&font, raw_data(FONT), 0) {
		panic("Failed to load font")
	}

	scale := stbtt.ScaleForPixelHeight(&font, 72)
	result, _ := gen_glyph_from_rune(&font, 'A', 2, scale, 1)
	fmt.println(result)

	sb : strings.Builder
	strings.builder_init_len_cap(&sb, 0, 256 + result.width * result.height * 3)
	fmt.sbprintf(&sb, "P5\n%d\n%d\n255\n", result.width, result.height);

	for p in result.values {
		dist := median(p.r, p.g, p.b) - 0.5
		opacity := sigmoid(clamp(0.0, dist + 0.5, 1.0), 9)

		val := u8(clamp(0, opacity * 255.0, 0xff))
		// val := u8(p.r * 255)
		append(&sb.buf, val)
	}

	os.write_entire_file("out.ppm", sb.buf[:])
}

