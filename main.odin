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
    return max(min(r, g), min(max(r, g), b));
}

sigmoid :: proc(x, s: f32) -> f32{
	return 1.0 / (1 + math.exp(-s * (x - 0.5)));
}

load_ascii :: proc(font: ^stbtt.fontinfo, pixel_height: int) -> map[rune]Field {
	chars := make(map[rune]Field)

	scale := stbtt.ScaleForPixelHeight(font, auto_cast pixel_height)

	for r in 0..=rune(0x7f) {
		field, err := gen_glyph_from_rune(font, r, 4, scale, 1.0)
		if err == nil {
			chars[r] = field
		}
	}

	return chars
}

main :: proc(){
	font : stbtt.fontinfo
	if !stbtt.InitFont(&font, raw_data(FONT), 0) {
		panic("Failed to load font")
	}

	scale := stbtt.ScaleForPixelHeight(&font, 24)

	begin := time.now()
	chars := load_ascii(&font, 256)
	fmt.println("Elapsed:", time.since(begin))

	result, _ := gen_glyph_from_rune(&font, 'g', 2, scale, 1.0)
	fmt.println(result)

	sb : strings.Builder
	strings.builder_init_len_cap(&sb, 0, 256 + result.width * result.height * 3)
	fmt.sbprintf(&sb, "P5\n%d\n%d\n255\n", result.width, result.height);


	for p in result.values {
		dist := median(p.r, p.g, p.b) - 0.5
		d := clamp(0.0, dist + 0.5, 1.0)

		opacity := sigmoid(d, 8)

		val := u8(clamp(0, opacity * 255.0, 0xff))
		append(&sb.buf, val)
	}
	fmt.printfln("Peak: %.2fKiB", f64(_internal_arena.peak_used) / f64(mem.Kilobyte))

	os.write_entire_file("out.ppm", sb.buf[:])
}

