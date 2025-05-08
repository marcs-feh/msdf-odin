package msdf

import "base:runtime"
import "core:sync"
import "core:slice"
import "core:mem"
import "core:c"
import stbtt "vendor:stb/truetype"

// clang -Os -fno-strict-aliasing -Wall -Wextra -fPIC -c msdf_c.c -o msdf_c.o
foreign import msdf_c "libmsdf.o"

ARENA_SIZE :: #config(MSDF_ARENA_SIZE_KB, 4096) * mem.Kilobyte

@(link_prefix="msdf_")
foreign msdf_c {
	genGlyph :: proc(
		result: ^msdf_Result,
		font: ^stbtt.fontinfo,
		stbttGlyphIndex: c.int,
		borderWidth: i32,
		scale: f32,
		range: f32,
		alloc: ^msdf_AllocCtx,
	) -> c.int ---
}

msdf_Result :: struct {
	glyph_index: c.int,
	left_bearing: c.int,
	advance: c.int,
	rgb: [^]f32,
	width: c.int,
	height: c.int,
	y_offset: c.int,
}

msdf_AllocCtx :: struct {
	alloc: proc "c" (size: c.size_t, ctx: rawptr) -> rawptr,
	free: proc "c" (ptr: rawptr, ctx: rawptr),
	ctx: rawptr,
}

Field :: struct {
	glyph_index: int,
	left_bearing: int,
	advance: int,
	values: [][3]f32 `fmt:"p"`,
	width: int,
	height: int,
	y_offset: int,
}

_internal_arena : mem.Arena

_arena_mutex : sync.Mutex

_alloc_ctx : msdf_AllocCtx

@(init)
_initialize :: proc(){
	@static arena_memory : [ARENA_SIZE]u8
	mem.arena_init(&_internal_arena, arena_memory[:])

	_alloc_ctx.alloc = proc "c" (size: c.size_t, _: rawptr) -> rawptr {
		context = runtime.Context {}

		ptr, _ := mem.arena_alloc(&_internal_arena, int(size))
		return ptr
	}

	_alloc_ctx.free = proc "c" (_, _: rawptr){}

	_alloc_ctx.ctx = nil
}

Error :: enum {
	None = 0,
	Memory_Error,
	Glyph_Error,
}

gen_glyph :: proc {
	gen_glyph_from_rune,
	gen_glyph_from_index,
}

gen_glyph_from_rune :: proc(
	font: ^stbtt.fontinfo,
	codepoint: rune,
	border_width: int,
	scale: f32,
	range: f32,
	allocator := context.allocator,
) -> (result: Field, err: Error)
{
	index := int(stbtt.FindGlyphIndex(font, codepoint))
	return gen_glyph_from_index(font, index, border_width, scale, range, allocator)
}

import "core:fmt"
gen_glyph_from_index :: proc(
	font: ^stbtt.fontinfo,
	glyph_index: int,
	border_width: int,
	scale: f32,
	range: f32,
	allocator := context.allocator,
) -> (result: Field, err: Error)
{

	if glyph_index <= 0 {
		err = .Glyph_Error
		return
	}

	sync.lock(&_arena_mutex)
	defer sync.unlock(&_arena_mutex)

	field_res : msdf_Result
	status := genGlyph(&field_res, font, c.int(glyph_index), i32(border_width), scale, range, &_alloc_ctx)
	defer mem.arena_free_all(&_internal_arena)

	if status == 0 {
		err = .Glyph_Error
		return
	}

	pixels, mem_err := make([][3]f32, field_res.width * field_res.height, allocator)
	if mem_err != nil {
		err = .Memory_Error
		return
	}

	original_pixels := field_res.rgb[:field_res.width * field_res.height * 3]
	mem.copy_non_overlapping(raw_data(pixels), raw_data(original_pixels), slice.size(original_pixels))

	result = Field {
		glyph_index  = glyph_index,
		left_bearing = int(field_res.left_bearing),
		advance      = int(field_res.advance),
		values       = pixels,
		width        = int(field_res.width),
		height       = int(field_res.height),
		y_offset     = int(field_res.y_offset),
	}

	return
}

