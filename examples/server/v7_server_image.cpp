// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_image.cpp -- image decoding: base64 bytes -> raw 24bpp RGB.
 *
 * Windows decodes through GDI+ (PNG/JPEG/BMP/GIF/TIFF, EXIF orientation,
 * alpha flattened onto white). Linux decodes PNG/JPEG/BMP/GIF through the
 * vendored single-header stb_image library (Windows also handles TIFF),
 * with the same output contract: no downscale, a 64-megapixel decode cap,
 * EXIF orientation applied, alpha composited onto a white canvas.
 */

#include "v7_server_common.h"

#ifdef _WIN32

/* ========================================================================
 * GDI+ image decoding — base64 bytes -> raw 24bpp RGB (top-down rows)
 * (decoded_image lives in v7_server_common.h)
 * ======================================================================== */

/* Decode an encoded image (PNG/JPEG/BMP/GIF/TIFF — whatever GDI+ knows)
 * from a memory buffer into raw RGB. Returns 0 on success. */
int image_decode(unsigned char *data, int data_len, decoded_image *out)
{
	memset(out, 0, sizeof(*out));

	HGLOBAL hglob = GlobalAlloc(GMEM_MOVEABLE, (SIZE_T)data_len);
	if (!hglob)
		return -1;
	void *mem = GlobalLock(hglob);
	if (!mem)
	{
		GlobalFree(hglob);
		return -1;
	}
	memcpy(mem, data, (int)data_len);
	GlobalUnlock(hglob);

	IStream *stream = 0;
	if (CreateStreamOnHGlobal(hglob, TRUE, &stream) != S_OK)
	{
		GlobalFree(hglob);
		return -1;
	}

	/* Keep the stream alive for the bitmap's whole lifetime: GDI+ decodes
	 * LAZILY (e.g. during LockBits), reading from the IStream. Releasing it
	 * early frees hglob (fDeleteOnRelease=TRUE) -> use-after-free in GDI+. */
	Gdiplus::Bitmap *bmp = Gdiplus::Bitmap::FromStream(stream);
	if (!bmp || bmp->GetLastStatus() != Gdiplus::Ok)
	{
		delete bmp;
		stream->Release();
		return -1;
	}

	UINT w = bmp->GetWidth();
	UINT h = bmp->GetHeight();
	if (w < 1 || h < 1)
	{
		delete bmp;
		stream->Release();
		return -1;
	}

	/* Metadata pixel cap, checked BEFORE RotateFlip: the orientation
	 * transform forces the FULL frame decode, so a tiny JPEG declaring
	 * huge dimensions plus an EXIF orientation tag would otherwise
	 * allocate gigabytes before the draw-time check runs. Rotation only
	 * swaps w/h, the product is unchanged, so the cap is valid here.
	 * 64 M pixels covers every real phone photo and rejects the
	 * amplification vector. */
	if ((unsigned long long)w * (unsigned long long)h > 64ull * 1024 * 1024)
	{
		delete bmp;
		stream->Release();
		return -1;
	}

	/* EXIF orientation: phone JPEGs carry a rotate flag instead of pre-
	 * rotated pixels. Normalize with RotateFlip so the model sees the
	 * image upright (PropertyTagOrientation values 1-8; 1 = none). */
	UINT ori = 1;
	UINT ori_size = bmp->GetPropertyItemSize(PropertyTagOrientation);
	if (ori_size > 0)
	{
		Gdiplus::PropertyItem *pi = (Gdiplus::PropertyItem *)malloc(ori_size);
		if (pi && bmp->GetPropertyItem(PropertyTagOrientation, ori_size, pi) == Gdiplus::Ok && pi->value && pi->length >= 2)
		{
			ori = (UINT)*(unsigned short *)pi->value;
			/* EXIF shorts follow the FILE's byte order ("II" little / "MM"
			 * big): valid orientations are 1-8, so anything above 8 is the
			 * wrong byte order -- swap the halves. */
			if (ori > 8)
				ori = ((ori & 0xFF) << 8) | ((ori >> 8) & 0xFF);
			if (ori > 8)
				ori = 1; /* neither order is valid -- leave the image untouched */
		}
		free(pi);
	}
	switch (ori)
	{
	case 2: bmp->RotateFlip(Gdiplus::RotateNoneFlipX); break;
	case 3: bmp->RotateFlip(Gdiplus::Rotate180FlipNone); break;
	case 4: bmp->RotateFlip(Gdiplus::Rotate180FlipX); break;
	case 5: bmp->RotateFlip(Gdiplus::Rotate90FlipX); break;
	case 6: bmp->RotateFlip(Gdiplus::Rotate90FlipNone); break;
	case 7: bmp->RotateFlip(Gdiplus::Rotate270FlipX); break;
	case 8: bmp->RotateFlip(Gdiplus::Rotate270FlipNone); break;
	default: break; /* 1 (or unknown): already upright */
	}
	/* orientations 5-8 SWAP width/height — re-read AFTER the rotation or
	 * the flat canvas squashes the image and the model sees wrong dims. */
	w = bmp->GetWidth();
	h = bmp->GetHeight();

	/* NO downscale: the model sees the image at its NATIVE size (the old
	 * 4096-px clamp silently resampled large scans and destroyed their
	 * text legibility). The pixel CAP against decode amplification already
	 * ran before RotateFlip -- that one is the legitimate crash guard. */

	/* Alpha composite: a transparent PNG decoded straight into 24bpp gives
	 * black/garbage background pixels. Draw the image onto a WHITE 24bpp
	 * bitmap instead — the alpha channel is flattened the way viewers do.
	 * The TileFlipXY wrap mode mirrors edge pixels instead of blending them
	 * with the canvas -- the default produces the known faint bright border
	 * on all four edges of the scaled image. */
	Gdiplus::Bitmap flat((INT)w, (INT)h, PixelFormat24bppRGB);
	Gdiplus::Graphics gfx(&flat);
	gfx.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic); /* the default nearest-neighbour aliases downscales -- the model sees degraded edges */
	gfx.Clear(Gdiplus::Color(255, 255, 255, 255));
	Gdiplus::ImageAttributes ia;
	ia.SetWrapMode(Gdiplus::WrapModeTileFlipXY);
	UINT src_w = bmp->GetWidth();
	UINT src_h = bmp->GetHeight();
	if (gfx.DrawImage(bmp, Gdiplus::Rect(0, 0, (INT)w, (INT)h), 0, 0, (Gdiplus::REAL)src_w, (Gdiplus::REAL)src_h, Gdiplus::UnitPixel, &ia) != Gdiplus::Ok)
	{
		delete bmp;
		stream->Release();
		return -1;
	}
	/* All rendering queued into `flat` completes NOW: LockBits below must
	 * never race the still-alive Graphics object's internal batching. */
	gfx.Flush(Gdiplus::FlushIntentionSync);
	delete bmp;
	stream->Release(); /* only now is the underlying HGLOBAL freed */

	unsigned char *rgb = (unsigned char *)malloc((size_t)w * h * 3);
	if (!rgb)
		return -1;

	Gdiplus::Rect rect(0, 0, (INT)w, (INT)h);
	Gdiplus::BitmapData bd;
	if (flat.LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat24bppRGB, &bd) != Gdiplus::Ok)
	{
		free(rgb);
		return -1;
	}
	/* GDI+ 24bpp memory order is B,G,R — the engine expects R,G,B (the same
	 * layout as dbg_image_recog's .raw input). Swap the channels in flight.
	 * Stride is ALWAYS positive: `flat` is a freshly constructed top-down
	 * Bitmap (GDI+ default), not a bottom-up DIB -- a negative stride never
	 * reaches this loop. */
	unsigned char *src_start = (unsigned char *)bd.Scan0;
	for (UINT y = 0; y < h; y++)
	{
		unsigned char *src_row = src_start + (size_t)y * bd.Stride;
		unsigned char *dst_row = rgb + (size_t)y * w * 3;
		for (UINT x = 0; x < w; x++)
		{
			dst_row[x * 3 + 0] = src_row[x * 3 + 2]; /* R */
			dst_row[x * 3 + 1] = src_row[x * 3 + 1]; /* G */
			dst_row[x * 3 + 2] = src_row[x * 3 + 0]; /* B */
		}
	}
	flat.UnlockBits(&bd);

	out->rgb = rgb;
	out->width = (int)w;
	out->height = (int)h;
	return 0;
}

#else /* Linux */

/* ========================================================================
 * stb_image decoding (the vendored single-header library in src/linux/):
 * PNG/JPEG/BMP/GIF in, RGBA out (4 channels forced so the alpha channel is
 * always present for flattening), then the shared orientation step. The
 * library is public domain and decoded results match the GDI+ contract.
 * ======================================================================== */

#include <stdint.h>

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_STDIO
#define STBI_ASSERT(x)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* 64 M pixels: the same decode-amplification cap the Windows path applies
 * before any pixel work. */
#define IMAGE_MAX_PIXELS (64ull * 1024 * 1024)

/* read a big-endian unsigned value of the given byte width */
static unsigned img_be(const unsigned char *p, int bytes)
{
	unsigned value = 0;
	for (int i = 0; i < bytes; i++)
		value = (value << 8) | p[i];
	return value;
}

/* read a little-endian unsigned value of the given byte width */
static unsigned img_le(const unsigned char *p, int bytes)
{
	unsigned value = 0;
	for (int i = bytes - 1; i >= 0; i--)
		value = (value << 8) | p[i];
	return value;
}

/* flatten one RGBA pixel onto white (the Windows path clears a white canvas
 * before drawing) */
static void img_flatten(unsigned char *rgb, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	if (a == 255)
	{
		rgb[0] = r;
		rgb[1] = g;
		rgb[2] = b;
		return;
	}
	if (a == 0)
	{
		rgb[0] = 255;
		rgb[1] = 255;
		rgb[2] = 255;
		return;
	}
	unsigned inv = 255 - a;
	rgb[0] = (unsigned char)((r * a + 255 * inv) / 255);
	rgb[1] = (unsigned char)((g * a + 255 * inv) / 255);
	rgb[2] = (unsigned char)((b * a + 255 * inv) / 255);
}

/* apply the EXIF orientation to a finished RGB image by producing a rotated
 * copy; orientations 5-8 swap width/height. Returns a fresh buffer (the
 * caller frees BOTH) or 0 when no transform is needed. */
static unsigned char *img_apply_orientation(unsigned char *src, int *width_ptr, int *height_ptr, unsigned orientation)
{
	int w = *width_ptr;
	int h = *height_ptr;
	if (orientation <= 1 || orientation > 8)
		return 0;

	int swap = (orientation >= 5);
	int dst_w = swap ? h : w;
	int dst_h = swap ? w : h;
	unsigned char *dst = (unsigned char *)malloc((size_t)dst_w * dst_h * 3);
	if (!dst)
		return 0;

	for (int y = 0; y < dst_h; y++)
	{
		for (int x = 0; x < dst_w; x++)
		{
			int sx = 0;
			int sy = 0;
			switch (orientation)
			{
			case 2: sx = w - 1 - x; sy = y; break;
			case 3: sx = w - 1 - x; sy = h - 1 - y; break;
			case 4: sx = x; sy = h - 1 - y; break;
			case 5: sx = y; sy = x; break;
			case 6: sx = y; sy = h - 1 - x; break;
			case 7: sx = h - 1 - y; sy = w - 1 - x; break;
			case 8: sx = h - 1 - y; sy = x; break;
			}
			const unsigned char *src_pixel = src + ((size_t)sy * w + sx) * 3;
			unsigned char *dst_pixel = dst + ((size_t)y * dst_w + x) * 3;
			dst_pixel[0] = src_pixel[0];
			dst_pixel[1] = src_pixel[1];
			dst_pixel[2] = src_pixel[2];
		}
	}
	*width_ptr = dst_w;
	*height_ptr = dst_h;
	return dst;
}

/* read the EXIF orientation out of an APP1 segment body, or 1 */
static unsigned img_exif_orientation(const unsigned char *body, unsigned body_len)
{
	if (body_len < 14 || memcmp(body, "Exif\0\0", 6) != 0)
		return 1;
	const unsigned char *tiff = body + 6;
	size_t tiff_len = body_len - 6;
	int little = 0;
	if (tiff[0] == 'I' && tiff[1] == 'I')
		little = 1;
	else if (!(tiff[0] == 'M' && tiff[1] == 'M'))
		return 1;
	unsigned ifd_offset = little ? img_le(tiff + 4, 4) : img_be(tiff + 4, 4);
	if (ifd_offset + 2 > tiff_len)
		return 1;
	const unsigned char *ifd = tiff + ifd_offset;
	unsigned entries = little ? img_le(ifd, 2) : img_be(ifd, 2);
	for (unsigned i = 0; i < entries; i++)
	{
		size_t entry_off = ifd_offset + 2 + (size_t)i * 12;
		if (entry_off + 12 > tiff_len)
			return 1;
		const unsigned char *entry = tiff + entry_off;
		unsigned tag = little ? img_le(entry, 2) : img_be(entry, 2);
		if (tag != 0x0112)
			continue;
		unsigned value = little ? img_le(entry + 8, 2) : img_be(entry + 8, 2);
		if (value > 8)
			value = ((value & 0xFF) << 8) | ((value >> 8) & 0xFF);
		return (value >= 1 && value <= 8) ? value : 1;
	}
	return 1;
}

/* walk the JPEG marker chain for the first APP1 segment and read the
 * orientation from it; PNG/BMP/GIF carry none */
static unsigned img_file_orientation(const unsigned char *data, size_t len)
{
	size_t pos = 2;
	while (pos + 4 <= len)
	{
		if (data[pos] != 0xFF)
			return 1;
		unsigned char marker = data[pos + 1];
		if (marker == 0xFF)
		{
			pos++;
			continue;
		}
		if (marker == 0xE1)
		{
			unsigned seg_len = ((unsigned)data[pos + 2] << 8) | data[pos + 3];
			if (seg_len >= 2 && pos + 4 + seg_len - 2 <= len)
				return img_exif_orientation(data + pos + 4, seg_len - 2);
			return 1;
		}
		if (marker == 0xDA || marker == 0xD9)
			return 1; /* the scan started: no more metadata segments */
		if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7))
		{
			pos += 2;
			continue;
		}
		unsigned seg_len = ((unsigned)data[pos + 2] << 8) | data[pos + 3];
		if (seg_len < 2)
			return 1;
		pos += 2 + seg_len;
	}
	return 1;
}

/* Decode an encoded image (PNG/JPEG/BMP/GIF) from a memory buffer into raw
 * RGB. Returns 0 on success. */
int image_decode(unsigned char *data, int data_len, decoded_image *out)
{
	memset(out, 0, sizeof(*out));
	if (!data || data_len < 8)
		return -1;

	/* The pixel cap runs BEFORE the decode, from the header metadata, the
	 * way the Windows path checks before RotateFlip. */
	int width = 0;
	int height = 0;
	int channels_in_file = 0;
	if (!stbi_info_from_memory(data, data_len, &width, &height, &channels_in_file))
		return -1;
	if (width < 1 || height < 1 || (unsigned long long)width * (unsigned long long)height > IMAGE_MAX_PIXELS)
		return -1;

	/* 4 output channels: the alpha channel must be materialized for the
	 * white-flattening step, whatever the file carried. */
	unsigned char *pixels = stbi_load_from_memory(data, data_len, &width, &height, &channels_in_file, 4);
	if (!pixels)
		return -1;

	unsigned char *rgb = (unsigned char *)malloc((size_t)width * height * 3);
	if (!rgb)
	{
		stbi_image_free(pixels);
		return -1;
	}
	size_t pixel_count = (size_t)width * height;
	for (size_t i = 0; i < pixel_count; i++)
	{
		const unsigned char *src = pixels + i * 4;
		img_flatten(rgb + i * 3, src[0], src[1], src[2], src[3]);
	}
	stbi_image_free(pixels);

	/* EXIF orientation is metadata outside stb_image's scope */
	if (data[0] == 0xFF && data[1] == 0xD8)
	{
		unsigned orientation = img_file_orientation(data, (size_t)data_len);
		if (orientation > 1)
		{
			unsigned char *rotated = img_apply_orientation(rgb, &width, &height, orientation);
			if (rotated)
			{
				free(rgb);
				rgb = rotated;
			}
		}
	}

	out->rgb = rgb;
	out->width = width;
	out->height = height;
	return 0;
}

#endif /* _WIN32 / Linux */
