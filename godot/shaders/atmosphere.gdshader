// Single-scattering atmosphere as a full-screen post-process.
// The camera is at the origin (see PlanetLod); planet_center is camera-relative.
// For each pixel: ray from the camera, clipped to the atmosphere shell and to the
// terrain depth, integrate Rayleigh + Mie in-scattering with sun-ray optical depth.
shader_type spatial;
render_mode unshaded, cull_disabled, depth_test_disabled, depth_draw_never, fog_disabled;

uniform sampler2D screen_tex : hint_screen_texture, filter_linear;
uniform sampler2D depth_tex : hint_depth_texture, filter_nearest;

uniform vec3 planet_center = vec3(0.0, -6371000.0, 0.0);
uniform float planet_radius = 6371000.0;
uniform float atmosphere_height = 100000.0;
uniform vec3 sun_dir = vec3(0.0, 1.0, 0.0);
uniform float sun_intensity = 4.0; // aesthetic: physical would be ~pi x light energy
// Aerial perspective on terrain pixels is scaled down from the sky's strength: the
// physically consistent value buries the (very dark) satellite albedo in haze.
uniform float aerial_scale = 0.35;
uniform vec3 rayleigh_beta = vec3(5.8e-6, 13.5e-6, 33.1e-6); // per metre, Earth
uniform float rayleigh_height = 8000.0;
uniform float mie_beta = 2.1e-5;
uniform float mie_height = 1200.0;
uniform float mie_g = 0.76;
uniform int view_samples = 16;
uniform int sun_samples = 6;
uniform int debug_mode = 0; // 1: scene distance (km / 100), 2: transmittance, 3: raw depth

void vertex() {
	POSITION = vec4(VERTEX.xy, 1.0, 1.0);
}

// Ray/sphere: returns (t_enter, t_exit) or (-1, -1).
vec2 ray_sphere(vec3 o, vec3 d, vec3 c, float r) {
	vec3 oc = o - c;
	float b = dot(oc, d);
	float cc = dot(oc, oc) - r * r;
	float disc = b * b - cc;
	if (disc < 0.0) {
		return vec2(-1.0);
	}
	float s = sqrt(disc);
	return vec2(-b - s, -b + s);
}

vec2 densities(vec3 p) {
	float h = max(length(p - planet_center) - planet_radius, 0.0);
	return vec2(exp(-h / rayleigh_height), exp(-h / mie_height));
}

// Optical depth (rayleigh, mie) along a ray of length len from p in direction d.
vec2 optical_depth(vec3 p, vec3 d, float len, int n) {
	float ds = len / float(n);
	vec2 od = vec2(0.0);
	for (int i = 0; i < n; i++) {
		vec3 q = p + d * ((float(i) + 0.5) * ds);
		od += densities(q) * ds;
	}
	return od;
}

void fragment() {
	float depth = texture(depth_tex, SCREEN_UV).r;
	vec3 ndc = vec3(SCREEN_UV * 2.0 - 1.0, depth);
	vec4 view = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
	view.xyz /= view.w;
	vec3 world = (INV_VIEW_MATRIX * vec4(view.xyz, 1.0)).xyz; // camera-relative (camera at origin)
	float scene_dist = length(world);
	vec3 dir = world / max(scene_dist, 1e-6);
	// Reversed-Z: the far plane (sky) reads as depth 0.
	bool is_sky = depth <= 1e-7;

	vec3 color = texture(screen_tex, SCREEN_UV).rgb;
	vec3 o = vec3(0.0);
	float ra = planet_radius + atmosphere_height;
	vec2 hit = ray_sphere(o, dir, planet_center, ra);
	float t0 = max(hit.x, 0.0);
	float t1 = hit.y;
	if (!is_sky) {
		t1 = min(t1, scene_dist);
	}
	vec3 result = color;
	vec2 od_view = vec2(0.0);
	if (hit.y >= 0.0 && t1 > t0) {
	float len = t1 - t0;
	float ds = len / float(view_samples);
	float mu = dot(dir, sun_dir);
	float phase_r = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
	float g2 = mie_g * mie_g;
	float phase_m = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * mie_g * mu, 1.5));

	vec3 sum_r = vec3(0.0);
	vec3 sum_m = vec3(0.0);
	for (int i = 0; i < view_samples; i++) {
		vec3 p = o + dir * (t0 + (float(i) + 0.5) * ds);
		vec2 dens = densities(p) * ds;
		od_view += dens;
		// sun ray: skip if it hits the planet (shadowed by the ground)
		vec2 ground = ray_sphere(p, sun_dir, planet_center, planet_radius);
		if (ground.y > 0.0 && ground.x > 0.0) {
			continue;
		}
		vec2 exit = ray_sphere(p, sun_dir, planet_center, ra);
		vec2 od_sun = optical_depth(p, sun_dir, max(exit.y, 0.0), sun_samples);
		vec3 tau = rayleigh_beta * (od_view.x + od_sun.x) + vec3(mie_beta) * (od_view.y + od_sun.y);
		vec3 att = exp(-tau);
		sum_r += dens.x * att;
		sum_m += dens.y * att;
	}
	vec3 inscatter = sun_intensity * (rayleigh_beta * phase_r * sum_r + vec3(mie_beta) * phase_m * sum_m);
	vec3 transmittance = exp(-(rayleigh_beta * od_view.x + vec3(mie_beta) * od_view.y));
	if (!is_sky) {
		inscatter *= aerial_scale;
		transmittance = mix(vec3(1.0), transmittance, aerial_scale);
	}
	result = color * transmittance + inscatter;
	}
	if (debug_mode == 1) {
		result = vec3(fract(scene_dist / 1.0e5), is_sky ? 1.0 : 0.0, scene_dist / 1.0e7);
	} else if (debug_mode == 2) {
		result = exp(-(rayleigh_beta * od_view.x + vec3(mie_beta) * od_view.y));
	} else if (debug_mode == 3) {
		result = vec3(depth);
	} else if (debug_mode == 4) {
		result = color;
	}
	ALBEDO = result;
	ALPHA = 1.0;
}
