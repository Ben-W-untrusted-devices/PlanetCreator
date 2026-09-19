extends MeshInstance3D
## Feeds the atmosphere post-process shader with the camera-relative planet centre and
## the sun direction every frame.

@export var planet_path: NodePath = ^"../Planet"
@export var sun_path: NodePath = ^"../Sun"
@export var atmosphere_height := 100000.0
## Sun elevation above the horizon at the start position, degrees (the sun is fixed in
## space; this just chooses where noon is).
@export var sun_elevation_deg := 40.0
@export var sun_azimuth_deg := 60.0
@export var debug_mode := 0

var _planet: Node
var _sun: DirectionalLight3D


func _ready() -> void:
	_planet = get_node(planet_path)
	_sun = get_node(sun_path)
	extra_cull_margin = 1.0e9
	_place_sun()
	var mat := material_override as ShaderMaterial
	mat.set_shader_parameter("planet_radius", _planet.radius_m)
	mat.set_shader_parameter("atmosphere_height", atmosphere_height)
	mat.set_shader_parameter("debug_mode", debug_mode)


func _place_sun() -> void:
	var up: Vector3 = _planet.get_up()
	var ref := Vector3.UP if absf(up.dot(Vector3.UP)) < 0.98 else Vector3.RIGHT
	var east := up.cross(ref).normalized()
	var north := east.cross(up).normalized()
	var el := deg_to_rad(sun_elevation_deg)
	var az := deg_to_rad(sun_azimuth_deg)
	var to_sun := (up * sin(el) + (north * cos(az) + east * sin(az)) * cos(el)).normalized()
	_sun.look_at(-to_sun, ref)  # light travels away from the sun


func _process(_delta: float) -> void:
	var mat := material_override as ShaderMaterial
	mat.set_shader_parameter("planet_center", _planet.get_planet_center())
	mat.set_shader_parameter("sun_dir", _sun.global_transform.basis.z)  # toward the sun
