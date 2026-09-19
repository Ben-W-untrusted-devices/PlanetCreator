extends Camera3D
## Free-fly camera for the descent scene. The camera never leaves the engine origin:
## it only rotates, and movement is handed to PlanetLod, which keeps the true position
## in double precision and moves the world around us.
##
## Controls: right-drag (or Tab to toggle capture) to look, WASD to move, Q/E down/up,
## Shift x10, Ctrl /10, F to face the planet. Speed scales with altitude. Escape
## releases the mouse.

@export var planet_path: NodePath = ^"../Planet"
@export var mouse_sensitivity := 0.003
@export var speed_per_altitude := 0.6   # m/s per metre of altitude
@export var min_speed := 1.5
@export var min_altitude := 1.8         # eye height when walking

var yaw := 0.0
var pitch := -0.2
var _captured := false
var _planet: Node


func _ready() -> void:
	_planet = get_node(planet_path)
	position = Vector3.ZERO
	face_planet()
	_update_clip()


## Aim at the planet: straight down when high up (the planet is below the horizon from
## orbit), level when near the ground.
func face_planet() -> void:
	var altitude: float = _planet.get_altitude()
	pitch = -1.5 if altitude > 50000.0 else -0.15


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT:
		_set_capture(event.pressed)
	elif event is InputEventKey and event.pressed:
		if event.keycode == KEY_TAB:
			_set_capture(not _captured)
		elif event.keycode == KEY_ESCAPE:
			_set_capture(false)
		elif event.keycode == KEY_F:
			face_planet()
	elif event is InputEventMouseMotion and _captured:
		yaw -= event.relative.x * mouse_sensitivity
		pitch = clampf(pitch - event.relative.y * mouse_sensitivity, -1.55, 1.55)


func _set_capture(on: bool) -> void:
	_captured = on
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if on else Input.MOUSE_MODE_VISIBLE


func _process(delta: float) -> void:
	var up: Vector3 = _planet.get_up()
	var ref := Vector3.UP if absf(up.dot(Vector3.UP)) < 0.98 else Vector3.RIGHT
	var east := up.cross(ref).normalized()
	var north := east.cross(up).normalized()
	var forward := (north * cos(yaw) + east * sin(yaw)) * cos(pitch) + up * sin(pitch)
	transform.basis = Basis.looking_at(forward, up)

	var altitude: float = _planet.get_altitude()
	var speed := maxf(min_speed, altitude * speed_per_altitude)
	if Input.is_key_pressed(KEY_SHIFT):
		speed *= 10.0
	if Input.is_key_pressed(KEY_CTRL):
		speed *= 0.1
	var move := Vector3.ZERO
	var flat_forward := (north * cos(yaw) + east * sin(yaw)).normalized()
	var right := flat_forward.cross(up).normalized()
	if Input.is_key_pressed(KEY_W):
		move += forward if altitude > 50.0 else flat_forward
	if Input.is_key_pressed(KEY_S):
		move -= forward if altitude > 50.0 else flat_forward
	if Input.is_key_pressed(KEY_D):
		move += right
	if Input.is_key_pressed(KEY_A):
		move -= right
	if Input.is_key_pressed(KEY_E):
		move += up
	if Input.is_key_pressed(KEY_Q):
		move -= up
	if move.length() > 0.0:
		_planet.move_camera(move.normalized() * speed * delta)
	# Keep the eye above the ground.
	var alt_after: float = _planet.get_altitude()
	if alt_after < min_altitude:
		_planet.move_camera(up * (min_altitude - alt_after))
	_update_clip()


func _update_clip() -> void:
	# Keep far/near within what the renderer's culling can handle (~1e6-1e7): from the
	# ground we still need ~300 km for distant ranges, from orbit the whole planet.
	var altitude: float = _planet.get_altitude()
	near = clampf(altitude * 0.002, 0.05, 1000.0)
	far = clampf(altitude * 30.0 + 300000.0, 300000.0, 1.0e8)
