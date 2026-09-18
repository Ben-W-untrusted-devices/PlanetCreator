extends Camera3D
## Drag to orbit, scroll to zoom. Placeholder until the real space->surface camera exists.

@export var target := Vector3.ZERO
@export var distance := 3.0
@export var min_distance := 1.05
@export var max_distance := 20.0

var _yaw := 0.6
var _pitch := 0.3
var _dragging := false


func _ready() -> void:
	_update()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		match event.button_index:
			MOUSE_BUTTON_LEFT:
				_dragging = event.pressed
			MOUSE_BUTTON_WHEEL_UP:
				distance = maxf(min_distance, distance * 0.9)
			MOUSE_BUTTON_WHEEL_DOWN:
				distance = minf(max_distance, distance / 0.9)
		_update()
	elif event is InputEventMouseMotion and _dragging:
		_yaw -= event.relative.x * 0.005
		_pitch = clampf(_pitch + event.relative.y * 0.005, -1.5, 1.5)
		_update()


func _update() -> void:
	var dir := Vector3(cos(_pitch) * sin(_yaw), sin(_pitch), cos(_pitch) * cos(_yaw))
	position = target + dir * distance
	look_at(target, Vector3.UP)
