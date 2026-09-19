extends Label

@export var planet_path: NodePath = ^"../../Planet"
@export var camera_path: NodePath = ^"../../Camera"
var _planet: Node
var _camera: Node


func _ready() -> void:
	_planet = get_node(planet_path)
	_camera = get_node(camera_path)


func _process(_delta: float) -> void:
	var alt: float = _planet.get_altitude()
	var ll: Vector2 = _planet.get_lat_lon()
	var alt_s := "%.1f m" % alt if alt < 10000.0 else "%.1f km" % (alt / 1000.0)
	text = "alt %s   lat %.3f  lon %.3f   ground %.0f m   patches %d   fps %d\n" % [
		alt_s, ll.x, ll.y, _planet.get_ground_height(), _planet.get_patch_count(), Engine.get_frames_per_second()
	] + "right-drag / Tab: look   WASD: move   Q/E: down/up   Shift: x10   Ctrl: /10   Esc: release mouse"
