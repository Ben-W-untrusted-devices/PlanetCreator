extends SceneTree
## Render scenes/descent.tscn from a given position and save a PNG.
## Usage: godot --path godot -s scripts/capture_descent.gd -- out.png cube_dir lat lon alt_m yaw pitch [frames]

var _frames := 0
var _wait := 120
var _out := "descent.png"


func _initialize() -> void:
	var a := OS.get_cmdline_user_args()
	_out = a[0]
	var scene: Node = load("res://scenes/descent.tscn").instantiate()
	var planet: Node = scene.get_node("Planet")
	planet.cube_dir = a[1]
	planet.start_lat = float(a[2])
	planet.start_lon = float(a[3])
	planet.start_altitude_m = float(a[4])
	var cam: Node = scene.get_node("Camera")
	cam.yaw = float(a[5])
	cam.pitch = float(a[6])
	if a.size() > 7:
		_wait = int(a[7])
	scene.get_node("HUD").visible = false
	root.add_child(scene)


func _process(_delta: float) -> bool:
	_frames += 1
	if _frames == _wait:
		var img := root.get_viewport().get_texture().get_image()
		img.save_png(_out)
		var planet: Node = root.get_node("Descent/Planet")
		print("saved ", _out, " alt ", planet.get_altitude(), " patches ", planet.get_patch_count(), " fps ", Engine.get_frames_per_second())
		return true
	return false
