extends SceneTree
## Render scenes/main.tscn for a few frames and save the viewport to a PNG.
## Usage: godot --path godot -s scripts/capture.gd -- out.png [cube_dir] [yaw] [pitch]

var _frames := 0
var _out := "screenshot.png"


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() > 0:
		_out = args[0]
	var scene: Node = load("res://scenes/main.tscn").instantiate()
	if args.size() > 1:
		scene.get_node("Planet").cube_dir = args[1]
	if args.size() > 3:
		var cam: Node = scene.get_node("Camera")
		cam.set("_yaw", float(args[2]))
		cam.set("_pitch", float(args[3]))
	root.add_child(scene)


func _process(_delta: float) -> bool:
	_frames += 1
	if _frames == 30:
		var img := root.get_viewport().get_texture().get_image()
		img.save_png(_out)
		print("saved ", _out, " ", img.get_size())
		return true
	return false
