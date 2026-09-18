extends SceneTree
## Render scenes/main.tscn for a few frames and save the viewport to a PNG.
## Usage: godot --path godot -s scripts/capture.gd -- out.png

var _frames := 0
var _out := "screenshot.png"


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() > 0:
		_out = args[0]
	root.add_child(load("res://scenes/main.tscn").instantiate())


func _process(_delta: float) -> bool:
	_frames += 1
	if _frames == 30:
		var img := root.get_viewport().get_texture().get_image()
		img.save_png(_out)
		print("saved ", _out, " ", img.get_size())
		return true
	return false
