extends SceneTree
var _f := 0
func _initialize() -> void:
	root.add_child(load("res://scenes/descent.tscn").instantiate())
func _process(_d: float) -> bool:
	_f += 1
	if _f == 90:
		root.get_viewport().get_texture().get_image().save_png(OS.get_cmdline_user_args()[0])
		return true
	return false
