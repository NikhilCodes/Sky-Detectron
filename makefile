tflite:
	tflite_convert --saved_model_dir model_dir/1/ --output_file="tflite_saved_model/sky-detectron.tflite"

run:
	python camera_app.py
