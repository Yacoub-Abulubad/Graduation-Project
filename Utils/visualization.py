import visualkeras
from PIL import ImageFont
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D, Rescaling, Normalization, Activation, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, Multiply, Add, BatchNormalization
from collections import defaultdict

def visualize_model(model, save_figure=False):
    color_map = defaultdict(dict)
    color_map[Conv2D]['fill'] = 'blue'
    color_map[Rescaling]['fill'] = 'gray'
    color_map[Activation]['fill'] = 'pink'
    color_map[MaxPooling2D]['fill'] = 'orange'
    color_map[Dense]['fill'] = 'black'
    color_map[DepthwiseConv2D]['fill'] = 'green'
    color_map[Reshape]['fill'] = 'teal'
    color_map[Add]['fill'] = 'yellow'
    color_map[Multiply]['fill'] = 'red'

    font = ImageFont.truetype("arial.ttf", 32)
    if not save_figure:
        visualkeras.layered_view(model.model, legend=True, font=font, spacing=0, max_z=10, color_map=color_map, type_ignore=[ZeroPadding2D, Dropout, Flatten]).show()
    else:
        visualkeras.layered_view(model.model, legend=True, font=font, spacing=0, max_z=10, color_map=color_map, type_ignore=[ZeroPadding2D, Dropout, Flatten], to_file='Figures/model_out.png').show()
    