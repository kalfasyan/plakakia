import streamlit as st
import cv2
import numpy as np
from natsort import natsorted
import os 

st.set_page_config(page_title="Plakakia Tiling Explorer", layout="centered")

st.title("Plakakia Tiling Explorer")

# Let the user select the image they want to use
image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if image is None: st.info("Please upload an image file."); st.stop()

image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
# Delete any existing image.png file and altered.png file
try: os.remove("image.png")
except: pass
try: os.remove("altered.png")
except: pass


with st.form(key='my_form', clear_on_submit=False):
    st.subheader("Crop & Resize Image")
    col1, col2 = st.columns(2)
    top_crop = col1.number_input("Top Crop %", 0, 100, 0)
    bottom_crop = col2.number_input("Bottom Crop %", 0, 100, 0)
    col1, col2 = st.columns(2)
    left_crop = col1.number_input("Left Crop", 0, 100, 0)
    right_crop = col2.number_input("Right Crop", 0, 100, 0)

    # Resize the image to a user selected size
    resize = st.slider('Resize', 0, 100, 100)

    # Select the tile size and step size
    col1, col2 = st.columns(2)
    with col1: tile_size = st.slider('Tile Size', 0, 1000, 50, step=5)
    with col2: step_size = st.slider('Step Size', 0, 1000, 50, step=5)

    # Add a submit button
    submit_button = st.form_submit_button(label='▶️ Run') 

if submit_button:
    # Crop the image
    image = image[:, int(image.shape[1] * (left_crop / 100.0)):int(image.shape[1] * (1 - (right_crop / 100.0))), :]
    # Crop the image
    image = image[int(image.shape[0] * (top_crop / 100.0)):int(image.shape[0] * (1 - (bottom_crop / 100.0))), :, :]
    image = cv2.resize(image, (0, 0), fx=(resize / 100.0), fy=(resize / 100.0))
    cv2.imwrite("image.png", image)

    try:
        from plakakia.tiling import tile_image
        tiles, coordinates = tile_image(image, tile_size, step_size)
    except Exception as e:
        st.error("Error: " + str(e))
        st.stop()

    # Draw a rectangle around each tile and make it smaller by 1 pixel to see the grid
    tile_size = int(tile_size)
    step_size = int(step_size)
    for i in range(len(coordinates)):
        x1, y1, x2, y2 = coordinates[i]
        # Make the color the inverse of the average color of the tile
        inv_color = (255 - int(np.average(tiles[i, :, :, 0])), 255 - int(np.average(tiles[i, :, :, 1])), 255 - int(np.average(tiles[i, :, :, 2])))
        # Make the rectangle 1 pixel smaller to see the grid
        cv2.rectangle(image, (x1, y1), (x2-1, y2-1), inv_color, 1)
    st.write("Image size: ", image.shape)

    if image is not None:
        cv2.imwrite("altered.png", image) 
    else:
        st.error("Please initialize")    

    if image is not None:
        st.image("altered.png", use_column_width=True)
        
    else: 
        st.stop()
