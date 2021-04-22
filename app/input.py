def input_type_radio(st):
    return st.radio('Choose an input method', options=['Select Image', 'Upload Image'])

def input_select_images(st, files):
    return st.selectbox('Select an image', tuple(files))

def input_upload_images(st):
    return st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def input_mask_direction(st):
    return st.selectbox('Choose mask orientation', ('top', 'right', 'down', 'left'))