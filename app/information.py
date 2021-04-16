def website_info(st):
    t = """This model will recreate the edge of images. Select the side of the image that you wish
    to generate from the drop-down menu and either select one of the examples or upload your own image,
    and the model will show you the image with the edge replaced. \n
    This tool can be used to recreate the original content of the edge of an image, if there has been content 
    intentionally added onto it (such as text or a frame) or if the side of the image has somehow become 
    corrupted. It can also extend pictures past their original bounds, to show what may be lying outside the original image."""
    return st.text(t)


def model_info(st):
    t = """The model is based on the Pix2Pix model, which can be found here:
    \nhttps://paperswithcode.com/paper/image-to-image-translation-with-conditional\n
     The model is a convolutional neural network that operates through a series of 'down layers',
     which condense the dimensions of the image, followed up a series of 'up layers', which combine
     the condensed input data with the output of the down layers to expand the image back to the
     original dimensions. The model also includes a generator and discriminator which are trained
     adversarially. As the model trains, the discriminator learns to detect the differences between
     real-world images and generated images, and the generator learns to produce images that are more
     difficult for the discriminator to differentiate from reality. To this end, when we tell the model
     to predict the side of an image, the model is actually predicting the whole image, but we 'hide' only
     the portion of the image that we're trying to predict, so the model often is able to predict that
     section reliably.
     """
    return st.text(t)