import numpy as np, os
import matplotlib.pyplot as plt

def figure_to_image(fig):
    """
    https://matplotlib.org/gallery/user_interfaces/canvasagg.html#sphx-glr-gallery-user-interfaces-canvasagg-py
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    return X
# figure_to_image

def view_images(data, view_ids = list(range(16)), 
                      rows = 4, cols = 4, figsize = (12, 8), 
                      is_show = True,
                      save_path = None):
    
    if view_ids is None or len(view_ids)==0:
        view_ids = np.random.choice(len(data), size = rows * cols)
    # if
    
    rows = int((len(view_ids) + cols - 1) / cols)

    fig = plt.figure(figsize=figsize)
    for row in range(rows):
        for col in range(cols):
            id_pos = row * cols + col
            if id_pos >= len(view_ids): continue
            if id_pos >= len(data): continue
            
            plt.subplot(rows, cols, row * cols + col + 1)
            
            shape_image = data[view_ids[id_pos]].shape
            if len(shape_image) == 2:
                plt.imshow(data[view_ids[id_pos]], cmap = 'gray')
            else:
                plt.imshow(data[view_ids[id_pos]])
            plt.axis("off")
        # for
    # for
    if is_show==True: plt.show()
        
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "" and os.path.exists(save_dir) == False: 
            print(save_dir)
            os.makedirs(save_dir)
        fig.savefig(save_path)
    # if
    del fig
    plt.close()
# view_images