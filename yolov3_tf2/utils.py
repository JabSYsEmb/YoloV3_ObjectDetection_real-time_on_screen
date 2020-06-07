from absl import logging
import numpy as np
import tensorflow as tf
from scipy import ndimage
import pandas as pd 
import cv2
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    
    boxes, objectness, classes, nums = outputs

    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    wh = np.flip(img.shape[0:2]) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='./data/fonts/futur.ttf', size=(img.size[0] + img.size[1]) // 150)

    # creating an empty array for saving coordinates of objects for the colour classification process
    coordinates = np.zeros((nums,4))
    
    for i in range(nums):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
       
        obj_color = tuple(color)       

        if(class_names[int(classes[i])] == "cup") or ( class_names[int(classes[i])] == "bottle" ):
            for t in np.linspace(0, 1, thickness):
                x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
                x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            # coordinates of each object [y1 --> y2 , x1 --> x2]
            coordinates[i] = [ x1y1[1], x2y2[1], x1y1[0], x2y2[0] ] 
            
            obj_color = tuple(color)

            # pre-checking before entering an image in colour_checker to color the frame with colour of its obejct
            # Entering an Image with 0 width or height (0xp * X) causes an error
            precent = None
            ColorName = ""
            if(coordinates[i][0]>0 and coordinates[i][1]>0 and coordinates[i][2]>0 and coordinates[i][3]>0):
                ColorName,precent,obj_color = Frame_coloring(img,coordinates[i],tuple(color))
            else:
                obj_color = tuple(color)

            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline=obj_color,width=5)

            confidence = '{:.2f}%'.format(objectness[i]*100)
            
            if(class_names[int(classes[i])] == "cup"):
                #text = '{}\n{} {}'.format(ColorName,("Tasse"),confidence)
                text = '{} {}'.format(("Tasse"),confidence)
                text_size = draw.textsize(text, font=font)

            elif(( class_names[int(classes[i])] == "bottle" )):
                #text = '{}\n{} {} fuellstand = {}%'.format(ColorName,("Flasche"),confidence, precent)
                text = '{} {} fuellstand = {}%'.format(("Flasche"),confidence, precent)
                text_size = draw.textsize(text, font=font)
            else:
                text = ""
                text_size = draw.textsize(text, font=font)

            draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                            fill="black")
            draw.text((x0, y0 - text_size[1]), text, fill="white",
                              font=font,spacing=4,align='center')

    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)
    img = img_np
    #img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    coordinates = coordinates.astype(int)
    return img, coordinates, nums


def draw_labels(x, y, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 0), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
                
def Frame_coloring(img,coordinate,color = "yellow"):

    index=["color","color_name","hex","R","G","B"]
    csv = pd.read_csv('./yolov3_tf2/Farbe.csv', names=index, header=None,encoding ="ISO-8859-1")

    coordinate = np.array((coordinate[0],coordinate[1],coordinate[2],coordinate[3])).astype(int)
    
    img = np.array(img)
    img = img[coordinate[0]:coordinate[1],coordinate[2]:coordinate[3]]
    x = int((coordinate[3]-coordinate[2])/2)
    y = int((coordinate[1]-coordinate[0])/2 + (coordinate[1]-coordinate[0])/6)

    b = img[y,x,2]
    g = img[y,x,1]
    r = img[y,x,0]
    b = int(b)
    g = int(g)
    r = int(r)

    colour = getColorName(r,g,b,csv)
    precent = liquid_level(img)
    chex = getColorHex(r,g,b,csv)


    return colour,precent,chex

def getColorHex(R,G,B,csv):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            Hex = csv.loc[i,"hex"]
    return Hex

def getColorName(B,G,R,csv):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

def liquid_level(img):
    precent_1 = 0
    img_x = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    M = np.mean(img_x)
    S = img_x.std()
    pixel_count = img_x.shape[0]*img_x.shape[1]

    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 0, 0],
                             [ 0, -1, 0 ],
                             [ 0,0, 1 ]] )

    vertical = ndimage.convolve( img_x, roberts_cross_v )
    horizontal = ndimage.convolve( img_x, roberts_cross_h )

    output_image = np.sqrt( np.square(horizontal) + np.square(vertical))

    output_image_count = np.count_nonzero(output_image)

    precent_1 = ((output_image_count*100)/(pixel_count+0.3*pixel_count))
    
    return int(precent_1)