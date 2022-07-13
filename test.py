import sys
import tensorflow as tf
from model import Model
import numpy as np
import cv2
from PIL import Image
import conf

num2str = {
    0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9", 8: "A",
    9: "B", 10: "C", 11: "D", 12: "E", 13: "F", 14: "G", 15: "H", 16: "J",
    17: "K", 18: "L", 19: "M", 20: "N", 21: "P", 22: "Q", 23: "R", 24: "S",
    25: "T", 26: "U", 27: "V", 28: "W", 29: "X", 30: "Y"
}


def test(model, sess):
    result = tf.argmax(model.prediction, 1)
    f = open(conf.MAPPINGS, "w")

    for i in range(conf.TEST_NUMBER):
        # Open images.
        name = str(i).zfill(4)
        path = conf.TEST_IMAGE_PATH+"/"+name+".jpg"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Cut the captcha into four characters.
        cut_list = [image[:, 5:61], image[:, 52:108], image[:, 101:157], image[:, 144:200]]

        # Image preprocessing.
        for j in range(4):
            _, image = cv2.threshold(cut_list[j], 180, 255, cv2.THRESH_BINARY)
            image = Image.fromarray(image)
            image = image.resize((28, 40))
            image = np.array(image)/255.0
            cut_list[j] = image.reshape([28, 40, 1])

        # Get predictions.
        nums = sess.run(result, feed_dict={model.X: cut_list, model.keep_prob: 1.0})
        prediction_list = [num2str[num] for num in nums]

        # Write predictions into mappings.txt.
        f.write(str(i).zfill(4)+",")
        f.write("".join(prediction_list)+"\n")
        sys.stdout.write('\r>> Testing image %d/%d'%(i+1, conf.TEST_NUMBER))
        sys.stdout.flush()


if __name__=='__main__':
    model = Model()
    saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, conf.MODEL_PATH)
    test(model, sess)
