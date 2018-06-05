from styx_msgs.msg import TrafficLight
import numpy as np
import cv2
import tensorflow as tf
import rospy
import os.path

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.session = None
        self.input_layer_tensor = None
        self.logits_tensor = None
        
        self.session = tf.Session()
        
        cwd = os.path.dirname(os.path.realpath(__file__))
        saver = tf.train.import_meta_graph(cwd + '/my_traffic_light_model-25.meta')
        saver.restore(self.session, tf.train.latest_checkpoint(cwd + '/'))

        graph = tf.get_default_graph()
        self.input_layer_tensor = graph.get_tensor_by_name('input_layer:0')
        self.logits_tensor = graph.get_operation_by_name('logits/BiasAdd').outputs[0]
            
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if self.session is None or self.input_layer_tensor is None or self.logits_tensor is None:
            rospy.logerr("Classifier: variables not initialized")
            return TrafficLight.UNKNOWN
        else:
            image = cv2.resize(image, (800, 600))
            image = image[0:300,:,:]

            hls = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2HLS), (125,50))
            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (125,50))

            s_ch = hls[:,:,2]

            final_image = np.dstack([s_ch, gray])

            #output_class = self.session.run(tf.argmax(self.logits_tensor, 1), feed_dict = {self.input_layer_tensor: [final_image]})
            pred, output_class = self.session.run([tf.nn.softmax(self.logits_tensor), tf.argmax(self.logits_tensor, 1)], feed_dict = {self.input_layer_tensor: [final_image]})

            if output_class == 0:
                #rospy.logerr("Classifier: red light detected with confidence %s", pred[0,output_class][0]*100)
                return TrafficLight.RED
            '''
            elif output_class == 1:
                rospy.logwarn("Classifier: light detected but not red")
            elif output_class == 2:
                rospy.logwarn("Classifier: no light")
            else:
                rospy.logwarn("Classifier: I dont know!")
            '''
            return TrafficLight.UNKNOWN
