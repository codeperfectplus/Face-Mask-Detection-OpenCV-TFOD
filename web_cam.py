import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    print(prediction_dict)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


def run(label_map_path, config_file_path, checkpoint_path):

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(config_file_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint_path).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

    cap = cv2.VideoCapture(0)
    while True:
        # Read frame from camera
        ret, image_np = cap.read()

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor, detection_model)

        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy()
        detection_scores = detections['detection_scores'][0].numpy()

        indexes = np.array(tf.image.non_max_suppression(
            detection_boxes,
            detection_scores,
            max_output_size=100,
            iou_threshold=0.5,
            score_threshold=0.3))

        detection_boxes = detection_boxes[indexes]
        detection_classes = detection_classes[indexes]
        detection_scores = detection_scores[indexes]

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        if indexes.shape != 0:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detection_boxes,
                (detection_classes + label_id_offset).astype(int),
                detection_scores,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=.30,
                agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    PATH_TO_LABELMAP = r"model/label_map.pbtxt"
    PATH_TO_CONFIG = r"model/pipeline.config"
    PATH_TO_CHECKPOINT = r"model/checkpoint/ckpt-18"

    run(label_map_path=PATH_TO_LABELMAP, config_file_path=PATH_TO_CONFIG, checkpoint_path=PATH_TO_CHECKPOINT)
