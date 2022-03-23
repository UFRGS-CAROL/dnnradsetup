import numpy as np
import pandas as pd

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


def compute_iou(ground_truth_box, detection_box):
    g_y_min, g_x_min, g_y_max, g_x_max = tuple(ground_truth_box.tolist())
    d_y_min, d_x_min, d_y_max, d_x_max = tuple(detection_box.tolist())

    xa = max(g_x_min, d_x_min)
    ya = max(g_y_min, d_y_min)
    xb = min(g_x_max, d_x_max)
    yb = min(g_y_max, d_y_max)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    box_a_area = (g_x_max - g_x_min + 1) * (g_y_max - g_y_min + 1)
    box_b_area = (d_x_max - d_x_min + 1) * (d_y_max - d_y_min + 1)
    return intersection / float(box_a_area + box_b_area - intersection)


def process_detections(categories, ground_truth, detections):
    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

    matches = []
    ground_truth_boxes = ground_truth["detection_boxes"]
    ground_truth_classes = ground_truth['detection_classes']
    detection_scores = detections['detection_scores'][0].numpy()
    detection_boxes = detections['detection_boxes'][0].numpy()[detection_scores >= CONFIDENCE_THRESHOLD]
    detection_classes = detections['detection_classes'][0].numpy()[detection_scores >= CONFIDENCE_THRESHOLD].astype(
        'uint8')
    for i, ground_truth_box in enumerate(ground_truth_boxes):
        for j, detection_box in enumerate(detection_boxes):
            iou = compute_iou(ground_truth_box, detection_box)
            if iou > IOU_THRESHOLD:
                matches.append([i, j, iou])

    matches = np.array(matches)
    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending IOU. Removing duplicates doesn't preserve
        # our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    for i in range(len(ground_truth_boxes)):
        if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
            confusion_matrix[ground_truth_classes[i] - 1][
                int(detection_classes[int(matches[matches[:, 0] == i, 1][0])] - 1)] += 1
        else:
            confusion_matrix[ground_truth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1

    for i in range(len(detection_boxes)):
        if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
            confusion_matrix[confusion_matrix.shape[0] - 1][int(detection_classes[i] - 1)] += 1

    return confusion_matrix


def display(confusion_matrix, categories, output_path):
    """ Displays confusion matrix as pandas df to terminal and saves as CSV
    Args:
      confusion_matrix: matrix to be displayed
      categories: ordered array of class IDs
      output_path: where to save CSV
    """
    print('\nConfusion Matrix:')
    print(confusion_matrix, '\n')
    results = []

    for i in range(len(categories)):
        id = categories[i]['id'] - 1
        name = categories[i]['name']

        total_target = np.sum(confusion_matrix[id, :])
        total_predicted = np.sum(confusion_matrix[:, id])

        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)

        results.append(
            {'category': name, f'precision_@{IOU_THRESHOLD}IOU': precision, f'recall_@{IOU_THRESHOLD}IOU': recall})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_path)


def verify_classification_accuracy(predictions: list, ground_truth_csv: str):
    if ground_truth_csv is None:
        return
    ground_truth_df = pd.read_csv(ground_truth_csv)
    predictions_df = pd.DataFrame(predictions)
    predictions_df["img_name"] = predictions_df["img_name"].str.replace(r".JPEG", "", regex=True)
    merged = pd.merge(ground_truth_df, predictions_df, on="img_name")
    assert merged.shape[0] == ground_truth_df.shape[0], "Incorrect merged"
    merged["accuracy"] = merged["class_id"] == merged["class_id_predicted"]
    img_n = ground_truth_df.shape[0]
    accuracy = merged["accuracy"].value_counts() / img_n
    print("Accuracy measured on the input subset:")
    print(f" - Correct predicted: {accuracy[True] * 100:.2f}%")
    print(f" - Incorrect predicted: {accuracy[False] * 100:.2f}%")


def verify_detection_accuracy(predictions: list, ground_truth_csv: str):
    pass
