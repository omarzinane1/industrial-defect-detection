import cv2

def detect_contours(binary_img, retrieval_mode=cv2.RETR_EXTERNAL, approx_method=cv2.CHAIN_APPROX_SIMPLE):
    contours, hierarchy = cv2.findContours(binary_img.copy(), retrieval_mode, approx_method)
    return contours, hierarchy

def filter_contours_by_area(contours, min_area=20):
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

def draw_contours_on_image(gray_img, contours, thickness=2):
    img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_rgb, contours, -1, (255, 0, 0), thickness)
    return img_rgb

def draw_bounding_boxes(gray_img, contours, thickness=2, min_area=10):
    img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        H, W = gray_img.shape

        if w > 0.95 * W and h > 0.95 * H:
            continue

        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), thickness)

    return img_rgb