import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Function to display OpenCV images in Streamlit using Matplotlib
def st_cv2_imshow(image, format=".png"):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    st.pyplot(fig)

# Function to load and match images
def match_images(image1, image2):
    # Convert the uploaded files to OpenCV format
    image1_cv = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image2_cv = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Initialize keypoint detectors and descriptors
    detector = cv2.ORB_create()  # You can use other detectors such as SIFT or SURF
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(image1_cv, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2_cv, None)

    # Check if keypoints and descriptors are found
    if keypoints1 is None or keypoints2 is None or descriptors1 is None or descriptors2 is None:
        st.error("Error: Keypoints or descriptors not found.")
        return None

    # Match descriptors between the two images
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw only good matches, set a threshold for matching distances
    good_matches = [match for match in matches if match.distance < 50]

    # Check if good matches are found
    if not good_matches:
        st.error("Error: No good matches found.")
        return None

    # Draw the matches
    result_image = cv2.drawMatches(
        image1_cv, keypoints1, image2_cv, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Print the number of good matches
    st.write(f'Number of good matches: {len(good_matches)}')

    # Print the coordinates of matched keypoints
    for match in good_matches:
        kp1 = keypoints1[match.queryIdx].pt
        kp2 = keypoints2[match.trainIdx].pt
        st.write(f'Matched keypoints: {kp1} -> {kp2}')

    return result_image

# Streamlit app
def main():
    st.title("Image Matching with Keypoints")

    # File uploader for the two images
    image1 = st.file_uploader("Improvized Dall.E Image", type=["png", "jpg", "jpeg"])
    image2 = st.file_uploader("Original Image", type=["png", "jpg", "jpeg"])

    # Check if both images are uploaded
    if image1 and image2:
        # Display the uploaded images
        st.image(image1, caption="Improvized Dall.E", use_column_width=True)
        st.image(image2, caption="Real Image", use_column_width=True)

        # Match images when the user clicks the button
        if st.button("Match Images"):
            # Perform matching and get the result image
            result_image = match_images(image1, image2)

            # Display the result image
            if result_image is not None:
                st_cv2_imshow(result_image, format=".png")

# Run the app
if __name__ == "__main__":
    main()
