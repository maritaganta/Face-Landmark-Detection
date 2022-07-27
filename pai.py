# Modified by Augmented Startups 2021
# Face Landmark User Interface with StreamLit
# Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from utils import *

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = '../demo.mp4'
DEMO_IMAGE = '../demo.jpg'
## Application Title

## Will allow us to create the interface (sidebar etc)
## I write <style> and follow with data-testid as sidebar etc and close style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
## this is to actually add the sidebar with titles and subheaders
st.sidebar.title('PandionAI')
st.sidebar.subheader('Actions')


# resize image to not have it stretched across the whole screen
@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):  # cv2 interpolation (good for images)
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# create drop-down menu to have different app modes (instructions, [modes])
app_mode = st.sidebar.selectbox("Take a look at AlertSat's capabilities",
                                ['AlertSat Portal', 'About PandionAI', 'Run on Image', 'Run on Video']
                                )
# define each mode's functionality
if app_mode == 'About PandionAI':

    st.title('Securing your need for time information')

    st.markdown(
        'Up-to-date and swift information for efficient and credible decision-making')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.video('https://www.youtube.com/watch?v=DutNs97rDXc')

    st.markdown('''
          # About Us \n 
            With our satellite system AlertSat, we combine the capacity and agility of a smart constellation, with the strength and speed of edge-based AI change detection of the images in orbit.
            
            AlertSat gives you maximum information response at a minimal cost. \n

            Also check us out on [LinkedIn](https://www.linkedin.com/company/pandionai/?viewAsMember=true)

            Get in touch!
            ''')
elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # max faces
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=max_faces) as face_mesh:
        prevTime = 0

        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                # st.checkbox("Recording", value=True)
                out.write(frame)
            # Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()

elif app_mode == 'Run on Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')
    # very easily add what we want in the side bar
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0
    # Dashboard
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            # print('face_landmarks:', face_landmarks)

            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            ## write our kpis as we like
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)
# Watch Tutorial at www.augmentedstartups.info/YouTube

elif app_mode == 'AlertSat Portal':
    st.title('AlertSat Portal')

    c1, c2 = st.columns(2)

    # get and cache data from API
    parks = get_data()

    # layout map
    with c1:
        """(_Click on a pin to bring up more information_)"""
        m = folium.Map(location=[39.949610, -75.150282], zoom_start=4)

        for park in parks:
            popup = folium.Popup(f"""
                      <a href="{park["url"]}" target="_blank">{park["fullName"]}</a><br>
                      <br>
                      {park["operatingHours"][0]["description"]}<br>
                      <br>
                      Phone: {park["contacts"]["phoneNumbers"][0]["phoneNumber"]}<br>
                      """,
                                 max_width=250)
            folium.Marker(
                [park["latitude"], park["longitude"]], popup=popup
            ).add_to(m)

        map_data = st_folium(m, key="fig1", width=700, height=700)

    # get data from map for further processing
    map_bounds = Bounds.from_dict(map_data["bounds"])

    # when a point is clicked, display additional information about the park
    try:
        point_clicked: Optional[Point] = Point.from_dict(map_data["last_object_clicked"])

        if point_clicked is not None:
            with st.spinner(text="loading image..."):
                for park in parks:
                    if park["_point"].is_close_to(point_clicked):
                        with c2:
                            f"""### _{park["fullName"]}_"""
                            park["description"]
                            st.image(park["images"][0]["url"], caption=park["images"][0]["caption"])
                            st.expander("Show park full details").write(park)
    except TypeError:
        point_clicked = None

    # even though there is a c1 reference above, we can do it again
    # output will get appended after original content
    with c1:

        parks_in_view: List[Dict] = []
        for park in parks:
            if map_bounds.contains_point(park["_point"]):
                parks_in_view.append(park)

        "Parks visible:", len(parks_in_view)
        "Bounding box:", map_bounds

